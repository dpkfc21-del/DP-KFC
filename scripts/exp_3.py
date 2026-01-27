"""
Comprehensive experiment to validate noise preconditioning hypothesis.

5 EXPERIMENTAL SCENARIOS - Isolating Input Geometry (A) from Output Geometry (G):

Scenario 1: Ideal Transfer (MNIST → FashionMNIST)
    - Feature Alignment (A): HIGH (Centered, Sparse, Grayscale)
    - Task Alignment (G): HIGH (10 Classes, Easy)
    - Prediction: Public should win (Public ≈ Oracle)

Scenario 2: Texture Mismatch (MNIST → PathMNIST)
    - Feature Alignment (A): ORTHOGONAL (Object vs Texture)
    - Task Alignment (G): MEDIUM (Classification)
    - Prediction: Noise should win (Public A actively hurts)

Scenario 3: Task Shift (CIFAR-10 → CIFAR-100)
    - Feature Alignment (A): HIGH (Natural Images)
    - Task Alignment (G): LOW (10 vs 100 Classes)
    - Prediction: Hybrid should win (A_pub good, G_pub biased)

Scenario 4: Total Mismatch (MNIST → CIFAR-100)
    - Feature Alignment (A): PARTIAL (Spatial prior helps)
    - Task Alignment (G): VERY LOW (Easy vs Hard)
    - Prediction: Hybrid should win (A_pub helps spatially, G_pub hurts)

Scenario 5: MedMNIST-Global "Checkmate" (MNIST → MedMNIST-Global)
    - Feature Alignment (A): ORTHOGONAL (Center-biased digits vs Mixed Texture+Structure)
    - Task Alignment (G): VERY LOW (10 easy classes vs 47 hard medical classes)
    - Prediction: Pink Noise should win (translation invariant + spatial correlation)
    - This is the ultimate test: HARD task + NO center bias + Mixed domains

KEY INSIGHT:
- "Mixing" via averaging (0.5*A1 + 0.5*A2) != "Hybridizing" (A_pub ⊗ G_noise)
- The hybrid uses Public data for ROTATION (A matrix) and Noise for SCALING (G matrix)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, TensorDataset
from torchvision import datasets, transforms
from opacus import GradSampleModule
from opacus.accountants.utils import get_noise_multiplier
import numpy as np
import random
import medmnist
import timm
import csv
from datetime import datetime

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ==========================================
# Scenario Definitions
# ==========================================

SCENARIOS = {
    "1_DistMatch": {
        "name": "Ideal Transfer",
        "private": "fashionmnist",
        "public": "mnist",
        "img_size": 28,
        "classes": 10,
        "description": "MNIST → FashionMNIST: High A alignment, High G alignment",
        "prediction": "Public should win (Public ≈ Oracle)"
    },
    "2_DistDisjoint": {
        "name": "Texture Mismatch",
        "private": "pathmnist",
        "public": "mnist",
        "img_size": 28,
        "classes": 9,
        "description": "MNIST → PathMNIST: Orthogonal A (Object vs Texture), Medium G",
        "prediction": "Noise should win (Public A actively hurts)"
    },
    "3_TaskHarder": {
        "name": "Task Shift",
        "private": "cifar100",
        "public": "cifar10",
        "img_size": 32,
        "classes": 100,
        "description": "CIFAR-10 → CIFAR-100: High A alignment, Low G alignment",
        "prediction": "Hybrid should win (A_pub good, G_pub biased)"
    },
    "4_TotalMismatch": {
        "name": "Total Mismatch",
        "private": "cifar100",
        "public": "mnist",
        "img_size": 32,
        "classes": 100,
        "description": "MNIST → CIFAR-100: Partial A (spatial prior), Very Low G",
        "prediction": "Hybrid should win (A_pub spatial help, G_pub hurts)"
    },
    "5_MedGlobal": {
        "name": "MedMNIST-Global (Checkmate)",
        "private": "medmnist_global",
        "public": "mnist",
        "img_size": 28,
        "classes": 47,  # Will be dynamically set
        "description": "MNIST → MedMNIST-Global: HARD test - Mixed Texture+Structure, No Center Bias",
        "prediction": "Pink Noise should win (translation invariant + spatial correlation)"
    }
}



ALL_PRECOND_TYPES = {
    # Baselines
    "Oracle (Private)": "oracle",
    "Identity (No Precond)": "identity",

    # === 6 Meaningful A ⊗ G Combinations ===

    # A from Public Images (captures public data geometry)
    "A_pub ⊗ G_pub_pub": "A_pub__G_pub_pub",       # Full public (original "public")
    "A_pub ⊗ G_pub_noise": "A_pub__G_pub_noise",   # Public geometry, no task supervision
    "A_pub ⊗ G_pink_noise": "A_pub__G_pink_noise", # Public A, noise G (original hybrid)

    # A from Pink Noise (captures natural image statistics without dataset bias)
    "A_pink ⊗ G_pub_pub": "A_pink__G_pub_pub",     # Pink A, full public G
    "A_pink ⊗ G_pub_noise": "A_pink__G_pub_noise", # Pink A, public geometry G
    "A_pink ⊗ G_pink_noise": "A_pink__G_pink_noise", # Full pink noise (original "pink_noise")

    "A_syn ⊗ G_syn (Clustered Pink)": "A_syn__G_syn_clustered",
}


# ==========================================
# Configuration
# ==========================================
ACTIVE_SCENARIO = "4_TotalMismatch"  # Change this or set to 'all'
ACTIVE_PRECOND_TYPES = 'all' # Change this or set to a list like ['oracle', 'A_pub__G_pub_pub']

SEEDS = [42] #, 123, 456]  # Run with multiple seeds for robustness
BATCH_SIZE = 1024
EPOCHS = 5
LEARNING_RATE = 1e-2
EPSILON = 2.0
DELTA = 1e-5
PRECOND_STEPS = 10
CLIP_NORM = 1.0
NUM_WORKERS = 16

PRECOND_UPDATE_EVERY = 1  # e.g., 0 = once at start, 1 = every epoch, 5 = every 5 epochs

# Model options: 'simple_cnn', 'crossvit', 'convnext', 'auto'
# 'auto' selects based on dataset: SimpleCNN for MNIST-like, pretrained for CIFAR/MedMNIST
# 'convnext' is faster than 'crossvit' with competitive accuracy
MODEL_TYPE = 'convnext'
CROSSVIT_IMG_SIZE = 240
CONVNEXT_IMG_SIZE = 224  # ConvNeXt works well at 224x224

# ImageNet normalization (for CrossViT pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Standard normalization (for SimpleCNN)
STANDARD_MEAN = [0.5, 0.5, 0.5]
STANDARD_STD = [0.5, 0.5, 0.5]

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10, img_size=28):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        fc_size = img_size // 4
        self.fc1 = nn.Linear(32 * fc_size * fc_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class CrossViTClassifier(nn.Module):

    def __init__(self, num_classes=100, img_size=240, hidden_dim=512):
        super().__init__()

        self.backbone = timm.create_model(
            "crossvit_tiny_240", pretrained=True, num_classes=0
        )

        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get the feature dimension automatically
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]

        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)

        logits = self.classifier(features)
        return logits


class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt-based classifier with frozen backbone.

    ConvNeXt is a modern CNN architecture that:
    - Is competitive with Vision Transformers in accuracy
    - Trains significantly faster than ViTs (pure convolutions)
    - Has better inductive biases for smaller datasets
    - Works well with standard 224x224 images

    We use convnext_tiny for efficiency, but larger variants are available.
    """

    def __init__(self, num_classes=100, img_size=224):
        super().__init__()

        # Load pretrained ConvNeXt-Tiny (smallest variant, ~28M params)
        # Other options: convnext_small, convnext_base, convnext_large
        self.backbone = timm.create_model(
            "convnext_tiny.fb_in22k_ft_in1k", pretrained=True, num_classes=0
        )

        # Freeze backbone
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Get the feature dimension automatically
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, img_size, img_size)
            features = self.backbone(dummy_input)
            self.feature_dim = features.shape[1]

        # Trainable classifier head
        self.classifier = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)

        logits = self.classifier(features)
        return logits

def get_model(num_classes, img_size, private_dataset):
    """
    Get model based on MODEL_TYPE configuration.

    MODEL_TYPE options:
    - 'simple_cnn': Always use SimpleCNN (fast, no pretrained weights)
    - 'crossvit': Always use CrossViT (ViT-based, slower but powerful)
    - 'convnext': Always use ConvNeXt (CNN-based, fast and powerful)
    - 'auto': Select based on dataset complexity
             SimpleCNN for MNIST-like, pretrained model for CIFAR/MedMNIST

    Returns:
        (model, actual_img_size, model_type_str)
    """
    # Determine which model to use
    if MODEL_TYPE == 'simple_cnn':
        model_choice = 'simple_cnn'
    elif MODEL_TYPE == 'crossvit':
        model_choice = 'crossvit'
    elif MODEL_TYPE == 'convnext':
        model_choice = 'convnext'
    elif MODEL_TYPE == 'auto':
        # Auto-select based on dataset complexity
        if private_dataset in ['cifar100', 'medmnist_global']:
            model_choice = 'convnext'  # Default to ConvNeXt for complex datasets
        else:
            model_choice = 'simple_cnn'
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {MODEL_TYPE}")

    # Create the model
    if model_choice == 'crossvit':
        print(f"    Using CrossViT (pretrained, frozen backbone) for {private_dataset}")
        model = CrossViTClassifier(num_classes=num_classes, img_size=CROSSVIT_IMG_SIZE)
        actual_img_size = CROSSVIT_IMG_SIZE
        model_type_str = 'crossvit'

    elif model_choice == 'convnext':
        print(f"    Using ConvNeXt (pretrained, frozen backbone) for {private_dataset}")
        model = ConvNeXtClassifier(num_classes=num_classes, img_size=CONVNEXT_IMG_SIZE)
        actual_img_size = CONVNEXT_IMG_SIZE
        model_type_str = 'convnext'

    else:  # simple_cnn
        print(f"    Using SimpleCNN for {private_dataset}")
        model = SimpleCNN(num_classes=num_classes, img_size=img_size)
        actual_img_size = img_size
        model_type_str = 'simple_cnn'

    return model, actual_img_size, model_type_str

def get_mnist_loaders(batch_size, img_size=28, use_imagenet_norm=False):
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if use_imagenet_norm else (STANDARD_MEAN, STANDARD_STD)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_ds = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=use_imagenet_norm)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=NUM_WORKERS, pin_memory=use_imagenet_norm)
    return train_loader, test_loader, len(train_ds)


def get_fashionmnist_loaders(batch_size, img_size=28, use_imagenet_norm=False):
    """FashionMNIST - clothing items (grayscale → 3-channel)"""
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if use_imagenet_norm else (STANDARD_MEAN, STANDARD_STD)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_ds = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
    test_ds = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, drop_last=True, pin_memory=use_imagenet_norm)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=NUM_WORKERS, pin_memory=use_imagenet_norm)
    return train_loader, test_loader, len(train_ds)


def get_pathmnist_loaders(batch_size, img_size=28, use_imagenet_norm=False):
    """PathMNIST - medical tissue pathology (RGB)"""
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if use_imagenet_norm else (STANDARD_MEAN, STANDARD_STD)
    num_workers = 4 if use_imagenet_norm else 2
    info = medmnist.INFO['pathmnist']
    DataClass = getattr(medmnist, info['python_class'])
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_ds = DataClass(split='train', transform=transform, download=True, root='./data')
    test_ds = DataClass(split='test', transform=transform, download=True, root='./data')
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=use_imagenet_norm)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=num_workers, pin_memory=use_imagenet_norm)
    return train_loader, test_loader, len(train_ds)


def get_cifar10_loaders(batch_size, img_size=32, use_imagenet_norm=False):
    """CIFAR-10 - natural images (10 classes)"""
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if use_imagenet_norm else (STANDARD_MEAN, STANDARD_STD)
    num_workers = 4 if use_imagenet_norm else 2
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_ds = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=use_imagenet_norm)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=num_workers, pin_memory=use_imagenet_norm)
    return train_loader, test_loader, len(train_ds)


def get_cifar100_loaders(batch_size, img_size=32, use_imagenet_norm=False):
    """CIFAR-100 - natural images (100 fine-grained classes)"""
    mean, std = (IMAGENET_MEAN, IMAGENET_STD) if use_imagenet_norm else (STANDARD_MEAN, STANDARD_STD)
    num_workers = 4 if use_imagenet_norm else 2
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    train_ds = datasets.CIFAR100('./data', train=True, download=True, transform=transform)
    test_ds = datasets.CIFAR100('./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=use_imagenet_norm)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=num_workers, pin_memory=use_imagenet_norm)
    return train_loader, test_loader, len(train_ds)


class MedMNISTSubset(torch.utils.data.Dataset):
    """
    Wrapper for a MedMNIST dataset that applies label offset and transforms on-the-fly.
    This avoids pre-loading all images into memory (crucial for large img_size like 240).
    """
    def __init__(self, medmnist_ds, label_offset, transform, max_samples=None):
        self.ds = medmnist_ds
        self.label_offset = label_offset
        self.transform = transform
        # Limit to max_samples if specified (use first N samples)
        self.max_samples = min(max_samples, len(medmnist_ds)) if max_samples else len(medmnist_ds)

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        if idx >= self.max_samples:
            raise IndexError(f"Index {idx} out of range for dataset of size {self.max_samples}")
        img, label = self.ds[idx]
        # img is a PIL Image, label is numpy array
        if self.transform:
            img = self.transform(img)
        label = int(label.item() if hasattr(label, 'item') else label) + self.label_offset
        return img, label


def get_medmnist_global_loaders(batch_size, img_size=28, use_imagenet_norm=False):
    """
    Constructs 'MedMNIST-Global' by merging multiple single-label 2D classification datasets.
    This creates a HARD (Many Class) Medical task - the ultimate "Checkmate" experiment.

    Why this dataset:
    - Total ~47 classes (HARD task like CIFAR-100)
    - 28x28 images (perfect match for MNIST public data)
    - Mixed Textures (Pathology, Tissue) + Structures (Organs, Cells)
    - NO center bias (unlike MNIST/CIFAR which have objects in center)

    This tests whether Pink Noise wins because it has:
    - Spatial correlation (good for structure images)
    - Translation invariance (good for texture images)
    - No dataset-specific bias
    """
    mean = IMAGENET_MEAN if use_imagenet_norm else STANDARD_MEAN
    std = IMAGENET_STD if use_imagenet_norm else STANDARD_STD
    num_workers = 4 if use_imagenet_norm else 2

    # List of datasets to merge (excluding multi-label ChestMNIST)
    flags = [
        'pathmnist',    # 9 classes (Texture - colon pathology)
        'bloodmnist',   # 8 classes (Cells - blood cells, somewhat centered)
        'dermamnist',   # 7 classes (Lesions - skin lesions)
        'octmnist',     # 4 classes (Retina - OCT scans, structural)
        'tissuemnist',  # 8 classes (Tissue - kidney cortex, texture)
        'organamnist',  # 11 classes (Organ - abdominal CT, structural)
    ]

    # Build transform that resizes on-the-fly (lazy, memory-efficient)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=3),  # Some MedMNIST are grayscale
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_datasets = []
    test_datasets = []

    current_label_offset = 0
    total_train = 0
    total_test = 0

    print(f"Constructing MedMNIST-Global from: {flags}")

    for flag in flags:
        info = medmnist.INFO[flag]
        n_classes = len(info['label'])
        DataClass = getattr(medmnist, info['python_class'])

        # Load datasets with PIL transform (default) - images loaded lazily
        train_ds = DataClass(split='train', transform=None, download=True, root='./data')
        test_ds = DataClass(split='test', transform=None, download=True, root='./data')

        # Use only half of each dataset to reduce total size (still ~200K samples)
        train_max = len(train_ds) // 2
        test_max = len(test_ds) // 2

        # Wrap with our custom dataset that applies transform + label offset + sample limit
        train_subset = MedMNISTSubset(train_ds, current_label_offset, transform, max_samples=train_max)
        test_subset = MedMNISTSubset(test_ds, current_label_offset, transform, max_samples=test_max)
        train_datasets.append(train_subset)
        test_datasets.append(test_subset)

        print(f"  + {flag}: {n_classes} classes (IDs {current_label_offset}-{current_label_offset+n_classes-1}), "
              f"train={len(train_subset)}/{len(train_ds)}, test={len(test_subset)}/{len(test_ds)}")

        total_train += len(train_subset)
        total_test += len(test_subset)
        current_label_offset += n_classes

    total_classes = current_label_offset
    print(f"Total MedMNIST-Global: {total_classes} classes")

    # Merge all datasets
    full_train_ds = ConcatDataset(train_datasets)
    full_test_ds = ConcatDataset(test_datasets)

    print(f"Total samples: train={len(full_train_ds)}, test={len(full_test_ds)}")

    train_loader = DataLoader(full_train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True, pin_memory=use_imagenet_norm)
    test_loader = DataLoader(full_test_ds, batch_size=256, num_workers=num_workers, pin_memory=use_imagenet_norm)

    # Return with total_classes so caller can update scenario config
    return train_loader, test_loader, len(full_train_ds), total_classes


def get_dataset_loaders(dataset_name, batch_size, img_size, use_imagenet_norm=False):
    """Universal loader dispatcher. Returns (train_loader, test_loader, train_len) or
    (train_loader, test_loader, train_len, num_classes) for medmnist_global."""
    if dataset_name == 'medmnist_global':
        # Special case: returns 4 values including dynamic num_classes
        return get_medmnist_global_loaders(batch_size, img_size, use_imagenet_norm)

    loaders = {
        'mnist': get_mnist_loaders,
        'fashionmnist': get_fashionmnist_loaders,
        'pathmnist': get_pathmnist_loaders,
        'cifar10': get_cifar10_loaders,
        'cifar100': get_cifar100_loaders,
    }
    if dataset_name not in loaders:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return loaders[dataset_name](batch_size, img_size, use_imagenet_norm)


def get_noise_batch(batch_size, img_size, device):
    """Pure Gaussian (White) noise - domain agnostic, uncorrelated pixels"""
    return torch.randn(batch_size, 3, img_size, img_size, device=device)


def get_pink_noise_batch(batch_size, img_size, device, alpha=1.0):
    """
    Generates 'Pink' noise (1/f decay) to mimic natural image statistics.

    White noise has uncorrelated pixels (A ≈ Identity matrix).
    Pink noise has spatially correlated pixels that mimic the power spectrum of natural images.

    The key insight:
    - Real images obey a Power Law: low frequencies (smooth patches) have high energy,
      high frequencies (edges/noise) have low energy.
    - Pink noise provides Rotation (spatial correlation) without object-specific bias.
    - It has Translation Invariance (no center bias), so it won't break PathMNIST like MNIST did.

    Args:
        batch_size: Number of images to generate
        img_size: Image dimensions (assumes square)
        device: torch device
        alpha: Spectral decay factor
               0.0 = White Noise (uncorrelated) - equivalent to get_noise_batch
               1.0 = Pink Noise (natural image-like)
               2.0 = Brownian Noise (very smooth/cloudy)

    Returns:
        Tensor of shape [batch_size, 3, img_size, img_size]
    """
    # 1. Generate White Noise in Frequency Domain (complex for phase)
    white_noise_freq = torch.randn(batch_size, 3, img_size, img_size, dtype=torch.cfloat, device=device)

    # 2. Create Frequency Grid (fx, fy)
    freqs = torch.fft.fftfreq(img_size, device=device)
    fx, fy = torch.meshgrid(freqs, freqs, indexing='ij')

    # Calculate magnitude of frequency f = sqrt(fx^2 + fy^2)
    f = torch.sqrt(fx**2 + fy**2)

    # Avoid division by zero at DC component (f=0)
    f[0, 0] = 1.0

    # 3. Apply 1/f^alpha scaling (scale amplitude)
    scale = 1.0 / (f ** alpha)
    scale[0, 0] = 0.0  # Zero out DC component (mean center the image)

    # Reshape scale to broadcast: [1, 1, H, W]
    scale = scale.view(1, 1, img_size, img_size)

    # 4. Filter the noise in frequency domain
    pink_noise_freq = white_noise_freq * scale

    # 5. Inverse FFT to get spatial domain
    pink_noise = torch.fft.ifft2(pink_noise_freq).real

    # 6. Normalize per-image (instance normalization) to ensure each sample has ~unit variance
    # This is important for K-FAC stability - batch-wise normalization could suppress
    # variance if one image has a spike, breaking the assumption that each x_i has ~unit variance
    std = pink_noise.view(batch_size, -1).std(dim=1, keepdim=True)
    pink_noise = pink_noise / (std.view(batch_size, 1, 1, 1) + 1e-8) * 0.5

    return pink_noise


class SyntheticPinkClusterTask:
    """
    Generates a synthetic classification task using Pink Noise Prototypes (GMM-style).

    This is the mathematically pure version of privacy-preserving preconditioning:
    - Gaussian Mixture Model where the "Mean" of each cluster is a Pink Noise image.
    - Jitter around centroids is ALSO Pink Noise (preserves spatial correlations).

    Why "Clustered Pink Noise" is the Ultimate Test:

    1. Input Geometry (A): The images have 1/f power spectrum (Pink Noise), so
       neighboring pixels are correlated. This keeps the A matrix non-trivial.

    2. Gradient Geometry (G): The data has clear, separable clusters. This creates
       a "Classification Manifold" where gradients point towards specific class
       centroids. This gives the G matrix structure.

    Key insight:
    - If you use White Noise for clusters → A becomes Identity (bad)
    - If you use Unclustered Pink Noise → G becomes Identity (bad)
    - Clustered Pink Noise → structured A AND structured G, entirely synthetically

    Privacy: Perfect. 0 bits of public data leakage because the "data" is
    procedurally generated noise.
    """

    def __init__(self, num_classes, img_size, device, jitter_scale=0.5):
        """
        Initialize the synthetic pink cluster task.

        Args:
            num_classes: Number of clusters/classes to generate
            img_size: Image dimensions (square)
            device: torch device
            jitter_scale: How much pink noise jitter to add around centroids (0.5 = moderate)
        """
        self.num_classes = num_classes
        self.img_size = img_size
        self.device = device
        self.jitter_scale = jitter_scale

        # Create the 'Platonic Ideals' (Class Centroids)
        # These are fixed for the entire experiment - each class has a unique pink noise pattern
        print(f"  [Synthetic] Generating {num_classes} Pink Noise Centroids...")
        self.centroids = get_pink_noise_batch(num_classes, img_size, device, alpha=1.0)

        # Normalize centroids to have distinct separation
        self.centroids = self.centroids / self.centroids.std()

    def get_batch(self, batch_size):
        """
        Generate a batch of synthetic clustered pink noise samples.

        Returns:
            inputs: Tensor of shape [batch_size, 3, img_size, img_size]
            labels: Tensor of shape [batch_size] with class indices
        """
        # 1. Sample Labels (which cluster each sample belongs to)
        labels = torch.randint(0, self.num_classes, (batch_size,), device=self.device)

        # 2. Get the corresponding Centroid for each sample
        # shape: [batch_size, 3, H, W]
        base_signals = self.centroids[labels]

        # 3. Generate Jitter (MUST BE PINK NOISE to preserve spatial correlations!)
        # If we added white noise, we would destroy the spatial correlations we care about.
        jitter = get_pink_noise_batch(batch_size, self.img_size, self.device, alpha=1.0)

        # 4. Combine: sample = centroid + scaled_jitter
        inputs = base_signals + (jitter * self.jitter_scale)

        # 5. Re-normalize per image to keep variance unit-like for K-FAC
        # (Inputs should generally be ~unit variance for neural nets)
        inputs = (inputs - inputs.mean(dim=(1, 2, 3), keepdim=True)) / (inputs.std(dim=(1, 2, 3), keepdim=True) + 1e-6)

        return inputs, labels


# ==========================================
# KFAC Components
# ==========================================

class KFACRecorder:
    def __init__(self, model, only_trainable=False):
        """
        Record activations and gradients for KFAC computation.

        Args:
            model: The model to record from
            only_trainable: If True, only register hooks for layers with requires_grad=True
                           (useful for CrossViT where only the classifier is trained)
        """
        self.activations = {}
        self.backprops = {}
        self.handles = []
        self.enabled = False

        target = model._module if hasattr(model, '_module') else model
        for name, module in target.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                # Skip frozen layers if only_trainable is True
                if only_trainable:
                    has_trainable = any(p.requires_grad for p in module.parameters())
                    if not has_trainable:
                        continue

                self.handles.append(
                    module.register_forward_hook(self._save_act(name))
                )
                self.handles.append(
                    module.register_full_backward_hook(self._save_grad(name))
                )

    def _save_act(self, name):
        def hook(mod, inp, out):
            if self.enabled:
                self.activations[name] = inp[0].detach()
        return hook

    def _save_grad(self, name):
        def hook(mod, grad_in, grad_out):
            if self.enabled:
                self.backprops[name] = grad_out[0].detach()
        return hook

    def enable(self): self.enabled = True
    def disable(self): self.enabled = False
    def clear(self):
        self.activations = {}
        self.backprops = {}


def compute_kfac_covs(model, recorder, data, labels, device, num_classes):
    """Compute KFAC covariance matrices from a single batch."""
    recorder.enable()
    model.zero_grad()

    data = data.to(device)
    labels = labels.to(device)

    out = model(data)
    loss = F.cross_entropy(out, labels)
    loss.backward()

    cov_A, cov_G = {}, {}
    eps = 1e-4

    target = model._module if hasattr(model, '_module') else model
    name_to_mod = dict(target.named_modules())

    for name in recorder.backprops:
        if name not in name_to_mod:
            continue
        mod = name_to_mod[name]

        if isinstance(mod, nn.Linear):
            A = recorder.activations[name]
            if A.dim() > 2:
                A = A.reshape(-1, A.shape[-1])
            cov_A[name] = (A.T @ A) / A.size(0) + eps * torch.eye(A.size(1), device=device)

            G = recorder.backprops[name]
            if G.dim() > 2:
                G = G.reshape(-1, G.shape[-1])
            cov_G[name] = (G.T @ G) / G.size(0) + eps * torch.eye(G.size(1), device=device)

        elif isinstance(mod, nn.Conv2d):
            X = recorder.activations[name]
            X_unfold = F.unfold(X, mod.kernel_size, padding=mod.padding, stride=mod.stride)
            X_unfold = X_unfold.transpose(1, 2).reshape(-1, X_unfold.size(1))
            cov_A[name] = (X_unfold.T @ X_unfold) / X_unfold.size(0) + eps * torch.eye(X_unfold.size(1), device=device)

            GY = recorder.backprops[name]
            GY = GY.permute(0, 2, 3, 1).reshape(-1, GY.size(1))
            cov_G[name] = (GY.T @ GY) / GY.size(0) + eps * torch.eye(GY.size(1), device=device)

    recorder.disable()
    recorder.clear()
    model.zero_grad()

    return cov_A, cov_G


def accumulate_covs(cov_list_A, cov_list_G):
    """Average multiple covariance estimates."""
    if not cov_list_A:
        return {}, {}

    avg_A, avg_G = {}, {}
    for name in cov_list_A[0]:
        avg_A[name] = sum(c[name] for c in cov_list_A) / len(cov_list_A)
    for name in cov_list_G[0]:
        avg_G[name] = sum(c[name] for c in cov_list_G) / len(cov_list_G)

    return avg_A, avg_G


def compute_inv_sqrt(cov_A, cov_G):
    """Compute inverse square root of covariance matrices."""
    inv_A, inv_G = {}, {}
    eps = 1e-3

    for name in cov_A:
        A = cov_A[name]
        eva, evc = torch.linalg.eigh(A + eps * torch.eye(A.size(0), device=A.device))
        inv_A[name] = evc @ torch.diag(eva.clamp(min=1e-6).rsqrt()) @ evc.T

        G = cov_G[name]
        evg, evcg = torch.linalg.eigh(G + eps * torch.eye(G.size(0), device=G.device))
        inv_G[name] = evcg @ torch.diag(evg.clamp(min=1e-6).rsqrt()) @ evcg.T

    return inv_A, inv_G


def precondition_grads(model, inv_A, inv_G):
    """Apply preconditioning to per-sample gradients."""
    target = model._module if hasattr(model, '_module') else model

    for name, mod in target.named_modules():
        if name not in inv_A or name not in inv_G:
            continue
        if not hasattr(mod.weight, 'grad_sample') or mod.weight.grad_sample is None:
            continue

        g_w = mod.weight.grad_sample

        if isinstance(mod, nn.Conv2d):
            bs, out_ch = g_w.shape[:2]
            g_w = g_w.view(bs, out_ch, -1)

        temp = torch.einsum("oj,bjk->bok", inv_G[name], g_w)
        new_w = torch.einsum("bok,ki->boi", temp, inv_A[name])

        if isinstance(mod, nn.Conv2d):
            new_w = new_w.view_as(mod.weight.grad_sample)

        mod.weight.grad_sample = new_w

        if mod.bias is not None and hasattr(mod.bias, 'grad_sample') and mod.bias.grad_sample is not None:
            g_b = mod.bias.grad_sample
            new_b = torch.einsum("oj,bj->bo", inv_G[name], g_b)
            mod.bias.grad_sample = new_b


def apply_dp(model, noise_mult, clip_norm, batch_size):
    """Apply DP-SGD: clip and add noise."""
    params = [p for p in model.parameters() if hasattr(p, 'grad_sample') and p.grad_sample is not None]
    if not params:
        return

    device = params[0].device

    norms_sq = torch.zeros(batch_size, device=device)
    for p in params:
        norms_sq += p.grad_sample.reshape(batch_size, -1).norm(2, dim=1) ** 2
    norms = norms_sq.sqrt()

    clip_factor = (clip_norm / (norms + 1e-6)).clamp(max=1.0)

    for p in params:
        flat = p.grad_sample.reshape(batch_size, -1)
        clipped = flat * clip_factor.unsqueeze(1)
        summed = clipped.sum(dim=0)
        noise = torch.randn_like(summed) * (noise_mult * clip_norm)
        p.grad = ((summed + noise) / batch_size).view_as(p)
        p.grad_sample = None

def compute_cov_A(model, recorder, img_source, steps, device, img_size, num_classes, public_iter=None):
    """
    Compute A (activation) covariances from an image source.

    A only depends on the forward pass (input images), not on labels.

    Args:
        img_source: 'pink' for pink noise, 'pub' for public data iterator
        public_iter: Iterator for public data (required if img_source='pub')
    """
    if hasattr(model, 'disable_hooks'):
        model.disable_hooks()

    cov_A_list = []

    for _ in range(steps):
        if img_source == 'pink':
            data = get_pink_noise_batch(BATCH_SIZE, img_size, device, alpha=1.0)
            # Labels don't matter for A, but we need them for the forward/backward pass
            labels = torch.randint(0, num_classes, (BATCH_SIZE,), device=device)
        elif img_source == 'pub':
            data, labels = next(public_iter)
            data = data.to(device)
            labels = labels.to(device).squeeze().long().clamp(max=num_classes - 1)
        else:
            raise ValueError(f"Unknown img_source for A: {img_source}")

        cov_A, _ = compute_kfac_covs(model, recorder, data, labels, device, num_classes)
        cov_A_list.append(cov_A)

    if hasattr(model, 'enable_hooks'):
        model.enable_hooks()

    # Average across steps
    avg_A = {}
    if cov_A_list:
        for name in cov_A_list[0]:
            avg_A[name] = sum(c[name] for c in cov_A_list) / len(cov_A_list)

    return avg_A


def compute_cov_G(model, recorder, img_source, label_source, steps, device, img_size, num_classes, public_iter=None):
    """
    Compute G (gradient) covariances from image and label sources.

    G depends on BOTH images (forward pass) AND labels (backward pass).

    Args:
        img_source: 'pink' for pink noise, 'pub' for public data images
        label_source: 'noise' for random labels, 'pub' for public data labels
        public_iter: Iterator for public data (required if img_source='pub' or label_source='pub')

    Valid combinations:
        - img='pub', label='pub':   Full public supervision (G_pub_pub)
        - img='pub', label='noise': Public images, random labels (G_pub_noise)
        - img='pink', label='noise': Pink noise images, random labels (G_pink_noise)

    Invalid:
        - img='pink', label='pub': Pink noise has no corresponding public labels!
    """
    if img_source == 'pink' and label_source == 'pub':
        raise ValueError("G_pink_pub is invalid: pink noise images have no corresponding public labels!")

    if hasattr(model, 'disable_hooks'):
        model.disable_hooks()

    cov_G_list = []

    for _ in range(steps):
        if img_source == 'pink':
            data = get_pink_noise_batch(BATCH_SIZE, img_size, device, alpha=1.0)
            # Pink noise always uses random labels
            labels = torch.randint(0, num_classes, (BATCH_SIZE,), device=device)
        elif img_source == 'pub':
            data, pub_labels = next(public_iter)
            data = data.to(device)
            pub_labels = pub_labels.to(device).squeeze().long().clamp(max=num_classes - 1)

            if label_source == 'pub':
                labels = pub_labels
            elif label_source == 'noise':
                labels = torch.randint(0, num_classes, (data.size(0),), device=device)
            else:
                raise ValueError(f"Unknown label_source for G: {label_source}")
        else:
            raise ValueError(f"Unknown img_source for G: {img_source}")

        _, cov_G = compute_kfac_covs(model, recorder, data, labels, device, num_classes)
        cov_G_list.append(cov_G)

    if hasattr(model, 'enable_hooks'):
        model.enable_hooks()

    # Average across steps
    avg_G = {}
    if cov_G_list:
        for name in cov_G_list[0]:
            avg_G[name] = sum(c[name] for c in cov_G_list) / len(cov_G_list)

    return avg_G


def compute_preconditioner_oracle(model, recorder, private_iter, steps, device, img_size, num_classes):
    """Compute oracle preconditioner from private data (for upper bound comparison)."""
    if hasattr(model, 'disable_hooks'):
        model.disable_hooks()

    cov_A_list, cov_G_list = [], []

    for _ in range(steps):
        data, labels = next(private_iter)
        data = data.to(device)
        labels = labels.to(device).squeeze().long()

        cov_A, cov_G = compute_kfac_covs(model, recorder, data, labels, device, num_classes)
        cov_A_list.append(cov_A)
        cov_G_list.append(cov_G)

    if hasattr(model, 'enable_hooks'):
        model.enable_hooks()

    return accumulate_covs(cov_A_list, cov_G_list)


def compute_cov_synthetic_clustered(model, recorder, steps, device, img_size, num_classes, jitter_scale=0.5):
    """
    Compute A and G covariances from Synthetic Clustered Pink Noise.

    This is the "Holy Grail" of privacy-preserving preconditioning:
    - A matrix: Sees Pink Noise images (good spatial prior from 1/f spectrum)
    - G matrix: Sees gradients that separate GMM clusters (good task prior)
    - Privacy: Perfect. 0 bits of public data leakage.

    The key insight: Both A and G get structure from the same synthetic task,
    which has natural image statistics (pink noise) AND classification structure
    (GMM clusters).

    Args:
        model: The model to compute covariances for
        recorder: KFACRecorder instance
        steps: Number of batches to average over
        device: torch device
        img_size: Image dimensions
        num_classes: Number of classes (used for GMM clusters)
        jitter_scale: How much pink noise jitter around centroids (default 0.5)

    Returns:
        (cov_A, cov_G): Averaged covariance dictionaries
    """
    if hasattr(model, 'disable_hooks'):
        model.disable_hooks()

    # Create the synthetic clustered task
    # Note: We use num_classes clusters to match the private task structure
    synthetic_task = SyntheticPinkClusterTask(
        num_classes=num_classes,
        img_size=img_size,
        device=device,
        jitter_scale=jitter_scale
    )

    cov_A_list, cov_G_list = [], []

    for _ in range(steps):
        # Get a batch from the synthetic clustered pink noise task
        data, labels = synthetic_task.get_batch(BATCH_SIZE)

        # Compute KFAC covariances
        cov_A, cov_G = compute_kfac_covs(model, recorder, data, labels, device, num_classes)
        cov_A_list.append(cov_A)
        cov_G_list.append(cov_G)

    if hasattr(model, 'enable_hooks'):
        model.enable_hooks()

    return accumulate_covs(cov_A_list, cov_G_list)


def make_iterator(loader):
    """Infinite iterator over a dataloader."""
    while True:
        for batch in loader:
            yield batch


def train_epoch(model, train_loader, optimizer, inv_A, inv_G, noise_mult, clip_norm, device, use_sum_reduction=False):
    """Train one epoch with preconditioning.

    Args:
        use_sum_reduction: If True, use sum reduction for loss (required for CrossViT with
                          GradSampleModule loss_reduction="sum"). The loss is computed as sum
                          then divided by batch size for reporting, but grad_sample expects
                          the summed loss for correct per-sample gradient scaling.
    """
    model.train()
    total_loss = 0
    n_batches = 0

    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device).squeeze().long()
        bs = data.size(0)

        optimizer.zero_grad()
        out = model(data)

        if use_sum_reduction:
            # Sum reduction: loss_per_sample.sum() - GradSampleModule divides by batch internally
            loss = nn.CrossEntropyLoss(reduction='sum')(out, target)
            loss.backward()
            # Report mean loss for consistency
            total_loss += loss.item() / bs
        else:
            # Mean reduction (default)
            loss = nn.CrossEntropyLoss(reduction='mean')(out, target)
            loss.backward()
            total_loss += loss.item()

        if inv_A and inv_G:
            precondition_grads(model, inv_A, inv_G)

        apply_dp(model, noise_mult, clip_norm, bs)
        optimizer.step()

        n_batches += 1

    return total_loss / n_batches


def evaluate(model, test_loader, device):
    """Evaluate model accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device).squeeze().long()
            pred = model(data).argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

    return 100.0 * correct / total


# ==========================================
# Experiment Runner
# ==========================================

def compute_preconditioner_for_type(model, recorder, precond_type, img_size, num_classes,
                                     private_iter, public_loader):
    """
    Compute preconditioner covariances based on type.
    Returns (cov_A, cov_G) dictionaries.
    """
    public_iter = make_iterator(public_loader)
    cov_A, cov_G = {}, {}

    if precond_type == 'oracle':
        # Oracle: use private data (upper bound, not privacy-preserving)
        cov_A, cov_G = compute_preconditioner_oracle(model, recorder, private_iter, PRECOND_STEPS, DEVICE, img_size, num_classes)

    elif precond_type == 'identity':
        # No preconditioning (lower bound)
        cov_A, cov_G = {}, {}

    # === 6 Meaningful A ⊗ G Combinations ===

    # A from Public images
    elif precond_type == 'A_pub__G_pub_pub':
        # Full public: A from public images, G from public images + public labels
        cov_A = compute_cov_A(model, recorder, 'pub', PRECOND_STEPS, DEVICE, img_size, num_classes, public_iter)
        public_iter = make_iterator(public_loader)  # Reset iterator
        cov_G = compute_cov_G(model, recorder, 'pub', 'pub', PRECOND_STEPS, DEVICE, img_size, num_classes, public_iter)

    elif precond_type == 'A_pub__G_pub_noise':
        # Public geometry, no task: A from public, G from public images + random labels
        cov_A = compute_cov_A(model, recorder, 'pub', PRECOND_STEPS, DEVICE, img_size, num_classes, public_iter)
        public_iter = make_iterator(public_loader)  # Reset iterator
        cov_G = compute_cov_G(model, recorder, 'pub', 'noise', PRECOND_STEPS, DEVICE, img_size, num_classes, public_iter)

    elif precond_type == 'A_pub__G_pink_noise':
        # Public A, noise G: A from public images, G from pink noise + random labels
        cov_A = compute_cov_A(model, recorder, 'pub', PRECOND_STEPS, DEVICE, img_size, num_classes, public_iter)
        cov_G = compute_cov_G(model, recorder, 'pink', 'noise', PRECOND_STEPS, DEVICE, img_size, num_classes)

    # A from Pink Noise
    elif precond_type == 'A_pink__G_pub_pub':
        # Pink A, full public G: A from pink noise, G from public images + public labels
        cov_A = compute_cov_A(model, recorder, 'pink', PRECOND_STEPS, DEVICE, img_size, num_classes)
        cov_G = compute_cov_G(model, recorder, 'pub', 'pub', PRECOND_STEPS, DEVICE, img_size, num_classes, public_iter)

    elif precond_type == 'A_pink__G_pub_noise':
        # Pink A, public geometry G: A from pink noise, G from public images + random labels
        cov_A = compute_cov_A(model, recorder, 'pink', PRECOND_STEPS, DEVICE, img_size, num_classes)
        cov_G = compute_cov_G(model, recorder, 'pub', 'noise', PRECOND_STEPS, DEVICE, img_size, num_classes, public_iter)

    elif precond_type == 'A_pink__G_pink_noise':
        # Full pink noise: A from pink noise, G from pink noise + random labels
        cov_A = compute_cov_A(model, recorder, 'pink', PRECOND_STEPS, DEVICE, img_size, num_classes)
        cov_G = compute_cov_G(model, recorder, 'pink', 'noise', PRECOND_STEPS, DEVICE, img_size, num_classes)

    # === Synthetic Clustered Pink Noise (The "Holy Grail") ===
    elif precond_type == 'A_syn__G_syn_clustered':
        # GMM with Pink Noise centroids + Pink Noise jitter
        # - A matrix: Sees Pink Noise images (good spatial prior)
        # - G matrix: Sees gradients that separate GMM clusters (good task prior)
        # - Privacy: Perfect. 0 bits of public data leakage.
        cov_A, cov_G = compute_cov_synthetic_clustered(
            model, recorder, PRECOND_STEPS, DEVICE, img_size, num_classes, jitter_scale=0.5
        )

    else:
        raise ValueError(f"Unknown preconditioner type: {precond_type}")

    return cov_A, cov_G


def run_single_experiment(scenario_config, precond_name, precond_type, cached_loaders, seed):
    """Run a single experiment with given scenario and preconditioning."""
    set_seed(seed)

    img_size = scenario_config['img_size']
    num_classes = scenario_config['classes']  # Already updated for medmnist_global
    private_dataset = scenario_config['private']

    # Use pre-loaded data from cache (avoids double-loading MedMNIST-Global)
    train_loader = cached_loaders['train_loader']
    test_loader = cached_loaders['test_loader']
    train_len = cached_loaders['train_len']
    public_loader = cached_loaders['public_loader']
    model_type = cached_loaders.get('model_type', 'simple_cnn')

    # Check if using a pretrained model
    use_pretrained = model_type in ['crossvit', 'convnext']

    # Create model
    model, _, _ = get_model(num_classes, img_size, private_dataset)
    model = model.to(DEVICE)

    # Setup for pretrained models (CrossViT or ConvNeXt)
    if use_pretrained:
        # Use loss_reduction="sum" for pretrained models
        model = GradSampleModule(model, batch_first=True, loss_reduction="sum")

        target = model._module

        # Freeze all parameters first
        for param in target.parameters():
            param.requires_grad = False

        # Unfreeze classifier head (always exists for both CrossViT and ConvNeXt)
        for param in target.classifier.parameters():
            param.requires_grad = True

        # Model-specific unfreezing of additional layers
        if model_type == 'crossvit':
            # CrossViT: optionally unfreeze norm and last block
            if hasattr(target.backbone, 'norm'):
                for param in target.backbone.norm.parameters():
                    param.requires_grad = True
            if hasattr(target.backbone, 'blocks') and len(target.backbone.blocks) > 0:
                last_block = target.backbone.blocks[-1]
                for param in last_block.parameters():
                    param.requires_grad = True

        elif model_type == 'convnext':
            # ConvNeXt: optionally unfreeze the final norm layer and last stage
            if hasattr(target.backbone, 'norm_pre'):
                for param in target.backbone.norm_pre.parameters():
                    param.requires_grad = True
            # Optionally unfreeze the last stage (stages.3 in ConvNeXt)
            if hasattr(target.backbone, 'stages') and len(target.backbone.stages) > 0:
                last_stage = target.backbone.stages[-1]
                for param in last_stage.parameters():
                    param.requires_grad = True

        trainable_params = [p for p in model.parameters() if p.requires_grad]
        unfrozen_names = [name for name, param in target.named_parameters() if param.requires_grad]

        model_name = 'CrossViT' if model_type == 'crossvit' else 'ConvNeXt'
        print(f"  + {model_name} trainable params: {len(trainable_params)} / {len(list(model.parameters()))}")
        print(f"  + Unfrozen layers check: {unfrozen_names[:3]} ... {unfrozen_names[-3:] if len(unfrozen_names) > 3 else unfrozen_names}")

        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=LEARNING_RATE
        )
        use_sum_reduction = True
    else:
        # SimpleCNN: standard GradSampleModule and SGD
        model = GradSampleModule(model)
        optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)
        use_sum_reduction = False

    # DP params
    sample_rate = BATCH_SIZE / train_len
    noise_mult = get_noise_multiplier(
        target_epsilon=EPSILON, target_delta=DELTA,
        sample_rate=sample_rate, epochs=EPOCHS, accountant="rdp"
    )

    # For pretrained models, only register hooks on trainable layers
    recorder = KFACRecorder(model, only_trainable=use_pretrained)

    # Create iterators
    private_iter = make_iterator(train_loader)

    # Compute initial preconditioner
    cov_A, cov_G = compute_preconditioner_for_type(
        model, recorder, precond_type, img_size, num_classes, private_iter, public_loader
    )

    # Compute inverse sqrt
    if cov_A and cov_G:
        inv_A, inv_G = compute_inv_sqrt(cov_A, cov_G)
    else:
        inv_A, inv_G = {}, {}

    # Training loop with optional periodic preconditioner updates
    results = []
    for epoch in range(EPOCHS):
        # Check if we need to update preconditioner (skip epoch 0 since we just computed it)
        if PRECOND_UPDATE_EVERY and PRECOND_UPDATE_EVERY > 0 and epoch > 0 and epoch % PRECOND_UPDATE_EVERY == 0:
            if precond_type != 'identity':  # No need to recompute identity
                print(f"      [Updating preconditioner at epoch {epoch+1}]")
                private_iter = make_iterator(train_loader)  # Reset iterator
                cov_A, cov_G = compute_preconditioner_for_type(
                    model, recorder, precond_type, img_size, num_classes, private_iter, public_loader
                )
                if cov_A and cov_G:
                    inv_A, inv_G = compute_inv_sqrt(cov_A, cov_G)

        loss = train_epoch(model, train_loader, optimizer, inv_A, inv_G, noise_mult, clip_norm=CLIP_NORM, device=DEVICE, use_sum_reduction=use_sum_reduction)
        acc = evaluate(model, test_loader, DEVICE)
        results.append({'epoch': epoch, 'loss': loss, 'accuracy': acc})
        print(f"    Epoch {epoch+1}/{EPOCHS}: Loss={loss:.4f}, Acc={acc:.2f}%")

    return results


def run_scenario(scenario_key, scenario_config):
    """Run all preconditioning experiments for a scenario with multiple seeds."""
    scenario_config = scenario_config.copy()  # Don't modify original
    img_size = scenario_config['img_size']
    private_dataset = scenario_config['private']
    public_dataset = scenario_config['public']
    num_classes = scenario_config['classes']

    # Determine model type and image size
    _, actual_img_size, model_type_str = get_model(num_classes, img_size, private_dataset)

    # Check if we're using a pretrained model (needs ImageNet norm and larger images)
    use_pretrained = model_type_str in ['crossvit', 'convnext']

    # Override image size for pretrained models
    if use_pretrained:
        img_size = actual_img_size
        scenario_config['img_size'] = img_size
        print(f"\n[Using {model_type_str.upper()} - resizing images to {img_size}x{img_size}]")

    # Pre-load datasets once (avoids double-loading for MedMNIST-Global)
    # Use ImageNet normalization for pretrained models
    loader_result = get_dataset_loaders(private_dataset, BATCH_SIZE, img_size, use_imagenet_norm=use_pretrained)
    if private_dataset == 'medmnist_global':
        train_loader, test_loader, train_len, actual_classes = loader_result
        scenario_config['classes'] = actual_classes
    else:
        train_loader, test_loader, train_len = loader_result

    public_loader, _, _ = get_dataset_loaders(public_dataset, BATCH_SIZE, img_size, use_imagenet_norm=use_pretrained)[:3]

    # Model name for display
    model_display_names = {
        'simple_cnn': 'SimpleCNN',
        'crossvit': 'CrossViT (frozen backbone)',
        'convnext': 'ConvNeXt (frozen backbone)'
    }

    print(f"\n{'='*70}")
    print(f"SCENARIO: {scenario_config['name']}")
    print(f"{'='*70}")
    print(f"Description: {scenario_config['description']}")
    print(f"Prediction: {scenario_config['prediction']}")
    print(f"Private: {scenario_config['private']} | Public: {scenario_config['public']}")
    print(f"Image size: {scenario_config['img_size']}x{scenario_config['img_size']} | Classes: {scenario_config['classes']}")
    print(f"Model: {model_display_names.get(model_type_str, model_type_str)}")
    print(f"Seeds: {SEEDS}")
    print(f"{'='*70}")

    # Determine which preconditioner types to run
    if ACTIVE_PRECOND_TYPES == 'all':
        precond_types = ALL_PRECOND_TYPES
    else:
        # Filter to only selected types
        precond_types = {k: v for k, v in ALL_PRECOND_TYPES.items() if v in ACTIVE_PRECOND_TYPES}

    # Cache loaders to pass to experiments
    cached_loaders = {
        'train_loader': train_loader,
        'test_loader': test_loader,
        'train_len': train_len,
        'public_loader': public_loader,
        'model_type': model_type_str,  # Store model type instead of use_crossvit
    }

    # Run each preconditioner with multiple seeds
    all_results = {}  # {precond_name: {'mean': [...], 'std': [...], 'seeds': {seed: [...]}}}

    for precond_name, precond_type in precond_types.items():
        print(f"\n  Running: {precond_name}")
        seed_results = {}

        for seed in SEEDS:
            print(f"    Seed {seed}:")
            results = run_single_experiment(scenario_config, precond_name, precond_type, cached_loaders, seed)
            seed_results[seed] = results

        # Aggregate results across seeds
        # Each results is a list of {epoch, loss, accuracy}
        n_epochs = len(seed_results[SEEDS[0]])
        aggregated = []

        for epoch_idx in range(n_epochs):
            accs = [seed_results[s][epoch_idx]['accuracy'] for s in SEEDS]
            losses = [seed_results[s][epoch_idx]['loss'] for s in SEEDS]
            aggregated.append({
                'epoch': epoch_idx,
                'accuracy': np.mean(accs),
                'accuracy_std': np.std(accs),
                'loss': np.mean(losses),
                'loss_std': np.std(losses),
            })

        final_accs = [seed_results[s][-1]['accuracy'] for s in SEEDS]
        print(f"    Final Acc: {np.mean(final_accs):.2f}% ± {np.std(final_accs):.2f}%")

        all_results[precond_name] = {
            'aggregated': aggregated,
            'seeds': seed_results,
            'final_mean': np.mean(final_accs),
            'final_std': np.std(final_accs),
        }

    return all_results


def print_scenario_summary(scenario_key, scenario_config, all_results):
    """Print summary for a scenario with multi-seed results."""
    print(f"\n{'='*80}")
    print(f"RESULTS: {scenario_config['name']} ({scenario_key})")
    print(f"{'='*80}")
    print(f"{'Method':<30} | {'Final Acc (mean±std)':>20} | {'Best Acc':>10}")
    print("-"*80)

    for name, result_data in all_results.items():
        final_mean = result_data['final_mean']
        final_std = result_data['final_std']
        best_acc = max(r['accuracy'] for r in result_data['aggregated'])
        print(f"{name:<30} | {final_mean:>8.2f}% ± {final_std:>5.2f}% | {best_acc:>9.2f}%")

    # Helper to safely get accuracy mean and std (returns None if not run)
    def get_acc(key):
        if key in all_results:
            return all_results[key]['final_mean'], all_results[key]['final_std']
        return None, None

    # Extract key metrics using new naming convention
    oracle_mean, oracle_std = get_acc("Oracle (Private)")
    identity_mean, identity_std = get_acc("Identity (No Precond)")

    # 6 A ⊗ G combinations
    A_pub_G_pub_pub, A_pub_G_pub_pub_std = get_acc("A_pub ⊗ G_pub_pub")
    A_pub_G_pub_noise, A_pub_G_pub_noise_std = get_acc("A_pub ⊗ G_pub_noise")
    A_pub_G_pink_noise, A_pub_G_pink_noise_std = get_acc("A_pub ⊗ G_pink_noise")

    A_pink_G_pub_pub, A_pink_G_pub_pub_std = get_acc("A_pink ⊗ G_pub_pub")
    A_pink_G_pub_noise, A_pink_G_pub_noise_std = get_acc("A_pink ⊗ G_pub_noise")
    A_pink_G_pink_noise, A_pink_G_pink_noise_std = get_acc("A_pink ⊗ G_pink_noise")

    # Synthetic Clustered Pink Noise (The "Holy Grail")
    A_syn_G_syn, A_syn_G_syn_std = get_acc("A_syn ⊗ G_syn (Clustered Pink)")

    print(f"\n{'='*80}")
    print("KEY INSIGHTS (mean ± std):")
    print("-"*80)

    # Effect of A source (holding G constant)
    if A_pub_G_pub_pub is not None and A_pink_G_pub_pub is not None:
        diff = A_pub_G_pub_pub - A_pink_G_pub_pub
        print(f"  A effect (G=pub_pub):  A_pub={A_pub_G_pub_pub:.2f}±{A_pub_G_pub_pub_std:.2f}% vs A_pink={A_pink_G_pub_pub:.2f}±{A_pink_G_pub_pub_std:.2f}%  (Δ={diff:+.2f}%)")
    if A_pub_G_pub_noise is not None and A_pink_G_pub_noise is not None:
        diff = A_pub_G_pub_noise - A_pink_G_pub_noise
        print(f"  A effect (G=pub_noise): A_pub={A_pub_G_pub_noise:.2f}±{A_pub_G_pub_noise_std:.2f}% vs A_pink={A_pink_G_pub_noise:.2f}±{A_pink_G_pub_noise_std:.2f}%  (Δ={diff:+.2f}%)")
    if A_pub_G_pink_noise is not None and A_pink_G_pink_noise is not None:
        diff = A_pub_G_pink_noise - A_pink_G_pink_noise
        print(f"  A effect (G=pink_noise): A_pub={A_pub_G_pink_noise:.2f}±{A_pub_G_pink_noise_std:.2f}% vs A_pink={A_pink_G_pink_noise:.2f}±{A_pink_G_pink_noise_std:.2f}%  (Δ={diff:+.2f}%)")

    print()

    # Effect of G label source (holding A and G image source constant)
    if A_pub_G_pub_pub is not None and A_pub_G_pub_noise is not None:
        diff = A_pub_G_pub_pub - A_pub_G_pub_noise
        print(f"  Label effect (A=pub, G_img=pub): pub_label={A_pub_G_pub_pub:.2f}% vs noise_label={A_pub_G_pub_noise:.2f}%  (Δ={diff:+.2f}%)")
    if A_pink_G_pub_pub is not None and A_pink_G_pub_noise is not None:
        diff = A_pink_G_pub_pub - A_pink_G_pub_noise
        print(f"  Label effect (A=pink, G_img=pub): pub_label={A_pink_G_pub_pub:.2f}% vs noise_label={A_pink_G_pub_noise:.2f}%  (Δ={diff:+.2f}%)")

    print()

    # Effect of G image source (holding A constant, using noise labels)
    if A_pub_G_pub_noise is not None and A_pub_G_pink_noise is not None:
        diff = A_pub_G_pub_noise - A_pub_G_pink_noise
        print(f"  G_img effect (A=pub, noise_label): G_pub={A_pub_G_pub_noise:.2f}% vs G_pink={A_pub_G_pink_noise:.2f}%  (Δ={diff:+.2f}%)")
    if A_pink_G_pub_noise is not None and A_pink_G_pink_noise is not None:
        diff = A_pink_G_pub_noise - A_pink_G_pink_noise
        print(f"  G_img effect (A=pink, noise_label): G_pub={A_pink_G_pub_noise:.2f}% vs G_pink={A_pink_G_pink_noise:.2f}%  (Δ={diff:+.2f}%)")

    print()

    # Synthetic Clustered Pink Noise comparison
    if A_syn_G_syn is not None:
        print(f"  SYNTHETIC CLUSTERED (Holy Grail): {A_syn_G_syn:.2f}±{A_syn_G_syn_std:.2f}%")
        if A_pink_G_pink_noise is not None:
            diff = A_syn_G_syn - A_pink_G_pink_noise
            print(f"    vs A_pink⊗G_pink_noise: Δ={diff:+.2f}% (clustered structure helps?)")
        if oracle_mean is not None:
            gap = oracle_mean - A_syn_G_syn
            print(f"    Gap to Oracle: {gap:.2f}% (0 bits of public data used!)")

    print(f"\nPREDICTION: {scenario_config['prediction']}")

    # Determine winner (excluding oracle and identity)
    methods = {}
    if A_pub_G_pub_pub is not None:
        methods["A_pub⊗G_pub_pub"] = A_pub_G_pub_pub
    if A_pub_G_pub_noise is not None:
        methods["A_pub⊗G_pub_noise"] = A_pub_G_pub_noise
    if A_pub_G_pink_noise is not None:
        methods["A_pub⊗G_pink_noise"] = A_pub_G_pink_noise
    if A_pink_G_pub_pub is not None:
        methods["A_pink⊗G_pub_pub"] = A_pink_G_pub_pub
    if A_pink_G_pub_noise is not None:
        methods["A_pink⊗G_pub_noise"] = A_pink_G_pub_noise
    if A_pink_G_pink_noise is not None:
        methods["A_pink⊗G_pink_noise"] = A_pink_G_pink_noise
    if A_syn_G_syn is not None:
        methods["A_syn⊗G_syn (Clustered)"] = A_syn_G_syn

    if methods:
        winner = max(methods, key=methods.get)
        print(f"ACTUAL WINNER: {winner} ({methods[winner]:.2f}%)")
    else:
        winner = "N/A"

    # Helper to format mean±std for CSV
    def fmt_mean_std(mean, std):
        if mean is not None:
            return f"{mean:.2f}±{std:.2f}"
        return None

    return {
        'scenario': scenario_key,
        'oracle': fmt_mean_std(oracle_mean, oracle_std),
        'identity': fmt_mean_std(identity_mean, identity_std),
        'A_pub_G_pub_pub': fmt_mean_std(A_pub_G_pub_pub, A_pub_G_pub_pub_std),
        'A_pub_G_pub_noise': fmt_mean_std(A_pub_G_pub_noise, A_pub_G_pub_noise_std),
        'A_pub_G_pink_noise': fmt_mean_std(A_pub_G_pink_noise, A_pub_G_pink_noise_std),
        'A_pink_G_pub_pub': fmt_mean_std(A_pink_G_pub_pub, A_pink_G_pub_pub_std),
        'A_pink_G_pub_noise': fmt_mean_std(A_pink_G_pub_noise, A_pink_G_pub_noise_std),
        'A_pink_G_pink_noise': fmt_mean_std(A_pink_G_pink_noise, A_pink_G_pink_noise_std),
        'A_syn_G_syn_clustered': fmt_mean_std(A_syn_G_syn, A_syn_G_syn_std),
        'winner': winner,
        'prediction': scenario_config['prediction'],
    }


def main():
    print("="*70)
    print("COMPREHENSIVE PRECONDITIONING TRANSFER EXPERIMENT")
    print("="*70)
    print(f"Testing {len(SCENARIOS)} scenarios to isolate A (Input) vs G (Output) effects")
    print(f"ε={EPSILON}, Epochs={EPOCHS}, Batch={BATCH_SIZE}, LR={LEARNING_RATE}")
    print(f"Seeds: {SEEDS} (results are mean ± std)")
    if PRECOND_UPDATE_EVERY and PRECOND_UPDATE_EVERY > 0:
        print(f"Preconditioner update: every {PRECOND_UPDATE_EVERY} epoch(s)")
    else:
        print(f"Preconditioner update: once at start")
    print("="*70)

    # Determine which scenarios to run
    if ACTIVE_SCENARIO == 'all':
        scenarios_to_run = SCENARIOS
    else:
        scenarios_to_run = {ACTIVE_SCENARIO: SCENARIOS[ACTIVE_SCENARIO]}

    all_scenario_results = {}
    summaries = []

    for scenario_key, scenario_config in scenarios_to_run.items():
        results = run_scenario(scenario_key, scenario_config)
        all_scenario_results[scenario_key] = results
        summary = print_scenario_summary(scenario_key, scenario_config, results)
        summaries.append(summary)

    # Final summary across all scenarios
    if len(summaries) > 1:
        print("\n" + "="*70)
        print("GRAND SUMMARY: ALL SCENARIOS (mean ± std)")
        print("="*150)
        print(f"{'Scenario':<16} | {'A_pub⊗G_pp':>12} | {'A_pub⊗G_pn':>12} | {'A_pub⊗G_kn':>12} | {'A_pink⊗G_pp':>12} | {'A_pink⊗G_pn':>12} | {'A_pink⊗G_kn':>12} | {'Winner':>18}")
        print("-"*150)

        def fmt(val):
            return f"{val:>12}" if val is not None else "     N/A    "

        for s in summaries:
            print(f"{s['scenario']:<16} | {fmt(s['A_pub_G_pub_pub'])} | {fmt(s['A_pub_G_pub_noise'])} | {fmt(s['A_pub_G_pink_noise'])} | {fmt(s['A_pink_G_pub_pub'])} | {fmt(s['A_pink_G_pub_noise'])} | {fmt(s['A_pink_G_pink_noise'])} | {s['winner']:>18}")

        print("="*150)
        print("Legend: G_pp = G_pub_pub, G_pn = G_pub_noise, G_kn = G_pink_noise")

    # Save results to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"transfer_results_{timestamp}.csv"
    with open(csv_filename, 'w', newline='') as f:
        fieldnames = ['scenario', 'oracle', 'identity',
                      'A_pub_G_pub_pub', 'A_pub_G_pub_noise', 'A_pub_G_pink_noise',
                      'A_pink_G_pub_pub', 'A_pink_G_pub_noise', 'A_pink_G_pink_noise',
                      'A_syn_G_syn_clustered', 'winner', 'prediction']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    print(f"\nResults saved to: {csv_filename}")


if __name__ == "__main__":
    main()
