# DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning

> Anonymous code submission for ICML 2026

## Overview

Differentially private optimization suffers from a fundamental geometric mismatch: deep networks have highly anisotropic loss landscapes, yet DP-SGD injects isotropic noise. Second-order preconditioning can resolve this, but estimating curvature typically requires private data (consuming privacy budget) or public data (introducing distribution shift).

**DP-KFC** constructs KFAC preconditioners by probing networks with structured synthetic noise, requiring neither private nor public data. The Fisher Information Matrix decouples into *architectural sensitivity*, recoverable via synthetic noise, and *input correlations*, approximable from modality-specific frequency statistics.

### Key Contributions

- **Decoupled Geometry:** The Fisher Information Matrix decouples into architectural scaling (recoverable via synthetic noise) and input correlation (recoverable via general frequency statistics), enabling preconditioner construction without spending privacy budget.
- **Privacy-Free Preconditioning:** KFAC factors are estimated using synthetic noise alone -- no private or public data required -- preconditioning the loss landscape to match the isotropic privacy noise.
- **Robustness to Negative Transfer:** Public-data proxies fail under domain mismatch; DP-KFC strictly dominates these baselines, offering a domain-agnostic approach for private learning.

## Project Structure

```
dp-kfac/
├── main.py                         # Entry point (experiment / visualize)
├── config.yaml                     # Default experiment configuration
├── pyproject.toml                  # Dependencies and project metadata
│
├── src/dp_kfac/                    # Core library
│   ├── trainer.py                  # Training orchestration (baseline, DP-SGD, DP-KFAC)
│   ├── optimizer.py                # DPKFACOptimizer with pink/white noise generation
│   ├── covariance.py               # KFAC covariance computation (A and G factors)
│   ├── precondition.py             # Per-sample gradient preconditioning
│   ├── privacy.py                  # Clipping and Gaussian noise injection
│   ├── recorder.py                 # Activation/gradient hook recorder
│   ├── models.py                   # Model definitions (CNN, CrossViT, ConvNeXt, RoBERTa)
│   ├── data.py                     # Dataset loaders (vision and NLP)
│   ├── methods.py                  # Preconditioning method definitions
│   ├── config.py                   # YAML-based configuration system
│   ├── experiment.py               # Experiment run tracking
│   └── types.py                    # Shared type definitions
│
├── configs/                        # Experiment configurations
│   ├── exp_scenarios.yaml          # Multi-scenario transfer study
│   ├── exp1_cnn_mnist.yaml         # CNN on MNIST
│   ├── exp1_cross_vit_cifar100.yaml# CrossViT on CIFAR-100
│   └── exp1_bert_stackoverflow.yaml# RoBERTa on StackOverflow
│
├── scripts/                        # Experiment runner scripts
│   ├── exp_1_cnn_mnist.py
│   ├── exp_1_cross_vit_cifar100.py
│   ├── exp_1_robert_stackoverflow.py
│   ├── exp_3.py
│   ├── results_exp_1.py
│   └── vis_1.py
│
└── experiments/                    # Output directory (created at runtime)
```

## Requirements

- Python >= 3.13
- PyTorch >= 2.9.1
- [Opacus](https://github.com/pytorch/opacus) >= 1.5.4
- torchvision >= 0.24.1
- timm >= 1.0.0
- transformers >= 4.40.0 (for NLP experiments)

## Installation

```bash
# Using uv (recommended)
uv sync

# Or using pip
pip install -e .
```

## Usage

### Running Experiments

```bash
# Default experiment (CNN on MNIST / FashionMNIST)
python main.py experiment --config config.yaml

# Multi-scenario transfer study
python main.py experiment --config configs/exp_scenarios.yaml

# Vision transformer on CIFAR-100
python main.py experiment --config configs/exp1_cross_vit_cifar100.yaml

# RoBERTa on StackOverflow
python main.py experiment --config configs/exp1_bert_stackoverflow.yaml
```

### Visualization

```bash
python main.py visualize --config config.yaml
```

### Configuration

Experiments are configured via YAML files. Key parameters:

```yaml
experiment:
  seeds: [42, 43, 44, 45, 46]
  device: "cuda"

data:
  private_dataset: "mnist"
  public_dataset: "fashionmnist"
  batch_size: 256

privacy:
  epsilons: [0.5, 1.0, 2.0, 3.0, 5.0, 7.5, 10.0]
  delta: 1e-5
  max_grad_norm: 1.0

methods:
  - "dp_sgd"
  - "dp_kfac_public"
  - "dp_kfac_noise"
  - "dp_kfac_pink"
```

### Preconditioning Methods

| Method | Activation Source (A) | Gradient Source (G) |
|---|---|---|
| `dp_sgd` | -- | -- |
| `dp_kfac_public` | Public data | Public data |
| `dp_kfac_noise` | White noise | White noise |
| `dp_kfac_pink` | Pink noise | Pink noise |
| `dp_kfac_hybrid_pub_pub_noise` | Public data | Public + noise |
| `dp_kfac_hybrid_pub_pink_noise` | Public data | Pink + noise |
| `dp_kfac_hybrid_pink_pub_pub` | Pink noise | Public data |
| `dp_kfac_hybrid_pink_pub_noise` | Pink noise | Public + noise |

## Supported Models and Datasets

**Models:** SimpleCNN, MLP, CrossViT (pretrained), ConvNeXt (pretrained), RoBERTa (pretrained)

**Datasets:** MNIST, FashionMNIST, CIFAR-10, CIFAR-100, PathMNIST, SST-2, StackOverflow

## Output

Each experiment run produces:

```
experiments/dp_kfac_<timestamp>/
├── config.yaml       # Configuration used
├── results.json      # Per-method, per-seed, per-epsilon results
├── metrics.json      # Additional tracked metrics
└── run.log           # Timestamped log
```

## License

This repository is released for anonymous review purposes.
