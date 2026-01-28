# DP-KFC: Data-Free Preconditioning for Privacy-Preserving Deep Learning

> Anonymous code submission for ICML 2026

Differentially private SGD injects isotropic noise into gradients, ignoring the highly anisotropic curvature of deep networks. Second-order preconditioners like KFAC can correct this geometry, but estimating curvature typically requires either private data (spending privacy budget) or public data (risking domain mismatch).

**DP-KFC** sidesteps both problems. We show that the Fisher Information Matrix decomposes into *architectural sensitivity* (recoverable via synthetic noise) and *input correlations* (approximable from modality-specific frequency statistics). This lets us build effective KFAC preconditioners **without any real data**, closing the gap to public-data oracles and strictly dominating baselines under domain shift.

## Main Results

### Test Accuracy (%) — All Datasets (Table 4)

Underlined values are best within 1 std. Syn. = Synthetic noise preconditioner, Pub. = Public data preconditioner.

**MNIST / CNN**

| epsilon | Non-DP | DP-SGD | Syn. KFAC | Pub. KFAC |
|:---:|:---:|:---:|:---:|:---:|
| 0.5 | 98.8 +/- 0.1 | 90.8 +/- 0.2 | 93.3 +/- 0.6 | **94.7 +/- 0.4** |
| 1 | | 91.7 +/- 0.2 | 94.2 +/- 0.5 | **95.3 +/- 0.4** |
| 2 | | 92.5 +/- 0.3 | 95.0 +/- 0.4 | **95.7 +/- 0.3** |
| 3 | | 92.9 +/- 0.3 | 95.3 +/- 0.4 | **96.0 +/- 0.3** |
| 5 | | 93.4 +/- 0.3 | 95.7 +/- 0.3 | **96.2 +/- 0.3** |
| 8 | | 93.7 +/- 0.3 | 95.9 +/- 0.3 | **96.4 +/- 0.3** |
| 10 | | 94.0 +/- 0.3 | 96.0 +/- 0.3 | **96.5 +/- 0.2** |

**CIFAR-100 / CrossViT**

| epsilon | Non-DP | DP-SGD | Syn. KFAC | Pub. KFAC |
|:---:|:---:|:---:|:---:|:---:|
| 0.5 | 68.3 +/- 0.3 | 52.1 +/- 0.5 | **53.6 +/- 0.5** | 53.1 +/- 0.5 |
| 1 | | 56.2 +/- 0.4 | **57.6 +/- 0.5** | 57.3 +/- 0.4 |
| 2 | | 58.5 +/- 0.3 | **59.8 +/- 0.4** | 59.6 +/- 0.5 |
| 3 | | 59.5 +/- 0.3 | **60.9 +/- 0.3** | 60.6 +/- 0.4 |
| 5 | | 60.7 +/- 0.3 | **62.0 +/- 0.3** | 61.6 +/- 0.4 |
| 8 | | 61.4 +/- 0.4 | **62.7 +/- 0.4** | 62.4 +/- 0.3 |
| 10 | | 62.0 +/- 0.4 | **63.2 +/- 0.3** | 62.9 +/- 0.3 |

**StackOverflow / BERT**

| epsilon | Non-DP | DP-SGD | Syn. KFAC | Pub. KFAC |
|:---:|:---:|:---:|:---:|:---:|
| 0.5 | 99.0 +/- 0.5 | 78.9 +/- 7.3 | 81.3 +/- 7.0 | **89.8 +/- 4.2** |
| 1 | | 89.5 +/- 1.8 | 91.8 +/- 1.9 | **96.1 +/- 1.0** |
| 2 | | 92.9 +/- 0.7 | 95.4 +/- 0.3 | **97.5 +/- 0.3** |
| 3 | | 93.6 +/- 0.7 | 95.9 +/- 0.3 | **97.9 +/- 0.3** |
| 5 | | 94.2 +/- 0.7 | 96.4 +/- 0.2 | **98.1 +/- 0.3** |
| 8 | | 94.6 +/- 0.6 | 96.5 +/- 0.1 | **98.2 +/- 0.2** |
| 10 | | 94.8 +/- 0.6 | 96.6 +/- 0.1 | **98.3 +/- 0.2** |

**IMDB / Logistic Regression**

| epsilon | Non-DP | DP-SGD | Syn. KFAC | Pub. KFAC |
|:---:|:---:|:---:|:---:|:---:|
| 0.5 | 88.0 +/- 0.5 | 82.1 +/- 0.3 | **83.5 +/- 0.1** | **83.5 +/- 0.1** |
| 1 | | 82.9 +/- 0.1 | **85.1 +/- 0.0** | **85.1 +/- 0.1** |
| 1.5 | | 83.1 +/- 0.0 | **85.5 +/- 0.1** | **85.5 +/- 0.1** |
| 2.8 | | 83.1 +/- 0.1 | **85.9 +/- 0.1** | 85.8 +/- 0.1 |
| 8 | | 83.2 +/- 0.1 | **86.0 +/- 0.0** | **86.0 +/- 0.0** |

### Transfer Robustness (Table 1)

Test accuracy (%) under domain mismatch at epsilon = 1.0. Oracle uses private data.

| Method | Ideal Alignment (Fashion <- MNIST) | Texture Disjoint (Path <- MNIST) |
|:---|:---:|:---:|
| Oracle (Private) | 88.3 +/- 0.2 | 78.4 +/- 1.7 |
| DP-SGD | 83.5 +/- 0.7 | 68.5 +/- 2.3 |
| AdaDPS (Public) | 84.7 +/- 0.3 | 70.5 +/- 2.0 |
| AdaDPS (Pink) | 84.2 +/- 0.5 | 71.2 +/- 1.9 |
| Public DP-KFC | 87.6 +/- 0.2 | 73.4 +/- 1.3 |
| **Synthetic DP-KFC** | **87.8 +/- 0.2** | **78.2 +/- 1.9** |

Synthetic DP-KFC matches the private-data oracle even under severe domain mismatch, where public-data methods degrade.

### Bayesian Hyperparameter Optimization (Table 5)

epsilon = 2.0, 150 Optuna trials per method.

| Dataset | DP-SGD | **DP-KFC** | Delta | Optimal alpha |
|:---|:---:|:---:|:---:|:---:|
| MNIST | 94.2% | **97.1%** | +2.9% | 1.80 |
| FashionMNIST | 84.3% | **87.1%** | +2.8% | 1.48 |
| CIFAR-10 | 50.5% | **59.1%** | +8.6% | 1.78 |

## Installation

```bash
# Clone and install
git clone https://github.com/<anonymous>/DP-KFC.git
cd DP-KFC

# Core install
uv sync

# With NLP dependencies (transformers, datasets)
uv sync --extra nlp

# With medical imaging dependencies (MedMNIST)
uv sync --extra medical
```

Requires Python >= 3.13, PyTorch >= 2.9.1, and [Opacus](https://github.com/pytorch/opacus) >= 1.5.4.

## Reproducing Paper Results

All paper experiments live in `scripts/paper/`. Each script supports `--fast` for quick smoke-tests and `--seed`/`--epsilon` for single-configuration runs.

### Main Experiments (Tables 1--2)

```bash
# Table 1: Vision
uv run scripts/paper/exp_cnn_mnist.py                  # CNN on MNIST
uv run scripts/paper/exp_crossvit_cifar100.py           # CrossViT on CIFAR-100

# Table 2: NLP
uv run scripts/paper/exp_imdb_logreg.py                 # Logistic regression on IMDB
uv run scripts/paper/exp_stackoverflow.py               # BERT on StackOverflow
uv run scripts/paper/exp_sst2.py                        # DistilBERT on SST-2
```

### Ablations (Section 5)

```bash
# Fig 2: FIM eigenvalue spectra across preconditioner sources
uv run scripts/paper/ablation_fim_spectrum.py

# Fig 3: Covariance similarity tracking over training
uv run scripts/paper/ablation_cov_tracking.py

# Table 3: AdaDPS comparison
uv run scripts/paper/ablation_adadps.py

# Fig 4: Transfer alignment across domain match/mismatch
uv run scripts/paper/ablation_transfer_alignment.py
```

### Figures and Tables

```bash
# Generate all paper figures from saved results
uv run scripts/paper/visualize/visualize_vision.py      # Fig 5: Vision accuracy vs epsilon
uv run scripts/paper/visualize/visualize_nlp.py          # Fig 6: NLP accuracy vs epsilon
uv run scripts/paper/visualize/visualize_spectrum.py      # Fig 2: Eigenvalue spectra
uv run scripts/paper/visualize/visualize_cov_tracking.py  # Fig 3: Covariance evolution
uv run scripts/paper/visualize/generate_latex_tables.py   # LaTeX tables for paper
```

### Quick Validation

Run any experiment with reduced settings:

```bash
# Single seed, single epsilon, fewer epochs
uv run scripts/paper/exp_cnn_mnist.py --fast
uv run scripts/paper/exp_cnn_mnist.py --seed 42 --epsilon 1.0
```

## Project Structure

```
dp-kfac/
├── src/dp_kfac/              Core library
│   ├── trainer.py            Training loops (plain, DP-SGD, DP-KFAC)
│   ├── optimizer.py          DPKFACOptimizer with noise generation
│   ├── covariance.py         KFAC covariance computation (A and G factors)
│   ├── precondition.py       Per-sample gradient preconditioning
│   ├── privacy.py            Gradient clipping and Gaussian noise
│   ├── recorder.py           Forward/backward hook recorder
│   ├── methods.py            Method registry (KFAC, AdaDPS, noise variants)
│   ├── models.py             MLP, CNN, ViT, CrossViT, ConvNeXt, BERT, RoBERTa, DistilBERT
│   ├── data.py               Dataset loaders (vision + NLP + TF-IDF)
│   ├── analysis.py           Eigenvalue spectra and covariance tracking
│   └── results.py            CSV I/O and result aggregation
│
├── scripts/paper/            Reproducibility scripts
│   ├── exp_*.py              Main experiments (Tables 1--2)
│   ├── ablation_*.py         Ablation studies (Section 5)
│   └── visualize/            Figure and table generation
│
├── configs/                  YAML experiment configurations
├── main.py                   CLI entry point
└── results/                  Output directory (CSV, pickle, figures)
```

## Preconditioning Methods

| Method | Key | Activation Source (A) | Gradient Source (G) |
|:---|:---|:---|:---|
| DP-SGD | `dp_sgd` | -- | -- |
| KFAC (Public) | `dp_kfac_public` | Public data | Public data |
| KFAC (White noise) | `dp_kfac_noise` | White noise | White noise |
| **KFAC (Pink noise)** | `dp_kfac_pink` | Pink noise | Pink noise |
| AdaDPS | `adadps` | Diagonal E[g^2] | Diagonal E[g^2] |

## Supported Tasks

| Domain | Model | Private Data | Public Proxy |
|:---|:---|:---|:---|
| Vision | SimpleCNN | MNIST | FashionMNIST |
| Vision | CrossViT (frozen) | CIFAR-100 | CIFAR-10 |
| NLP | Logistic Regression | IMDB (TF-IDF) | AG News |
| NLP | BERT (frozen) | StackOverflow | AG News |
| NLP | DistilBERT (frozen) | SST-2 | IMDB |
| Medical | SimpleCNN | PathMNIST | MNIST |

## License

Released for anonymous review purposes.
