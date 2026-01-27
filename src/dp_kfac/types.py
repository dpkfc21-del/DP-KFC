from typing import Dict
from dataclasses import dataclass
import torch

Tensor = torch.Tensor
CovarianceDict = Dict[str, Tensor]


@dataclass
class KFACConfig:
    damping: float = 1e-3
    cov_ema_decay: float = 0.95
    update_freq: int = 1


@dataclass
class DPConfig:
    noise_multiplier: float
    max_grad_norm: float
    delta: float = 1e-5


@dataclass
class CovariancePair:
    A: CovarianceDict
    G: CovarianceDict
