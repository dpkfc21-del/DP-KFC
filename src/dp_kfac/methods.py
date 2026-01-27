from typing import Tuple, Iterator, Optional
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F

from .types import CovarianceDict, KFACConfig
from .recorder import KFACRecorder
from .covariance import compute_covariances, compute_inverse_sqrt, CovariancePair
from .optimizer import generate_pink_noise


class ActivationSource(Enum):
    PUBLIC = "public"
    PINK_NOISE = "pink"
    WHITE_NOISE = "white"


class GradientSource(Enum):
    PUBLIC_WITH_LABELS = "public_labels"
    PUBLIC_WITH_NOISE = "public_noise"
    PINK_NOISE = "pink_noise"
    WHITE_NOISE = "white_noise"


@dataclass
class PrecondMethod:
    name: str
    activation_source: ActivationSource
    gradient_source: GradientSource
    description: str


METHODS = {
    "dp_sgd": None,

    "dp_kfac_public": PrecondMethod(
        name="dp_kfac_public",
        activation_source=ActivationSource.PUBLIC,
        gradient_source=GradientSource.PUBLIC_WITH_LABELS,
        description="A from public, G from public with real labels",
    ),

    "dp_kfac_noise": PrecondMethod(
        name="dp_kfac_noise",
        activation_source=ActivationSource.WHITE_NOISE,
        gradient_source=GradientSource.WHITE_NOISE,
        description="A from white noise, G from white noise",
    ),

    "dp_kfac_pink": PrecondMethod(
        name="dp_kfac_pink",
        activation_source=ActivationSource.PINK_NOISE,
        gradient_source=GradientSource.PINK_NOISE,
        description="A from pink noise, G from pink noise",
    ),

    "dp_kfac_hybrid_pub_pub_noise": PrecondMethod(
        name="dp_kfac_hybrid_pub_pub_noise",
        activation_source=ActivationSource.PUBLIC,
        gradient_source=GradientSource.PUBLIC_WITH_NOISE,
        description="A from public, G from public images with random labels",
    ),

    "dp_kfac_hybrid_pub_pink_noise": PrecondMethod(
        name="dp_kfac_hybrid_pub_pink_noise",
        activation_source=ActivationSource.PUBLIC,
        gradient_source=GradientSource.PINK_NOISE,
        description="A from public, G from pink noise with random labels",
    ),

    # A from Pink Noise variants (pink A captures natural image statistics without dataset bias)
    "dp_kfac_hybrid_pink_pub_pub": PrecondMethod(
        name="dp_kfac_hybrid_pink_pub_pub",
        activation_source=ActivationSource.PINK_NOISE,
        gradient_source=GradientSource.PUBLIC_WITH_LABELS,
        description="A from pink noise, G from public with real labels",
    ),

    "dp_kfac_hybrid_pink_pub_noise": PrecondMethod(
        name="dp_kfac_hybrid_pink_pub_noise",
        activation_source=ActivationSource.PINK_NOISE,
        gradient_source=GradientSource.PUBLIC_WITH_NOISE,
        description="A from pink noise, G from public images with random labels",
    ),
}


def generate_white_noise(
    batch_size: int,
    input_shape: Tuple[int, ...],
    device: torch.device,
) -> torch.Tensor:
    if len(input_shape) == 3:
        channels, height, width = input_shape
    else:
        channels, height, width = 1, input_shape[0], input_shape[0]
    return torch.randn(batch_size, channels, height, width, device=device)


def compute_activation_covariance(
    model: nn.Module,
    recorder: KFACRecorder,
    source: ActivationSource,
    public_iter: Optional[Iterator],
    batch_size: int,
    input_shape: Tuple[int, ...],
    num_classes: int,
    device: torch.device,
    steps: int = 10,
) -> CovarianceDict:
    cov_A_list = []

    recorder.enable()
    for _ in range(steps):
        if source == ActivationSource.PUBLIC:
            if public_iter is None:
                raise ValueError("public_iter required for PUBLIC activation source")
            data, labels = next(public_iter)
            data = data.to(device)
            labels = labels.to(device)
        elif source == ActivationSource.PINK_NOISE:
            data = generate_pink_noise(batch_size, input_shape, device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)
        else:
            data = generate_white_noise(batch_size, input_shape, device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)

        model.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, labels)
        loss.backward()

        cov = compute_covariances(model, recorder.activations, recorder.backprops)
        cov_A_list.append(cov.A)
        recorder.clear()

    recorder.disable()
    model.zero_grad()

    if not cov_A_list:
        return {}

    avg_A: CovarianceDict = {}
    for name in cov_A_list[0]:
        stacked = torch.stack([c[name] for c in cov_A_list])
        avg_A[name] = stacked.mean(dim=0)

    return avg_A


def compute_gradient_covariance(
    model: nn.Module,
    recorder: KFACRecorder,
    source: GradientSource,
    public_iter: Optional[Iterator],
    batch_size: int,
    input_shape: Tuple[int, ...],
    num_classes: int,
    device: torch.device,
    steps: int = 10,
) -> CovarianceDict:
    cov_G_list = []

    recorder.enable()
    for _ in range(steps):
        if source == GradientSource.PUBLIC_WITH_LABELS:
            if public_iter is None:
                raise ValueError("public_iter required for PUBLIC_WITH_LABELS gradient source")
            data, labels = next(public_iter)
            data = data.to(device)
            labels = labels.to(device)
        elif source == GradientSource.PUBLIC_WITH_NOISE:
            if public_iter is None:
                raise ValueError("public_iter required for PUBLIC_WITH_NOISE gradient source")
            data, _ = next(public_iter)
            data = data.to(device)
            labels = torch.randint(0, num_classes, (data.size(0),), device=device)
        elif source == GradientSource.PINK_NOISE:
            data = generate_pink_noise(batch_size, input_shape, device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)
        else:
            data = generate_white_noise(batch_size, input_shape, device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)

        model.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, labels)
        loss.backward()

        cov = compute_covariances(model, recorder.activations, recorder.backprops)
        cov_G_list.append(cov.G)
        recorder.clear()

    recorder.disable()
    model.zero_grad()

    if not cov_G_list:
        return {}

    avg_G: CovarianceDict = {}
    for name in cov_G_list[0]:
        stacked = torch.stack([c[name] for c in cov_G_list])
        avg_G[name] = stacked.mean(dim=0)

    return avg_G


def compute_preconditioner(
    model: nn.Module,
    recorder: KFACRecorder,
    method: PrecondMethod,
    public_iter: Optional[Iterator],
    batch_size: int,
    input_shape: Tuple[int, ...],
    num_classes: int,
    device: torch.device,
    kfac_config: KFACConfig,
    steps: int = 10,
) -> Tuple[CovarianceDict, CovarianceDict]:
    cov_A = compute_activation_covariance(
        model=model,
        recorder=recorder,
        source=method.activation_source,
        public_iter=public_iter,
        batch_size=batch_size,
        input_shape=input_shape,
        num_classes=num_classes,
        device=device,
        steps=steps,
    )

    cov_G = compute_gradient_covariance(
        model=model,
        recorder=recorder,
        source=method.gradient_source,
        public_iter=public_iter,
        batch_size=batch_size,
        input_shape=input_shape,
        num_classes=num_classes,
        device=device,
        steps=steps,
    )

    if not cov_A or not cov_G:
        return {}, {}

    cov_pair = CovariancePair(A=cov_A, G=cov_G)
    inv_A, inv_G = compute_inverse_sqrt(cov_pair, damping=kfac_config.damping)

    return inv_A, inv_G


def get_method(method_name: str) -> Optional[PrecondMethod]:
    return METHODS.get(method_name)


def is_kfac_method(method_name: str) -> bool:
    return method_name.startswith("dp_kfac")


def list_methods() -> list[str]:
    return list(METHODS.keys())
