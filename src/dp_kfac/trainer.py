from typing import Dict, List, Any, Optional, Iterator, Tuple
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from opacus import GradSampleModule
from opacus.accountants import RDPAccountant
from opacus.accountants.utils import get_noise_multiplier
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn, MofNCompleteColumn
from rich.console import Console
from rich.table import Table

from .types import DPConfig, KFACConfig
from .recorder import KFACRecorder
from .covariance import compute_covariances, compute_inverse_sqrt, accumulate_covariances, CovariancePair
from .precondition import precondition_per_sample_gradients
from .privacy import clip_and_noise_gradients
from .optimizer import generate_pink_noise
from .methods import PrecondMethod, get_method, compute_preconditioner


def set_seed(seed: int) -> None:
    import random
    import os
    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def is_text_batch(batch) -> bool:
    return isinstance(batch, dict) and "input_ids" in batch


def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    is_text: bool = False,
) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    target_model = model._module if hasattr(model, "_module") else model

    with torch.no_grad():
        for batch in test_loader:
            if is_text or is_text_batch(batch):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                target = batch["labels"].to(device)
                output = target_model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                data, target = batch
                data, target = data.to(device), target.to(device)
                output = target_model(data)

            loss = F.cross_entropy(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)

    accuracy = correct / total
    avg_loss = total_loss / len(test_loader)

    return accuracy, avg_loss


def train_plain_sgd(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    epochs: int,
    device: torch.device,
    test_loader: Optional[DataLoader] = None,
    is_text: bool = False,
) -> List[Dict[str, Any]]:
    criterion = nn.CrossEntropyLoss()
    results = []
    console = Console()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        epoch_task = progress.add_task("[cyan]Training", total=epochs)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            batch_task = progress.add_task(f"[dim]Epoch {epoch+1}/{epochs}", total=len(train_loader))

            for batch in train_loader:
                if is_text or is_text_batch(batch):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    target = batch["labels"].to(device)
                    optimizer.zero_grad()
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    data, target = batch
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()
                    output = model(data)

                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
                progress.advance(batch_task)

            progress.remove_task(batch_task)
            avg_loss = epoch_loss / num_batches

            results.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "test_loss": 0.0,
                "accuracy": 0.0,
            })

            progress.advance(epoch_task)
            progress.update(epoch_task, description=f"[cyan]Training [dim]loss={avg_loss:.3f}")

    if test_loader is not None:
        acc, test_loss = evaluate(model, test_loader, device, is_text=is_text)
        results[-1]["test_loss"] = test_loss
        results[-1]["accuracy"] = acc

    return results


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


def generate_synthetic_text_batch(
    batch_size: int,
    max_len: int,
    vocab_size: int,
    num_classes: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

    seq_lens = torch.randint(5, max_len, (batch_size,), device=device)
    random_tokens = torch.randint(999, vocab_size, (batch_size, max_len), device=device)

    range_matrix = torch.arange(max_len, device=device).expand(batch_size, max_len)
    mask = range_matrix < seq_lens.unsqueeze(1)

    input_ids = input_ids + (random_tokens * mask.long())
    input_ids[:, 0] = 101  # CLS token

    sep_indices = (seq_lens - 1).unsqueeze(1)
    input_ids.scatter_(1, sep_indices, 102)  # SEP token

    attention_mask = mask.long()
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    return input_ids, attention_mask, labels


def train_dp_sgd(
    model: nn.Module,
    train_loader: DataLoader,
    public_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    noise_multiplier: float,
    sample_rate: float,
    max_grad_norm: float,
    epochs: int,
    device: torch.device,
    kfac: bool = False,
    use_public_data: bool = True,
    use_pink_noise: bool = False,
    kfac_config: Optional[KFACConfig] = None,
    test_loader: Optional[DataLoader] = None,
    num_classes: int = 10,
    is_text: bool = False,
    vocab_size: int = 30522,
    max_seq_len: int = 128,
    delta: float = 1e-5,
) -> List[Dict[str, Any]]:
    kfac_config = kfac_config or KFACConfig()
    accountant = RDPAccountant()
    recorder = KFACRecorder(model) if kfac else None
    public_iter = iter(itertools.cycle(public_loader)) if kfac else None
    criterion = nn.CrossEntropyLoss(reduction="sum")
    console = Console()

    results = []
    inv_A: Dict[str, torch.Tensor] = {}
    inv_G: Dict[str, torch.Tensor] = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        epoch_task = progress.add_task("[cyan]DP Training", total=epochs)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            batch_task = progress.add_task(f"[dim]Epoch {epoch+1}/{epochs}", total=len(train_loader))

            for batch_idx, batch in enumerate(train_loader):
                if is_text or is_text_batch(batch):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    target = batch["labels"].to(device)
                    batch_size = input_ids.size(0)
                else:
                    data, target = batch
                    data, target = data.to(device), target.to(device)
                    batch_size = data.size(0)

                model.zero_grad(set_to_none=True)

                if kfac and recorder is not None and batch_idx % len(train_loader) == 0:
                    recorder.enable()

                    if use_public_data and public_iter is not None:
                        pub_batch = next(public_iter)
                        if is_text or is_text_batch(pub_batch):
                            pub_input_ids = pub_batch["input_ids"].to(device)
                            pub_mask = pub_batch["attention_mask"].to(device)
                            pub_target = pub_batch["labels"].to(device)
                            pub_out = model(input_ids=pub_input_ids, attention_mask=pub_mask)
                        else:
                            pub_data, pub_target = pub_batch
                            pub_data = pub_data.to(device)
                            pub_target = pub_target.to(device)
                            pub_out = model(pub_data)
                    else:
                        if is_text or is_text_batch(batch):
                            noise_input_ids, noise_mask, noise_labels = generate_synthetic_text_batch(
                                batch_size=batch_size,
                                max_len=max_seq_len,
                                vocab_size=vocab_size,
                                num_classes=num_classes,
                                device=device,
                            )
                            pub_out = model(input_ids=noise_input_ids, attention_mask=noise_mask)
                            pub_target = noise_labels
                        else:
                            input_shape = (data.size(1), data.size(2), data.size(3))
                            if use_pink_noise:
                                pub_data = generate_pink_noise(batch_size, input_shape, device)
                            else:
                                pub_data = generate_white_noise(batch_size, input_shape, device)
                            pub_target = torch.randint(0, num_classes, (batch_size,), device=device)
                            pub_out = model(pub_data)

                    pub_loss = F.cross_entropy(pub_out, pub_target)
                    pub_loss.backward()

                    cov = compute_covariances(model, recorder.activations, recorder.backprops)
                    inv_A, inv_G = compute_inverse_sqrt(cov, damping=kfac_config.damping)

                    recorder.disable()
                    recorder.clear()
                    model.zero_grad(set_to_none=True)

                if is_text or is_text_batch(batch):
                    output = model(input_ids=input_ids, attention_mask=attention_mask)
                else:
                    output = model(data)

                loss = criterion(output, target)
                loss.backward()

                if kfac and inv_A and inv_G:
                    precondition_per_sample_gradients(model, inv_A, inv_G)

                clip_and_noise_gradients(model, noise_multiplier, max_grad_norm, batch_size)
                optimizer.step()
                accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

                epoch_loss += loss.item() / batch_size
                num_batches += 1
                progress.advance(batch_task)

            progress.remove_task(batch_task)
            avg_loss = epoch_loss / num_batches
            epsilon = accountant.get_epsilon(delta=delta)

            results.append({
                "epoch": epoch + 1,
                "train_loss": avg_loss,
                "test_loss": 0.0,
                "accuracy": 0.0,
                "epsilon_spent": epsilon,
            })

            progress.advance(epoch_task)
            progress.update(epoch_task, description=f"[cyan]DP Training [dim]loss={avg_loss:.3f} [yellow]ε={epsilon:.2f}")

    if recorder is not None:
        recorder.remove()

    if test_loader is not None:
        acc, test_loss = evaluate(model, test_loader, device, is_text=is_text)
        results[-1]["test_loss"] = test_loss
        results[-1]["accuracy"] = acc

    return results


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        public_loader: DataLoader,
        device: torch.device,
        learning_rate: float = 1e-3,
        optimizer_type: str = "adam",
        is_text: bool = False,
        vocab_size: int = 30522,
        max_seq_len: int = 128,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.public_loader = public_loader
        self.device = device
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type
        self.is_text = is_text
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

    def _create_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        params = [p for p in model.parameters() if p.requires_grad]

        if self.optimizer_type.lower() == "adam":
            return torch.optim.Adam(params, lr=self.learning_rate)
        elif self.optimizer_type.lower() == "adamw":
            return torch.optim.AdamW(params, lr=self.learning_rate)
        elif self.optimizer_type.lower() == "sgd":
            return torch.optim.SGD(params, lr=self.learning_rate, momentum=0.9)
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

    def train_baseline(
        self,
        epochs: int,
        seed: int,
    ) -> List[Dict[str, Any]]:
        set_seed(seed)

        model = self._fresh_model()
        model = model.to(self.device)
        optimizer = self._create_optimizer(model)

        return train_plain_sgd(
            model=model,
            train_loader=self.train_loader,
            optimizer=optimizer,
            epochs=epochs,
            device=self.device,
            test_loader=self.test_loader,
            is_text=self.is_text,
        )

    def train_dp_sgd(
        self,
        epochs: int,
        epsilon: float,
        delta: float,
        max_grad_norm: float,
        seed: int,
    ) -> List[Dict[str, Any]]:
        set_seed(seed)

        model = self._fresh_model()
        model = model.to(self.device)
        model = GradSampleModule(model, batch_first=True, loss_reduction="sum")

        optimizer = self._create_optimizer(model)

        train_size = len(self.train_loader.dataset)  # type: ignore[arg-type]
        batch_size = self.train_loader.batch_size or 256
        sample_rate = batch_size / train_size
        total_steps = epochs * (train_size // batch_size)

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            steps=total_steps,
            accountant="rdp",
        )

        return train_dp_sgd(
            model=model,
            train_loader=self.train_loader,
            public_loader=self.public_loader,
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            max_grad_norm=max_grad_norm,
            epochs=epochs,
            device=self.device,
            kfac=False,
            test_loader=self.test_loader,
            is_text=self.is_text,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            delta=delta,
        )

    def train_dp_kfac(
        self,
        epochs: int,
        epsilon: float,
        delta: float,
        max_grad_norm: float,
        seed: int,
        use_public_data: bool = True,
        use_pink_noise: bool = False,
        kfac_config: Optional[KFACConfig] = None,
    ) -> List[Dict[str, Any]]:
        set_seed(seed)

        model = self._fresh_model()
        model = model.to(self.device)
        model = GradSampleModule(model, batch_first=True, loss_reduction="sum")

        optimizer = self._create_optimizer(model)

        train_size = len(self.train_loader.dataset)  # type: ignore[arg-type]
        batch_size = self.train_loader.batch_size or 256
        sample_rate = batch_size / train_size
        total_steps = epochs * (train_size // batch_size)

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            steps=total_steps,
            accountant="rdp",
        )

        num_classes = 10
        if hasattr(self.model, "fc2"):
            num_classes = int(getattr(self.model.fc2, "out_features", 10))
        elif hasattr(self.model, "classifier"):
            num_classes = int(getattr(self.model.classifier, "out_features", 10))

        return train_dp_sgd(
            model=model,
            train_loader=self.train_loader,
            public_loader=self.public_loader,
            optimizer=optimizer,
            noise_multiplier=noise_multiplier,
            sample_rate=sample_rate,
            max_grad_norm=max_grad_norm,
            epochs=epochs,
            device=self.device,
            kfac=True,
            use_public_data=use_public_data,
            use_pink_noise=use_pink_noise,
            kfac_config=kfac_config,
            test_loader=self.test_loader,
            num_classes=num_classes,
            is_text=self.is_text,
            vocab_size=self.vocab_size,
            max_seq_len=self.max_seq_len,
            delta=delta,
        )

    def train_dp_kfac_method(
        self,
        method_name: str,
        epochs: int,
        epsilon: float,
        delta: float,
        max_grad_norm: float,
        seed: int,
        kfac_config: Optional[KFACConfig] = None,
    ) -> List[Dict[str, Any]]:
        method = get_method(method_name)
        if method is None:
            raise ValueError(f"Unknown method: {method_name}")

        set_seed(seed)
        kfac_config = kfac_config or KFACConfig()

        model = self._fresh_model()
        model = model.to(self.device)
        model = GradSampleModule(model, batch_first=True, loss_reduction="sum")

        optimizer = self._create_optimizer(model)

        train_size = len(self.train_loader.dataset)  # type: ignore[arg-type]
        batch_size = self.train_loader.batch_size or 256
        sample_rate = batch_size / train_size
        total_steps = epochs * (train_size // batch_size)

        noise_multiplier = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            steps=total_steps,
            accountant="rdp",
        )

        num_classes = 10
        if hasattr(self.model, "fc2"):
            num_classes = int(getattr(self.model.fc2, "out_features", 10))
        elif hasattr(self.model, "classifier"):
            num_classes = int(getattr(self.model.classifier, "out_features", 10))

        recorder = KFACRecorder(model)
        public_iter = iter(itertools.cycle(self.public_loader))

        if not self.is_text:
            sample_data, _ = next(iter(self.train_loader))
            input_shape: Tuple[int, ...] = (sample_data.size(1), sample_data.size(2), sample_data.size(3))
        else:
            input_shape = (self.max_seq_len,)

        inv_A, inv_G = compute_preconditioner(
            model=model,
            recorder=recorder,
            method=method,
            public_iter=public_iter,
            batch_size=batch_size,
            input_shape=input_shape,
            num_classes=num_classes,
            device=self.device,
            kfac_config=kfac_config,
            steps=kfac_config.update_freq * 10,
        )

        accountant = RDPAccountant()
        criterion = nn.CrossEntropyLoss(reduction="sum")
        results = []
        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            console=console,
        ) as progress:
            epoch_task = progress.add_task(f"[cyan]{method_name}", total=epochs)

            for epoch in range(epochs):
                model.train()
                epoch_loss = 0.0
                num_batches = 0

                batch_task = progress.add_task(f"[dim]Epoch {epoch+1}/{epochs}", total=len(self.train_loader))

                for batch in self.train_loader:
                    if self.is_text or is_text_batch(batch):
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        target = batch["labels"].to(self.device)
                        bs = input_ids.size(0)
                        model.zero_grad(set_to_none=True)
                        output = model(input_ids=input_ids, attention_mask=attention_mask)
                    else:
                        data, target = batch
                        data, target = data.to(self.device), target.to(self.device)
                        bs = data.size(0)
                        model.zero_grad(set_to_none=True)
                        output = model(data)

                    loss = criterion(output, target)
                    loss.backward()

                    if inv_A and inv_G:
                        precondition_per_sample_gradients(model, inv_A, inv_G)

                    clip_and_noise_gradients(model, noise_multiplier, max_grad_norm, bs)
                    optimizer.step()
                    accountant.step(noise_multiplier=noise_multiplier, sample_rate=sample_rate)

                    epoch_loss += loss.item() / bs
                    num_batches += 1
                    progress.advance(batch_task)

                progress.remove_task(batch_task)
                avg_loss = epoch_loss / num_batches
                eps_spent = accountant.get_epsilon(delta=delta)

                results.append({
                    "epoch": epoch + 1,
                    "train_loss": avg_loss,
                    "test_loss": 0.0,
                    "accuracy": 0.0,
                    "epsilon_spent": eps_spent,
                })

                progress.advance(epoch_task)
                progress.update(epoch_task, description=f"[cyan]{method_name} [dim]loss={avg_loss:.3f} [yellow]ε={eps_spent:.2f}")

        recorder.remove()

        acc, test_loss = evaluate(model, self.test_loader, self.device, is_text=self.is_text)
        results[-1]["test_loss"] = test_loss
        results[-1]["accuracy"] = acc

        return results

    def _fresh_model(self) -> nn.Module:
        import copy
        if hasattr(self.model, 'backbone') and hasattr(self.model, 'classifier'):
            model = nn.Module.__new__(type(self.model))
            nn.Module.__init__(model)
            model.backbone = self.model.backbone
            if hasattr(self.model, 'feature_dim'):
                model.feature_dim = self.model.feature_dim
            classifier = self.model.classifier
            if isinstance(classifier, nn.Linear):
                model.classifier = nn.Linear(
                    classifier.in_features,
                    classifier.out_features,
                )
            else:
                model.classifier = copy.deepcopy(classifier)
            return model
        return copy.deepcopy(self.model)
