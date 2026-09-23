"""Single-GPU training utilities for frozen Bonito Poly heads."""

from __future__ import annotations

import json
import platform
import random
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from holypoly.losses import PolyAuxiliaryLoss, PolyLossWeights
from holypoly.model import HolypolyModel
from holypoly.poly_heads import PolyClass, ReadStatus
from holypoly.training_data import PolyTrainingBatch


@dataclass(frozen=True)
class EpochResult:
    """Loss and metric aggregates for one epoch."""

    examples: int
    batches: int
    global_step: int
    losses: dict[str, float]
    location_accuracy: float
    location_macro_f1: float
    poly_iou: float
    length_mae_nt: float
    status_accuracy: float


@dataclass(frozen=True)
class CheckpointContext:
    """Reproducibility metadata stored with every training checkpoint."""

    random_seed: int
    bonito_commit: str
    model_config: Mapping[str, Any]
    manifest_schema_version: str
    loss_weights: PolyLossWeights
    git_commit: str


@dataclass(frozen=True)
class ResumeState:
    """Epoch and step restored from a checkpoint."""

    epoch: int
    global_step: int
    random_seed: int


def set_random_seed(seed: int) -> None:
    """Seed Python and PyTorch CPU/CUDA random number generators."""
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def require_single_cuda_device(device: torch.device) -> None:
    """Require one explicitly indexed CUDA device."""
    if device.type != "cuda" or device.index is None:
        raise ValueError(
            "device must be an explicitly indexed CUDA device such as cuda:0"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if device.index >= torch.cuda.device_count():
        raise ValueError(f"CUDA device index is unavailable: {device}")


def create_optimizer(
    model: HolypolyModel,
    *,
    learning_rate: float,
    weight_decay: float,
) -> torch.optim.Optimizer:
    """Create AdamW over only the two Poly heads."""
    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative")
    parameters = [parameter for parameter in model.trainable_parameters()]
    if not parameters:
        raise ValueError("model exposes no trainable Poly-head parameters")
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    expected = {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    if optimized != expected:
        raise ValueError(
            "optimizer parameters must exactly match trainable model parameters"
        )
    return optimizer


def train_one_epoch(
    model: HolypolyModel,
    batches: Iterable[PolyTrainingBatch],
    criterion: PolyAuxiliaryLoss,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    gradient_clip_norm: float,
    global_step: int = 0,
) -> EpochResult:
    """Train both Poly heads for one epoch on one explicit GPU."""
    require_single_cuda_device(device)
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be greater than zero")
    model.train()
    accumulator = _MetricAccumulator()
    for batch in batches:
        device_batch = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(device_batch.signal, device_batch.raw_padding_mask)
        losses = criterion(
            output,
            device_batch.location_targets,
            device_batch.true_length_nt,
            device_batch.status_targets,
        )
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(
            list(model.trainable_parameters()), gradient_clip_norm
        )
        optimizer.step()
        global_step += 1
        accumulator.update(output, device_batch, losses)
    return accumulator.result(global_step)


def validate_one_epoch(
    model: HolypolyModel,
    batches: Iterable[PolyTrainingBatch],
    criterion: PolyAuxiliaryLoss,
    *,
    device: torch.device,
    global_step: int,
) -> EpochResult:
    """Evaluate frozen-encoder auxiliary heads without gradients."""
    require_single_cuda_device(device)
    model.eval()
    accumulator = _MetricAccumulator()
    with torch.no_grad():
        for batch in batches:
            device_batch = _batch_to_device(batch, device)
            output = model(device_batch.signal, device_batch.raw_padding_mask)
            losses = criterion(
                output,
                device_batch.location_targets,
                device_batch.true_length_nt,
                device_batch.status_targets,
            )
            accumulator.update(output, device_batch, losses)
    return accumulator.result(global_step)


def save_checkpoint(
    path: Path,
    *,
    model: HolypolyModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    context: CheckpointContext,
    data_root: Path | None = None,
) -> None:
    """Atomically save model, optimizer, RNG, software, and label metadata."""
    _ensure_output_is_safe(path, data_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "random_seed": context.random_seed,
        "bonito_commit": context.bonito_commit,
        "model_config": dict(context.model_config),
        "manifest_schema_version": context.manifest_schema_version,
        "loss_config": asdict(context.loss_weights),
        "git_commit": context.git_commit,
        "python_version": sys.version,
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "python_random_state": random.getstate(),
        "torch_random_state": torch.get_rng_state(),
        "cuda_random_states": torch.cuda.get_rng_state_all(),
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    temporary_path.replace(path)


def load_checkpoint(
    path: Path,
    *,
    model: HolypolyModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> ResumeState:
    """Restore model, optimizer, epoch, step, and random states."""
    require_single_cuda_device(device)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    random.setstate(payload["python_random_state"])
    torch.set_rng_state(payload["torch_random_state"].cpu())
    torch.cuda.set_rng_state_all(
        [state.cpu() for state in payload["cuda_random_states"]]
    )
    return ResumeState(
        epoch=int(payload["epoch"]),
        global_step=int(payload["global_step"]),
        random_seed=int(payload["random_seed"]),
    )


def append_loss_log(
    path: Path,
    result: EpochResult,
    *,
    epoch: int,
    split: str | None = None,
) -> None:
    """Append one JSON record containing every loss and metric."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"epoch": epoch, **asdict(result)}
    if split is not None:
        record["split"] = split
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


class _MetricAccumulator:
    def __init__(self) -> None:
        self.examples = 0
        self.batches = 0
        self.loss_sums: dict[str, float] = {}
        self.location_confusion = torch.zeros((len(PolyClass), len(PolyClass)))
        self.poly_intersection = 0
        self.poly_union = 0
        self.length_absolute_error = 0.0
        self.length_examples = 0
        self.status_correct = 0

    def update(
        self,
        output: Any,
        batch: PolyTrainingBatch,
        losses: Mapping[str, Tensor],
    ) -> None:
        batch_size = batch.signal.shape[0]
        self.examples += batch_size
        self.batches += 1
        for name, value in losses.items():
            if name.endswith("_examples"):
                continue
            self.loss_sums[name] = (
                self.loss_sums.get(name, 0.0)
                + float(value.detach().item()) * batch_size
            )

        valid = (~output.feature_padding_mask) & (batch.location_targets != -100)
        predictions = output.location.location_logits.argmax(dim=-1)
        targets = batch.location_targets
        for target_class in PolyClass:
            for predicted_class in PolyClass:
                count = torch.count_nonzero(
                    valid
                    & (targets == int(target_class))
                    & (predictions == int(predicted_class))
                )
                self.location_confusion[target_class, predicted_class] += count.cpu()
        predicted_poly = predictions != int(PolyClass.BACKGROUND)
        target_poly = targets != int(PolyClass.BACKGROUND)
        self.poly_intersection += int(
            torch.count_nonzero(valid & predicted_poly & target_poly).item()
        )
        self.poly_union += int(
            torch.count_nonzero(valid & (predicted_poly | target_poly)).item()
        )

        length_mask = batch.status_targets != int(ReadStatus.FAILURE)
        if torch.any(length_mask):
            self.length_absolute_error += float(
                torch.abs(
                    output.length.length_nt[length_mask]
                    - batch.true_length_nt[length_mask]
                )
                .sum()
                .item()
            )
            self.length_examples += int(length_mask.sum().item())
        shared_status = (
            output.location.status_logits + output.length.status_logits
        ) / 2
        self.status_correct += int(
            torch.count_nonzero(
                shared_status.argmax(dim=-1) == batch.status_targets
            ).item()
        )

    def result(self, global_step: int) -> EpochResult:
        if self.batches == 0 or self.examples == 0:
            raise ValueError("epoch must contain at least one batch")
        losses = {name: value / self.examples for name, value in self.loss_sums.items()}
        total_locations = float(self.location_confusion.sum().item())
        correct_locations = float(self.location_confusion.diag().sum().item())
        f1_values = []
        for class_index in range(len(PolyClass)):
            true_positive = float(self.location_confusion[class_index, class_index])
            false_positive = (
                float(self.location_confusion[:, class_index].sum()) - true_positive
            )
            false_negative = (
                float(self.location_confusion[class_index, :].sum()) - true_positive
            )
            denominator = 2 * true_positive + false_positive + false_negative
            f1_values.append(
                0.0 if denominator == 0 else 2 * true_positive / denominator
            )
        return EpochResult(
            examples=self.examples,
            batches=self.batches,
            global_step=global_step,
            losses=losses,
            location_accuracy=(
                0.0 if total_locations == 0 else correct_locations / total_locations
            ),
            location_macro_f1=sum(f1_values) / len(f1_values),
            poly_iou=(
                0.0
                if self.poly_union == 0
                else self.poly_intersection / self.poly_union
            ),
            length_mae_nt=(
                float("nan")
                if self.length_examples == 0
                else self.length_absolute_error / self.length_examples
            ),
            status_accuracy=self.status_correct / self.examples,
        )


def _batch_to_device(
    batch: PolyTrainingBatch, device: torch.device
) -> PolyTrainingBatch:
    return PolyTrainingBatch(
        signal=batch.signal.to(device),
        raw_padding_mask=batch.raw_padding_mask.to(device),
        raw_lengths_samples=batch.raw_lengths_samples.to(device),
        location_targets=batch.location_targets.to(device),
        true_length_nt=batch.true_length_nt.to(device),
        status_targets=batch.status_targets.to(device),
        read_ids=batch.read_ids,
    )


def _ensure_output_is_safe(path: Path, data_root: Path | None) -> None:
    resolved_path = path.expanduser().resolve()
    if data_root is not None:
        resolved_data_root = data_root.expanduser().resolve()
        if resolved_path == resolved_data_root or resolved_path.is_relative_to(
            resolved_data_root
        ):
            raise ValueError(
                f"checkpoint path must not be inside data_root: {resolved_path}"
            )
