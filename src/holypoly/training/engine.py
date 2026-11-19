"""Shared metrics, checkpoints and single-GPU utilities for Poly training."""

from __future__ import annotations

import json
import math
import platform
import random
import shutil
import sys
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from tqdm import tqdm

from holypoly.models.model import HolypolyModel, head_config_from_checkpoint
from holypoly.models.poly_heads import PolyClass, ReadStatus
from holypoly.training.data import PolyTrainingBatch
from holypoly.training.loss_config import normalize_loss_config
from holypoly.training.losses import PolyAuxiliaryLoss, PolyLossWeights


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
    data_order_seed: int | None = None
    joint_config: Mapping[str, Any] | None = None
    length_batch_size_changes: tuple[Mapping[str, Any], ...] = ()
    input_pipeline: Mapping[str, Any] | None = None
    learning_rate_changes: tuple[Mapping[str, Any], ...] = ()
    branch_origin: Mapping[str, Any] | None = None


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
    progress_desc: str | None = None,
) -> EpochResult:
    """Train both Poly heads, optionally showing batch progress on stderr."""
    require_single_cuda_device(device)
    if gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be greater than zero")
    model.train()
    accumulator = _MetricAccumulator()
    with tqdm(
        batches,
        desc=progress_desc,
        disable=progress_desc is None,
        unit="batch",
        dynamic_ncols=True,
        mininterval=1.0,
        file=sys.stderr,
    ) as progress:
        for batch in progress:
            device_batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = _forward_batch(model, device_batch)
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
            if progress_desc is not None:
                progress.set_postfix(
                    loss=f"{losses['total_loss'].detach().item():.4f}", refresh=False
                )
    return accumulator.result(global_step)


def validate_one_epoch(
    model: HolypolyModel,
    batches: Iterable[PolyTrainingBatch],
    criterion: PolyAuxiliaryLoss,
    *,
    device: torch.device,
    global_step: int,
    progress_desc: str | None = None,
    poly_only: bool = False,
) -> EpochResult:
    """Evaluate without gradients, optionally showing batch progress on stderr."""
    require_single_cuda_device(device)
    model.eval()
    accumulator = _MetricAccumulator()
    with (
        torch.no_grad(),
        tqdm(
            batches,
            desc=progress_desc,
            disable=progress_desc is None,
            unit="batch",
            dynamic_ncols=True,
            mininterval=1.0,
            file=sys.stderr,
        ) as progress,
    ):
        for batch in progress:
            device_batch = _batch_to_device(batch, device)
            output = _forward_batch(model, device_batch, poly_only=poly_only)
            losses = criterion(
                output,
                device_batch.location_targets,
                device_batch.true_length_nt,
                device_batch.status_targets,
            )
            accumulator.update(output, device_batch, losses)
            if progress_desc is not None:
                progress.set_postfix(
                    loss=f"{losses['total_loss'].detach().item():.4f}", refresh=False
                )
    return accumulator.result(global_step)


def epoch_checkpoint_path(path: Path, epoch: int) -> Path:
    """Name a full epoch snapshot beside the latest checkpoint, starting at 1."""
    if epoch < 1:
        raise ValueError("checkpoint epoch must be greater than zero")
    return path.with_name(f"{path.stem}.epoch_{epoch:04d}{path.suffix}")


def save_checkpoint(
    path: Path,
    *,
    model: HolypolyModel,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    global_step: int,
    context: CheckpointContext,
    data_root: Path | None = None,
    work_root: Path | None = None,
    archive_epoch: bool = False,
) -> None:
    """Save a full checkpoint, optionally retaining an immutable epoch snapshot.

    Serialize once, copy to the epoch snapshot, then replace the latest file.
    Each file is published atomically. Existing epoch snapshots are rejected.
    """
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    archive_path = epoch_checkpoint_path(path, epoch) if archive_epoch else None
    archive_temporary = (
        archive_path.with_suffix(archive_path.suffix + ".tmp")
        if archive_path is not None
        else None
    )
    for output_path in (path, temporary_path, archive_path, archive_temporary):
        if output_path is not None:
            _ensure_output_is_safe(output_path, data_root, work_root=work_root)
    if archive_path is not None and (
        archive_path.exists() or archive_path.is_symlink()
    ):
        raise FileExistsError(f"epoch checkpoint already exists: {archive_path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "head_config": dict(model.head_config),
        "data_order_seed": context.data_order_seed,
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
    if context.joint_config is not None:
        payload["joint_config"] = dict(context.joint_config)
    if context.length_batch_size_changes:
        payload["length_batch_size_changes"] = list(context.length_batch_size_changes)
    if context.learning_rate_changes:
        payload["learning_rate_changes"] = list(context.learning_rate_changes)
    if context.branch_origin is not None:
        payload["branch_origin"] = dict(context.branch_origin)
    if context.input_pipeline is not None:
        payload["input_pipeline"] = dict(context.input_pipeline)
    torch.save(payload, temporary_path)
    if archive_path is not None and archive_temporary is not None:
        shutil.copyfile(temporary_path, archive_temporary)
        archive_temporary.replace(archive_path)
    temporary_path.replace(path)


def validate_joint_config(
    saved: Any,
    requested: Mapping[str, Any] | None,
    *,
    allow_batch_size_change: bool = False,
    allow_learning_rate_change: bool = False,
) -> None:
    """Match joint settings except explicitly permitted batch/rate changes."""
    expected = dict(requested) if requested is not None else None
    if saved == expected:
        return
    if isinstance(saved, Mapping) and expected is not None:
        mutable_keys = set()
        if allow_batch_size_change:
            sizes = (saved.get("length_batch_size"), expected.get("length_batch_size"))
            if all(type(size) is int and size > 0 for size in sizes):
                mutable_keys.add("length_batch_size")
        if allow_learning_rate_change:
            for key in ("head_learning_rate", "encoder_learning_rate"):
                rates = (saved.get(key), expected.get(key))
                if all(
                    isinstance(rate, (int, float))
                    and not isinstance(rate, bool)
                    and math.isfinite(rate)
                    and rate > 0
                    for rate in rates
                ):
                    mutable_keys.add(key)
        if {key: value for key, value in saved.items() if key not in mutable_keys} == {
            key: value for key, value in expected.items() if key not in mutable_keys
        }:
            return
    differing = ""
    if isinstance(saved, Mapping) and expected is not None:
        keys = sorted(
            key
            for key in saved.keys() | expected.keys()
            if key not in saved or key not in expected or saved[key] != expected[key]
        )
        differing = ": " + ", ".join(keys)
    raise ValueError(
        "joint training configuration differs from the checkpoint" + differing
    )


def load_checkpoint(
    path: Path,
    *,
    model: HolypolyModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    data_order_seed: int | None = None,
    loss_weights: PolyLossWeights | None = None,
    joint_config: Mapping[str, Any] | None = None,
    allow_batch_size_change: bool = False,
    allow_learning_rate_change: bool = False,
) -> ResumeState:
    """Restore model, optimizer, epoch, step, and random states."""
    require_single_cuda_device(device)
    if not path.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {path}")
    payload = torch.load(path, map_location=device, weights_only=False)
    validate_joint_config(
        payload.get("joint_config"),
        joint_config,
        allow_batch_size_change=allow_batch_size_change,
        allow_learning_rate_change=allow_learning_rate_change,
    )
    saved_loss_config = payload.get("loss_config")
    if not isinstance(saved_loss_config, Mapping):
        raise ValueError("checkpoint loss_config must be a mapping")
    saved_loss_config = normalize_loss_config(saved_loss_config)
    requested_loss_config = normalize_loss_config(
        asdict(loss_weights or PolyLossWeights())
    )
    # Delta has no effect outside Huber; still retain it in every saved record.
    loss_keys = ["lambda_location", "lambda_length", "lambda_status", "length_loss"]
    if requested_loss_config["length_loss"] == "huber":
        loss_keys.append("huber_delta_nt")
    if any(saved_loss_config[key] != requested_loss_config[key] for key in loss_keys):
        raise ValueError(
            "loss configuration differs from the checkpoint; "
            "use matching settings to resume, or start a new experiment"
        )
    if payload.get("data_order_seed") != data_order_seed:
        raise ValueError("data order seed differs from the checkpoint")
    head_config = head_config_from_checkpoint(payload)
    if head_config.get("head_type", "legacy") != model.head_config.get(
        "head_type", "legacy"
    ):
        raise ValueError("head architecture differs from the checkpoint")
    if head_config.get("head_type") == "tcn":
        for name in ("tcn_length_sum_scale", "tcn_length_features"):
            if head_config[name] != model.head_config[name]:
                raise ValueError(f"{name} differs from the checkpoint")
    keys: tuple[str, ...] = ("length_head_mode", "hidden_features", "dropout")
    if "head_config" not in payload:
        keys = ("length_head_mode", "hidden_features")
    if any(head_config[key] != model.head_config[key] for key in keys):
        raise ValueError("length head configuration differs from the checkpoint")
    if bool(head_config.get("use_timing_reference", False)) != bool(
        model.head_config.get("use_timing_reference", False)
    ):
        raise ValueError("timing reference architecture differs from the checkpoint")
    requested_rates = {}
    if allow_learning_rate_change and joint_config is not None:
        requested_rates = {
            "poly_heads": joint_config["head_learning_rate"],
            "encoder_tail": joint_config["encoder_learning_rate"],
        }
        for groups in (
            optimizer.param_groups,
            payload["optimizer_state_dict"]["param_groups"],
        ):
            if (
                len(groups) != 2
                or {group.get("group_name") for group in groups}
                != requested_rates.keys()
            ):
                raise ValueError("joint optimizer parameter group names differ")
        if [group["group_name"] for group in optimizer.param_groups] != [
            group["group_name"]
            for group in payload["optimizer_state_dict"]["param_groups"]
        ]:
            raise ValueError("joint optimizer parameter group order differs")
    model.load_state_dict(payload["model_state_dict"])
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    for group in optimizer.param_groups:
        if requested_rates:
            group["lr"] = requested_rates[group["group_name"]]
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
    loss_weights: PolyLossWeights | None = None,
) -> None:
    """Append one JSON record containing every loss and metric."""
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"epoch": epoch, **asdict(result)}
    if split is not None:
        record["split"] = split
    if loss_weights is not None:
        record["loss_config"] = asdict(loss_weights)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


class _MetricAccumulator:
    """Accumulate detached metrics on-device, transferring only at epoch end."""

    def __init__(self) -> None:
        self.examples = 0
        self.batches = 0
        self.loss_sums: dict[str, Tensor] = {}
        self.location_confusion: Tensor | None = None
        self.totals: Tensor | None = None

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
            weighted = value.detach().double() * batch_size
            self.loss_sums[name] = self.loss_sums.get(name, 0.0) + weighted

        valid = (~output.feature_padding_mask) & (batch.location_targets != -100)
        predictions = output.location.location_logits.argmax(dim=-1)
        targets = batch.location_targets
        classes = len(PolyClass)
        confusion = torch.zeros(
            classes * classes, dtype=torch.long, device=predictions.device
        )
        indices = targets.clamp_min(0) * classes + predictions
        confusion.scatter_add_(0, indices.flatten(), valid.long().flatten())
        confusion = confusion.reshape(classes, classes)
        self.location_confusion = (
            confusion
            if self.location_confusion is None
            else self.location_confusion + confusion
        )
        predicted_poly = predictions != int(PolyClass.BACKGROUND)
        target_poly = targets != int(PolyClass.BACKGROUND)
        length_mask = batch.status_targets != int(ReadStatus.FAILURE)
        absolute_error = torch.where(
            length_mask,
            (output.length.length_nt.detach() - batch.true_length_nt).abs(),
            0.0,
        ).sum()
        shared_status = (
            output.location.status_logits + output.length.status_logits
        ) / 2
        valid_status = batch.status_targets != -100
        totals = torch.stack(
            [
                torch.count_nonzero(valid & predicted_poly & target_poly),
                torch.count_nonzero(valid & (predicted_poly | target_poly)),
                absolute_error,
                length_mask.sum(),
                torch.count_nonzero(
                    valid_status
                    & (shared_status.argmax(dim=-1) == batch.status_targets)
                ),
                valid_status.sum(),
            ]
        ).double()
        self.totals = totals if self.totals is None else self.totals + totals

    def result(self, global_step: int) -> EpochResult:
        if self.batches == 0 or self.examples == 0:
            raise ValueError("epoch must contain at least one batch")
        assert self.location_confusion is not None and self.totals is not None
        losses = {
            name: float(value.item()) / self.examples
            for name, value in self.loss_sums.items()
        }
        confusion = self.location_confusion.cpu()
        intersection, union, error, length_count, status_correct, status_count = (
            self.totals.cpu().tolist()
        )
        total_locations = float(confusion.sum().item())
        correct_locations = float(confusion.diag().sum().item())
        f1_values = []
        for class_index in range(len(PolyClass)):
            true_positive = float(confusion[class_index, class_index])
            false_positive = float(confusion[:, class_index].sum()) - true_positive
            false_negative = float(confusion[class_index, :].sum()) - true_positive
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
            poly_iou=0.0 if union == 0 else intersection / union,
            length_mae_nt=(float("nan") if length_count == 0 else error / length_count),
            status_accuracy=(
                float("nan") if status_count == 0 else status_correct / status_count
            ),
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
        timing_reference=batch.timing_reference.to(device)
        if batch.timing_reference is not None
        else None,
    )


def _forward_batch(
    model: HolypolyModel, batch: PolyTrainingBatch, *, poly_only: bool = False
) -> Any:
    """Preserve the legacy call signature when timing fusion is disabled."""
    forward = model.forward_poly if poly_only else model
    if batch.timing_reference is not None:
        return forward(
            batch.signal,
            batch.raw_padding_mask,
            timing_reference=batch.timing_reference,
        )
    return forward(batch.signal, batch.raw_padding_mask)


def _ensure_output_is_safe(
    path: Path, data_root: Path | None, *, work_root: Path | None = None
) -> None:
    resolved_path = path.expanduser().resolve()
    if data_root is not None:
        resolved_data_root = data_root.expanduser().resolve()
        if resolved_path == resolved_data_root or resolved_path.is_relative_to(
            resolved_data_root
        ):
            raise ValueError(
                f"checkpoint path must not be inside data_root: {resolved_path}"
            )
    if work_root is not None:
        resolved_work_root = work_root.expanduser().resolve()
        if resolved_path == resolved_work_root or not resolved_path.is_relative_to(
            resolved_work_root
        ):
            raise ValueError(
                f"checkpoint path must be inside work_root: {resolved_path}"
            )
