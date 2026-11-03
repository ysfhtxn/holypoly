"""Balanced calibration of existing classifiers on frozen pooled features."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from tqdm import tqdm

from holypoly.models.model import HolypolyModel
from holypoly.models.poly_heads import ReadStatus
from holypoly.training.data import PolyTrainingBatch
from holypoly.training.engine import require_single_cuda_device
from holypoly.training.evaluate import status_agreement_metrics


@dataclass(frozen=True)
class StatusFeatures:
    """CPU tensors: two [N,H] pools, [N] status/nominal/raw-length arrays."""

    location: Tensor
    length: Tensor
    targets: Tensor
    nominal_lengths_nt: Tensor
    raw_lengths_nt: Tensor


def freeze_for_status(model: HolypolyModel) -> list[nn.Parameter]:
    """Freeze all parameters and buffers except the two existing final classifiers."""
    model.requires_grad_(False)
    model.eval()
    parameters = []
    for head in (model.location_head, model.length_head):
        head.status_classifier.requires_grad_(True)
        parameters.extend(head.status_classifier.parameters())
    return parameters


def balanced_status_weights(targets: Sequence[int]) -> list[float]:
    """Compute inverse-frequency weights from train only, averaging to one.

    Absent failure examples receive no target weight; failure logits still enter
    the three-class softmax denominator. This does not teach failure detection.
    """
    counts = [targets.count(int(status)) for status in ReadStatus]
    if sum(counts) != len(targets) or not all(counts[:2]) or counts[2]:
        raise ValueError("status calibration requires present/zero train labels only")
    return [len(targets) / (2 * count) if count else 0.0 for count in counts]


def extract_status_features(
    model: HolypolyModel,
    batches: Iterable[PolyTrainingBatch],
    nominal_lengths_nt: Sequence[float],
    *,
    device: torch.device,
    description: str | None = None,
) -> StatusFeatures:
    """Capture actual classifier inputs without changing pooling or model forward.

    The caller supplies non-shuffled batches. Only detached pooled features are
    retained in RAM; raw signal and encoder timestep features are not cached.
    """
    require_single_cuda_device(device)
    model.eval()
    pools: list[list[Tensor]] = [[], []]
    handles = []
    targets, lengths = [], []

    def capture(
        destination: list[Tensor],
    ) -> Callable[[nn.Module, tuple[Tensor, ...]], None]:
        def hook(_module: nn.Module, inputs: tuple[Tensor, ...]) -> None:
            destination.append(inputs[0].detach().cpu().clone())

        return hook

    for head, destination in zip(
        (model.location_head, model.length_head), pools, strict=True
    ):
        handles.append(
            head.status_classifier.register_forward_pre_hook(capture(destination))
        )
    try:
        with torch.no_grad():
            for batch in tqdm(batches, desc=description, disable=description is None):
                output = model(
                    batch.signal.to(device), batch.raw_padding_mask.to(device)
                )
                lengths.append(output.length.length_nt.detach().cpu())
                targets.append(batch.status_targets.cpu())
    finally:
        for handle in handles:
            handle.remove()
    if not targets:
        raise ValueError("status feature extraction received no examples")
    labels = torch.cat(targets)
    if len(nominal_lengths_nt) != labels.numel() or torch.any(
        (labels < 0) | (labels > 1)
    ):
        raise ValueError(
            "non-shuffled present/zero labels and matching nominals required"
        )
    return StatusFeatures(
        torch.cat(pools[0]),
        torch.cat(pools[1]),
        labels,
        torch.tensor(nominal_lengths_nt, dtype=torch.float32),
        torch.cat(lengths),
    )


def status_logits(model: HolypolyModel, location: Tensor, length: Tensor) -> Tensor:
    """Reuse the original average of [N,3] status logits without normalization."""
    return (
        model.location_head.status_classifier(location)
        + model.length_head.status_classifier(length)
    ) / 2


def balanced_status_loss(logits: Tensor, targets: Tensor, weights: Tensor) -> Tensor:
    """Mean per-read CE with train-derived importance weights, not batch weights."""
    return (
        functional.cross_entropy(logits, targets, reduction="none") * weights[targets]
    ).mean()


def train_status_epoch(
    model: HolypolyModel,
    features: StatusFeatures,
    optimizer: Any,
    weights: Tensor,
    *,
    device: torch.device,
    batch_size: int,
    seed: int,
    gradient_clip_norm: float,
) -> int:
    """Update classifier parameters only; encoder and shared projections stay eval."""
    require_single_cuda_device(device)
    model.eval()
    order = torch.randperm(
        features.targets.numel(), generator=torch.Generator().manual_seed(seed)
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    steps = 0
    for indices in order.split(batch_size):
        optimizer.zero_grad(set_to_none=True)
        logits = status_logits(
            model,
            features.location[indices].to(device),
            features.length[indices].to(device),
        )
        loss = balanced_status_loss(
            logits, features.targets[indices].to(device), weights
        )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite balanced status loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            parameters, gradient_clip_norm, error_if_nonfinite=True
        )
        optimizer.step()
        steps += 1
    return steps


def status_metrics(
    model: HolypolyModel,
    features: StatusFeatures,
    weights: Tensor,
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate fixed argmax, including zero recall and 20-nt false-zero fraction."""
    model.eval()
    with torch.no_grad():
        logits = status_logits(
            model, features.location.to(device), features.length.to(device)
        )
        loss = float(balanced_status_loss(logits, features.targets.to(device), weights))
        predicted = logits.argmax(-1).cpu().tolist()
    rows = [
        {
            "label_status": ReadStatus(int(target)).name.lower(),
            "shared_status": ReadStatus(int(prediction)).name.lower(),
            "prediction_status": "ok",
        }
        for target, prediction in zip(features.targets.tolist(), predicted, strict=True)
    ]
    result = status_agreement_metrics(rows)
    zero_count = result["zero_nt_reads"]
    if not zero_count:
        raise ValueError("status metrics require zero examples")
    zero_recall = (
        result["confusion_true_rows_predicted_columns"]["zero_nt"]["zero_nt"]
        / zero_count
    )
    twenty = [
        row
        for row, nt in zip(rows, features.nominal_lengths_nt.tolist(), strict=True)
        if nt == 20
    ]
    twenty_metrics = status_agreement_metrics(twenty)
    twenty_recall = twenty_metrics["poly_present_recall"]
    if twenty_recall is None:
        raise ValueError("status calibration requires a nominal 20-nt group")
    score = (
        2 * zero_recall * twenty_recall / (zero_recall + twenty_recall)
        if zero_recall + twenty_recall
        else 0.0
    )
    result.update(
        zero_nt_recall=zero_recall,
        twenty_nt_present_recall=twenty_recall,
        twenty_nt_false_zero_fraction=twenty_metrics[
            "poly_present_false_zero_fraction"
        ],
        selection_score=score,
        status_accuracy=result["accuracy"],
        losses={
            "total_loss": loss,
            "status_loss": loss,
            "weighted_status_loss": loss,
            "length_loss": 0.0,
            "location_loss": 0.0,
        },
        length_mae_nt=float(
            (features.raw_lengths_nt - features.nominal_lengths_nt).abs().mean()
        ),
        examples=len(rows),
    )
    return result


def status_selection_key(metrics: dict[str, Any]) -> tuple[float, float, float]:
    """Select validation harmonic recall, then all-positive recall and lower CE."""
    return (
        metrics["selection_score"],
        metrics["poly_present_recall"],
        -metrics["losses"]["status_loss"],
    )
