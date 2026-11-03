"""Explicit joint updates on unchanged Poly chunks and separate head chunks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import asdict
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor
from tqdm import tqdm

from holypoly.models.joint_model import JointHolypolyModel
from holypoly.training.data import PolyTrainingBatch
from holypoly.training.engine import (
    EpochResult,
    _batch_to_device,
    _MetricAccumulator,
    require_single_cuda_device,
    validate_one_epoch,
)
from holypoly.training.losses import PolyAuxiliaryLoss
from holypoly.training.sequence_data import SequenceExample


def sequence_scores(
    model: JointHolypolyModel, example: SequenceExample, device: torch.device
) -> Tensor:
    """Return [valid T,1,states] scores; alignment padding never enters CRF loss."""
    samples = example.signal.numel()
    if example.signal.ndim != 1 or samples < 1:
        raise ValueError("sequence signal must be nonempty [raw_samples]")
    padded = (samples + 11) // 12 * 12
    signal = functional.pad(example.signal.to(device), (0, padded - samples))[
        None, None, :
    ]
    scores = model.basecaller(signal)
    expected = padded // 6
    if scores.ndim != 3 or scores.shape[:2] != (expected, 1):
        raise ValueError(
            "basecalling output does not match supported Bonito v5 stride/interface"
        )
    return scores[: (samples + 5) // 6].float().contiguous()


def reference_crf_loss(
    model: JointHolypolyModel,
    scores: Tensor,
    example: SequenceExample,
    device: torch.device,
) -> Tensor:
    """Call the unchanged Bonito native CTC-CRF objective, not torch CTCLoss."""
    state_len = int(model.basecaller.seqdist.state_len)
    size = example.targets.numel()
    if size < state_len or size + 1 - state_len > scores.shape[0]:
        raise ValueError(
            "sequence target cannot align to the available valid CRF timesteps"
        )
    targets = example.targets.to(device=device, dtype=torch.long)[None, :]
    lengths = torch.tensor([size], dtype=torch.long, device=device)
    value: Tensor = model.basecaller.loss(scores, targets, lengths)
    if value.ndim != 0 or not torch.isfinite(value):
        raise ValueError("nonfinite or nonscalar native reference CTC-CRF loss")
    return value


def edit_distance(reference: str, prediction: str) -> int:
    """Compute Levenshtein distance for small head windows without new dependencies."""
    previous = list(range(len(prediction) + 1))
    for index, ref_base in enumerate(reference, 1):
        current = [index]
        for column, pred_base in enumerate(prediction, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (ref_base != pred_base),
                )
            )
        previous = current
    return previous[-1]


def validate_sequences(
    model: JointHolypolyModel,
    batches: Iterable[Sequence[SequenceExample]],
    *,
    device: torch.device,
) -> dict[str, float | int]:
    """Evaluate all accepted held-out head chunks, including decoded edit rate."""
    require_single_cuda_device(device)
    model.eval()
    losses, edits, bases, count = 0.0, 0, 0, 0
    with torch.no_grad():
        for examples in batches:
            for example in examples:
                scores = sequence_scores(model, example, device)
                losses += float(reference_crf_loss(model, scores, example, device))
                predictions = model.basecaller.decode_batch(scores)
                if len(predictions) != 1:
                    raise ValueError(
                        "basecaller decode_batch returned unexpected count"
                    )
                edits += edit_distance(example.record.sequence_target, predictions[0])
                bases += len(example.record.sequence_target)
                count += 1
    if count == 0:
        raise ValueError("sequence evaluation has no accepted chunks")
    return {
        "basecalling_loss": losses / count,
        "sequence_examples": count,
        "reference_edit_rate": edits / bases,
        "reference_bases": bases,
    }


def train_joint_epoch(
    model: JointHolypolyModel,
    length_batches: Iterable[PolyTrainingBatch],
    sequence_batches: Iterable[Sequence[SequenceExample]],
    criterion: PolyAuxiliaryLoss,
    optimizer: Any,
    *,
    lambda_basecall: float,
    device: torch.device,
    gradient_clip_norm: float,
    global_step: int,
    progress_desc: str | None = None,
) -> tuple[EpochResult, dict[str, float | int]]:
    """One optimizer update combines length gradients and head-sequence gradients.

    Each length batch pairs with one independently shuffled sequence batch.
    Short sequence loaders restart without caching signal tensors. Each sequence
    graph is freed immediately, limiting memory while preserving mean gradients.
    """
    require_single_cuda_device(device)
    model.train()
    accumulator = _MetricAccumulator()
    iterator = iter(sequence_batches)
    total, count = 0.0, 0
    used: set[str] = set()
    for batch in tqdm(
        length_batches, desc=progress_desc, disable=progress_desc is None, unit="batch"
    ):
        device_batch = _batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        output = model(device_batch.signal, device_batch.raw_padding_mask)
        losses = criterion(
            output,
            device_batch.location_targets,
            device_batch.true_length_nt,
            device_batch.status_targets,
        )
        if not torch.isfinite(losses["total_loss"]):
            raise ValueError("nonfinite auxiliary loss")
        losses["total_loss"].backward()
        try:
            examples = next(iterator)
        except StopIteration:
            iterator = iter(sequence_batches)
            examples = next(iterator)
        if not examples:
            raise ValueError("empty sequence batch")
        for example in examples:
            value = reference_crf_loss(
                model, sequence_scores(model, example, device), example, device
            )
            (lambda_basecall * value / len(examples)).backward()
            total += float(value.detach())
            count += 1
            used.add(example.record.read_id)
        torch.nn.utils.clip_grad_norm_(
            list(model.trainable_parameters()),
            gradient_clip_norm,
            error_if_nonfinite=True,
        )
        optimizer.step()
        global_step += 1
        accumulator.update(output, device_batch, losses)
    return accumulator.result(global_step), {
        "basecalling_loss": total / count,
        "sequence_examples": count,
        "sequence_unique_reads": len(used),
    }


def joint_log_record(
    result: EpochResult,
    sequence: dict[str, float | int],
    *,
    epoch: int,
    split: str,
    lambda_basecall: float,
    loss_config: dict[str, Any],
    joint_config: dict[str, Any],
) -> dict[str, Any]:
    """Keep ordinary length logs plot-compatible and record separate CRF metrics."""
    row = {
        "epoch": epoch,
        "split": split,
        **asdict(result),
        **sequence,
        "loss_config": loss_config,
        "joint_config": joint_config,
    }
    losses = row["losses"]
    losses["basecalling_loss"] = sequence["basecalling_loss"]
    losses["weighted_basecalling_loss"] = lambda_basecall * sequence["basecalling_loss"]
    losses["total_loss"] += losses["weighted_basecalling_loss"]
    return row


def validate_joint_epoch(
    model: JointHolypolyModel,
    length_batches: Iterable[PolyTrainingBatch],
    sequence_batches: Iterable[Sequence[SequenceExample]],
    criterion: PolyAuxiliaryLoss,
    *,
    device: torch.device,
    global_step: int,
) -> tuple[EpochResult, dict[str, float | int]]:
    """Evaluate all length and accepted sequence validation chunks."""
    result = validate_one_epoch(
        model, length_batches, criterion, device=device, global_step=global_step
    )
    return result, validate_sequences(model, sequence_batches, device=device)
