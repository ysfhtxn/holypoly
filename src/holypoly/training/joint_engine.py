"""Explicit joint updates on unchanged Poly chunks and separate head chunks."""

from __future__ import annotations

from collections.abc import Iterable, Sequence, Sized
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
from holypoly.training.prefetch import prefetch_batches as ordered_prefetch
from holypoly.training.progress import BatchProfiler, timed_stage
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
    prefetch_batches: int = 0,
    progress_desc: str | None = None,
    total_examples: int | None = None,
) -> dict[str, float | int]:
    """Evaluate sequence chunks with optional bounded CPU prefetch."""
    with ordered_prefetch(batches, prefetch_batches) as prepared:
        return _validate_sequences(
            model,
            prepared,
            device=device,
            progress_desc=progress_desc,
            total_examples=total_examples,
        )


def _validate_sequences(
    model: JointHolypolyModel,
    batches: Iterable[Sequence[SequenceExample]],
    *,
    device: torch.device,
    progress_desc: str | None = None,
    total_examples: int | None = None,
) -> dict[str, float | int]:
    """Evaluate all accepted held-out head chunks, including decoded edit rate."""
    require_single_cuda_device(device)
    model.eval()
    losses, edits, bases, count = 0.0, 0, 0, 0
    with (
        torch.no_grad(),
        tqdm(
            desc=progress_desc,
            total=total_examples,
            disable=progress_desc is None,
            unit="read",
            dynamic_ncols=True,
            mininterval=1.0,
        ) as progress,
    ):
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
                progress.update(1)
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
    prefetch_batches: int = 0,
    monitor_frozen_sequences: bool = True,
    profile_batches: int = 0,
    profile_result: dict[str, object] | None = None,
) -> tuple[EpochResult, dict[str, Any]]:
    """Prefetch independent CPU streams while keeping joint GPU updates ordered."""
    use_sequences = model.unfreeze_last_layers > 0 or monitor_frozen_sequences
    if not use_sequences:
        _require_frozen_basecaller(model)
    with (
        ordered_prefetch(length_batches, prefetch_batches) as lengths,
        ordered_prefetch(
            sequence_batches if use_sequences else (), prefetch_batches
        ) as sequences,
    ):
        return _train_joint_epoch(
            model,
            lengths,
            sequences,
            criterion,
            optimizer,
            lambda_basecall=lambda_basecall,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
            global_step=global_step,
            progress_desc=progress_desc,
            use_sequences=use_sequences,
            profile_batches=profile_batches,
            profile_result=profile_result,
        )


def _train_joint_epoch(
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
    use_sequences: bool = True,
    profile_batches: int = 0,
    profile_result: dict[str, object] | None = None,
) -> tuple[EpochResult, dict[str, Any]]:
    """One optimizer update combines length gradients and head-sequence gradients.

    Each length batch pairs with one independently shuffled sequence batch.
    Short sequence loaders restart without caching signal tensors. Each sequence
    graph is freed immediately, limiting memory while preserving mean gradients.
    """
    require_single_cuda_device(device)
    model.train()
    accumulator = _MetricAccumulator()
    iterator = iter(sequence_batches)
    total: Tensor | None = None
    count = 0
    used: set[str] = set()
    profiler = BatchProfiler(device, profile_batches)
    batches = iter(length_batches)
    # Keep a known total for ETA even though next() is timed explicitly.
    total_batches = len(length_batches) if isinstance(length_batches, Sized) else None
    profile_reported = False
    with tqdm(
        total=total_batches,
        desc=progress_desc,
        disable=progress_desc is None,
        unit="batch",
        mininterval=1.0,
        dynamic_ncols=True,
    ) as progress:
        while True:
            # An exhausted iterator is not counted as a measured batch.
            profiler.index += 1
            previous_seconds = profiler.seconds.copy() if profiler.active else None
            with profiler.measure("length_data_wait"):
                try:
                    batch = next(batches)
                except StopIteration:
                    # Restore after leaving measure(), which records in finally.
                    break
            if profiler.active:
                profiler.measured += 1
            with profiler.measure("transfer"):
                device_batch = _batch_to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model.forward_poly(
                device_batch.signal,
                device_batch.raw_padding_mask,
                timing_reference=device_batch.timing_reference,
                measure=profiler.measure,
            )
            with profiler.measure("auxiliary_loss_backward"):
                losses = criterion(
                    output,
                    device_batch.location_targets,
                    device_batch.true_length_nt,
                    device_batch.status_targets,
                )
                if not torch.isfinite(losses["total_loss"]):
                    raise ValueError("nonfinite auxiliary loss")
                losses["total_loss"].backward()
            if use_sequences:
                with profiler.measure("sequence_data_wait"):
                    try:
                        examples = next(iterator)
                    except StopIteration:
                        iterator = iter(sequence_batches)
                        examples = next(iterator)
                if not examples:
                    raise ValueError("empty sequence batch")
                with profiler.measure("sequence_forward_crf_backward"):
                    for example in examples:
                        value = reference_crf_loss(
                            model,
                            sequence_scores(model, example, device),
                            example,
                            device,
                        )
                        if model.unfreeze_last_layers:
                            (lambda_basecall * value / len(examples)).backward()
                        detached = value.detach().double()
                        total = detached if total is None else total + detached
                        count += 1
                        used.add(example.record.read_id)
            with profiler.measure("optimizer"):
                torch.nn.utils.clip_grad_norm_(
                    list(model.trainable_parameters()),
                    gradient_clip_norm,
                    error_if_nonfinite=True,
                )
                optimizer.step()
            global_step += 1
            with profiler.measure("metrics"):
                accumulator.update(output, device_batch, losses)
            progress.update(1)
            if (
                profile_batches
                and profiler.index + 1 == profiler.warmup + profiler.limit
            ):
                report = profiler.report()
                profile_reported = True
                if profile_result is not None:
                    profile_result.update(report)
    if previous_seconds is not None:
        profiler.seconds = previous_seconds
    if profile_batches and not profile_reported:
        report = profiler.report()
        if profile_result is not None:
            profile_result.update(report)
    sequence: dict[str, Any] = {
        "sequence_examples": count,
        "sequence_unique_reads": len(used),
    }
    if total is not None:
        sequence["basecalling_loss"] = float(total) / count
    else:
        sequence["sequence_metrics_source"] = "disabled_frozen"
    return accumulator.result(global_step), sequence


def joint_log_record(
    result: EpochResult,
    sequence: dict[str, Any],
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
    if "basecalling_loss" in sequence:
        losses = row["losses"]
        losses["basecalling_loss"] = sequence["basecalling_loss"]
        losses["weighted_basecalling_loss"] = (
            lambda_basecall * sequence["basecalling_loss"]
        )
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
    prefetch_batches: int = 0,
    progress_desc: str | None = None,
    sequence_examples: int | None = None,
    cached_sequence: dict[str, Any] | None = None,
    timings: dict[str, float] | None = None,
) -> tuple[EpochResult, dict[str, Any]]:
    """Evaluate all length and accepted sequence validation chunks."""
    durations = timings if timings is not None else {}
    label = progress_desc or "Validation"
    with (
        timed_stage(label + " length/status", device, durations),
        ordered_prefetch(length_batches, prefetch_batches) as prepared,
    ):
        result = validate_one_epoch(
            model,
            prepared,
            criterion,
            device=device,
            global_step=global_step,
            progress_desc=label + " length/status" if progress_desc else None,
            poly_only=True,
        )
    if cached_sequence is not None:
        _require_frozen_basecaller(model)
        print(f"[{label} head sequence] reused frozen baseline", flush=True)
        durations[label + " head sequence"] = 0.0
        return result, {**cached_sequence, "sequence_metrics_source": "frozen_baseline"}
    with timed_stage(label + " head sequence", device, durations):
        return result, validate_sequences(
            model,
            sequence_batches,
            device=device,
            prefetch_batches=prefetch_batches,
            progress_desc=label + " head sequence" if progress_desc else None,
            total_examples=sequence_examples,
        )


def _require_frozen_basecaller(model: JointHolypolyModel) -> None:
    """Reject skipped CRF computation when any basecaller parameter can change."""
    if model.unfreeze_last_layers or any(
        parameter.requires_grad for parameter in model.basecaller.parameters()
    ):
        raise ValueError("sequence monitoring reuse requires a fully frozen basecaller")
