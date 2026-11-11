"""Candidate-independent, provisional timing references from ordinary sequence."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import groupby
from statistics import mean, median, pstdev
from typing import Any

TIMING_REFERENCE_SCHEMA = "generic-nonpoly-timing-v1"


@dataclass(frozen=True)
class TimingWindowConfig:
    """Fixed QC and sampling; no construct/head identity or labels are consulted."""

    window_bases: int = 96
    max_probe_windows: int = 32
    selected_windows: int = 4
    min_valid_windows: int = 2
    min_mean_q: float = 15.0
    max_homopolymer_bases: int = 6
    max_base_fraction: float = 0.7
    min_entropy_bits: float = 1.2

    def __post_init__(self) -> None:
        if (
            not 1
            <= self.min_valid_windows
            <= self.selected_windows
            <= self.max_probe_windows
            <= 64
        ):
            raise ValueError("invalid timing window counts")
        if self.window_bases < 8 or self.max_homopolymer_bases < 1:
            raise ValueError("invalid timing window geometry")
        if not math.isfinite(self.min_mean_q) or self.min_mean_q < 0:
            raise ValueError("invalid timing quality threshold")
        if not 0 < self.max_base_fraction <= 1 or not 0 <= self.min_entropy_bits <= 2:
            raise ValueError("invalid timing complexity thresholds")


DEFAULT_TIMING_WINDOW_CONFIG = TimingWindowConfig()


def missing_timing_reference(reason: str) -> dict[str, str | int | float]:
    """Keep missing timing evidence distinct from zero-nt or read failure."""
    return {
        "reference_available": 0,
        "reference_status": reason,
        "reference_samples_per_base": "",
        "reference_log_rate_sd": "",
        "window_attempt_count": 0,
        "window_qc_pass_count": 0,
        "window_valid_count": 0,
    }


def timing_vector(row: dict[str, object]) -> tuple[float, float, float]:
    """Encode [log1p(samples/base), log-rate dispersion, available] without fitting."""
    if str(row["reference_available"]) == "0":
        if row["reference_status"] == "ok":
            raise ValueError("missing timing reference cannot have status ok")
        return 0.0, 0.0, 0.0
    if str(row["reference_available"]) != "1" or row["reference_status"] != "ok":
        raise ValueError("invalid timing availability/status")
    rate = float(row["reference_samples_per_base"])  # type: ignore[arg-type]
    dispersion = float(row["reference_log_rate_sd"])  # type: ignore[arg-type]
    if (
        not math.isfinite(rate)
        or rate <= 0
        or not math.isfinite(dispersion)
        or dispersion < 0
    ):
        raise ValueError("invalid timing reference values")
    return math.log1p(rate), dispersion, 1.0


def _spaced_indices(total: int, count: int) -> list[int]:
    """Return deterministic, spread-out indices, without label-dependent ranking."""
    return (
        [0]
        if count == 1
        else [round(i * (total - 1) / (count - 1)) for i in range(count)]
    )


def extract_timing_reference(
    sequence: str,
    qualities: Sequence[int] | None,
    moves: Sequence[int],
    *,
    trim_samples: int,
    raw_samples: int,
    config: TimingWindowConfig = DEFAULT_TIMING_WINDOW_CONFIG,
) -> tuple[dict[str, str | int | float], list[dict[str, str | int | float]]]:
    """Sample ordinary windows across the read, even when no Poly is detected.

    Input sequence and moves follow acquisition orientation; output intervals
    are original raw samples [start,end). QC cannot prove windows are non-Poly.
    Only called-sequence quality/complexity and provisional moves enter this API.
    """
    sequence = sequence.upper()
    missing = missing_timing_reference
    if qualities is None or len(qualities) != len(sequence):
        return missing("missing_or_mismatched_qualities"), []
    if len(sequence) < config.window_bases * config.min_valid_windows:
        return missing("sequence_too_short"), []
    if (
        not moves
        or moves[0] <= 0
        or trim_samples < 0
        or raw_samples <= 0
        or any(value < 0 for value in moves[1:])
    ):
        return missing("invalid_move_table"), []
    stride = int(moves[0])
    if trim_samples + (len(moves) - 1) * stride > raw_samples + stride - 1:
        return missing("moves_outside_raw_signal"), []
    positions = [
        trim_samples + index * stride
        for index, count in enumerate(moves[1:])
        for _ in range(count)
    ]
    if len(positions) != len(sequence):
        return missing("move_sequence_length_mismatch"), []
    if positions[-1] >= raw_samples:
        return missing("moves_outside_raw_signal"), []
    positions.append(min(raw_samples, trim_samples + (len(moves) - 1) * stride))
    count = min(config.max_probe_windows, len(sequence) // config.window_bases)
    starts = (
        [
            round(i * (len(sequence) - config.window_bases) / (count - 1))
            for i in range(count)
        ]
        if count > 1
        else [0]
    )
    windows: list[dict[str, str | int | float]] = []
    for index, start in enumerate(starts):
        end = start + config.window_bases
        window = sequence[start:end]
        counts = Counter(window)
        fractions = [value / len(window) for value in counts.values()]
        entropy = -sum(value * math.log2(value) for value in fractions)
        q = mean(qualities[start:end])
        reason = ""
        if not math.isfinite(q) or q < config.min_mean_q:
            reason = "low_quality"
        elif set(counts) - set("ACGT"):
            reason = "ambiguous_bases"
        elif max(fractions) > config.max_base_fraction:
            reason = "dominant_base"
        elif (
            max(len(list(run)) for _, run in groupby(window))
            > config.max_homopolymer_bases
        ):
            reason = "homopolymer"
        elif entropy < config.min_entropy_bits:
            reason = "low_entropy"
        elif start and positions[start] == positions[start - 1]:
            reason = "window_start_shares_move_block"
        elif end < len(sequence) and positions[end] == positions[end - 1]:
            reason = "window_end_shares_move_block"
        elif positions[end] <= positions[start]:
            reason = "empty_window_signal"
        windows.append(
            {
                "window_index": index,
                "start_base": start,
                "end_base": end,
                "start_sample": positions[start],
                "end_sample": positions[end],
                "mean_q": q,
                "status": reason or "ok",
                "selected": 0,
                "samples_per_base": (positions[end] - positions[start])
                / config.window_bases
                if not reason
                else "",
            }
        )
    valid = [row for row in windows if row["status"] == "ok"]
    selected = (
        [
            valid[index]
            for index in _spaced_indices(
                len(valid), min(len(valid), config.selected_windows)
            )
        ]
        if valid
        else []
    )
    for row in selected:
        row["selected"] = 1
    result = missing("insufficient_valid_windows")
    result.update(
        window_attempt_count=len(windows),
        window_qc_pass_count=len(valid),
        window_valid_count=len(selected),
    )
    if len(selected) >= config.min_valid_windows:
        rates = [float(row["samples_per_base"]) for row in selected]
        result.update(
            reference_available=1,
            reference_status="ok",
            reference_samples_per_base=median(rates),
            reference_log_rate_sd=pstdev(math.log(rate) for rate in rates),
        )
    return result, windows


def timing_reference_from_record(
    record: object,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read deployable BAM evidence; labels and reference alignments are unused."""
    from typing import Any, cast

    bam_record = cast(Any, record)
    if (
        bam_record.has_tag("pi")
        or (bam_record.has_tag("dx") and bam_record.get_tag("dx") != 0)
        or any(op == 5 for op, _ in (bam_record.cigartuples or []))
    ):
        return missing_timing_reference("unsupported_split_duplex_hardclip"), []
    if not all(bam_record.has_tag(tag) for tag in ("mv", "ts", "ns")):
        return missing_timing_reference("missing_move_tags"), []
    return extract_timing_reference(
        bam_record.get_forward_sequence() or "",
        bam_record.get_forward_qualities(),
        list(bam_record.get_tag("mv")),
        trim_samples=int(bam_record.get_tag("ts")),
        raw_samples=int(bam_record.get_tag("ns")),
    )
