"""Terminal A/T candidates, without library primers or training labels."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class TerminalCandidate:
    """A provisional interval in acquisition-oriented, 0-based base coordinates."""

    start_base: int
    end_base: int
    poly_type: str
    target_bases: int
    edge_distance_bases: int

    @property
    def span_bases(self) -> int:
        """Return sequence span, including permitted interruptions, in bases."""
        return self.end_base - self.start_base


def terminal_candidates(
    sequence: str,
    *,
    min_target_bases: int = 10,
    max_interruption_bases: int = 2,
    min_target_fraction: float = 0.8,
    end_margin_bases: int = 100,
) -> tuple[TerminalCandidate, ...]:
    """Find A/T-rich runs close to either sequence end, longest evidence first.

    Thresholds select candidates, not biological zero labels. Runs start/end
    with their target base; terminal adapters may occupy the end margin.
    Both types are returned so callers can report unsupported directions.
    """
    if (
        min_target_bases < 1
        or max_interruption_bases < 0
        or end_margin_bases < 0
        or not 0 < min_target_fraction <= 1
    ):
        raise ValueError("invalid terminal candidate thresholds")
    sequence = sequence.upper()
    candidates = []

    def save_run(start: int, last: int, count: int, poly_type: str) -> None:
        """Keep only qualifying terminal runs, with no trailing interruption."""
        if count < min_target_bases:
            return
        span = last + 1 - start
        distance = min(start, len(sequence) - last - 1)
        if count / span >= min_target_fraction and distance <= end_margin_bases:
            candidates.append(
                TerminalCandidate(start, last + 1, poly_type, count, distance)
            )

    for target, poly_type in (("A", "polyA"), ("T", "polyT")):
        start, last, count = -1, -1, 0
        for index, base in enumerate(sequence):
            if base != target:
                continue
            if start >= 0 and index - last - 1 > max_interruption_bases:
                save_run(start, last, count, poly_type)
                start, count = -1, 0
            if start < 0:
                start = index
            last = index
            count += 1
        save_run(start, last, count, poly_type)
    return tuple(
        sorted(
            candidates,
            key=lambda row: (
                -row.target_bases,
                row.edge_distance_bases,
                row.start_base,
            ),
        )
    )


def candidate_signal_bounds(
    moves: Sequence[int],
    *,
    trim_samples: int,
    raw_samples: int,
    sequence_length: int,
    start_base: int,
    end_base: int,
) -> tuple[int, int]:
    """Map acquisition-oriented bases through mv to raw samples [start, end).

    ``moves[0]`` is the actual basecaller stride; later values are emissions
    per block. Reject ambiguous shared boundary blocks rather than truncate.
    These are provisional move boundaries, not verified Poly boundaries.
    """
    if (
        not moves
        or moves[0] <= 0
        or trim_samples < 0
        or raw_samples <= 0
        or any(value < 0 for value in moves[1:])
        or not 0 <= start_base < end_base <= sequence_length
    ):
        raise ValueError("invalid_move_table_or_base_bounds")
    stride = int(moves[0])
    if trim_samples + (len(moves) - 1) * stride > raw_samples + stride - 1:
        raise ValueError("moves_outside_raw_signal")
    positions = [
        trim_samples + block * stride
        for block, count in enumerate(moves[1:])
        for _ in range(count)
    ]
    if len(positions) != sequence_length or not positions:
        raise ValueError("move_sequence_length_mismatch")
    if positions[-1] >= raw_samples:
        raise ValueError("moves_outside_raw_signal")
    if start_base and positions[start_base] == positions[start_base - 1]:
        raise ValueError("candidate_start_shares_move_block")
    if end_base < sequence_length and positions[end_base] == positions[end_base - 1]:
        raise ValueError("candidate_end_shares_move_block")
    positions.append(min(raw_samples, trim_samples + (len(moves) - 1) * stride))
    start, end = positions[start_base], positions[end_base]
    if start >= end:
        raise ValueError("empty_candidate_signal")
    return start, end
