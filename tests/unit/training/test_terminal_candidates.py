"""Synthetic candidate coordinates, with no BAM/POD5 or label dependencies."""

from __future__ import annotations

import pytest

from holypoly.inference.candidates import candidate_signal_bounds, terminal_candidates


def test_terminal_candidates_keep_types_interruptions_and_real_edges() -> None:
    """Allow two interruptions; do not label internal/short runs as zero."""
    sequence = "CG" * 20 + "A" * 6 + "CG" + "A" * 6 + "CC"
    candidates = terminal_candidates(sequence, end_margin_bases=2)
    assert len(candidates) == 1
    assert candidates[0].poly_type == "polyA"
    assert (candidates[0].start_base, candidates[0].end_base) == (40, 54)
    assert candidates[0].target_bases == 12
    assert candidates[0].span_bases == 14
    assert candidates[0].edge_distance_bases == 2
    assert not terminal_candidates("CG" * 20 + "A" * 12 + "CG" * 20, end_margin_bases=5)
    assert not terminal_candidates("A" * 6 + "CCC" + "A" * 6)
    both = terminal_candidates("T" * 12 + "CG" * 20 + "A" * 14)
    assert [candidate.poly_type for candidate in both] == ["polyA", "polyT"]


@pytest.mark.parametrize("sequence", ["", "CGCGCG", "AAAA", "TTTT"])
def test_no_candidate_is_not_a_zero_length_label(sequence: str) -> None:
    assert terminal_candidates(sequence) == ()


def test_move_bounds_use_actual_stride_trim_and_acquisition_coordinates() -> None:
    """Stride is from mv, not the length model; tail end cannot exceed real samples."""
    assert candidate_signal_bounds(
        [5, 1, 0, 1, 1, 0, 1],
        trim_samples=7,
        raw_samples=37,
        sequence_length=4,
        start_base=2,
        end_base=4,
    ) == (22, 37)


@pytest.mark.parametrize(
    ("moves", "length", "start", "end", "error"),
    [
        ([5, 1, 0], 3, 0, 1, "move_sequence_length_mismatch"),
        ([0, 1], 1, 0, 1, "invalid_move_table"),
        ([5, 1, -1], 1, 0, 1, "invalid_move_table"),
        ([5, 2, 1], 3, 1, 3, "candidate_start_shares_move_block"),
        ([5, 1, 2], 3, 0, 2, "candidate_end_shares_move_block"),
    ],
)
def test_invalid_or_ambiguous_moves_are_not_silently_truncated(
    moves: list[int],
    length: int,
    start: int,
    end: int,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        candidate_signal_bounds(
            moves,
            trim_samples=0,
            raw_samples=100,
            sequence_length=length,
            start_base=start,
            end_base=end,
        )
