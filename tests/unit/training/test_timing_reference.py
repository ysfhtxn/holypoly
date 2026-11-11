"""Torch-free tests of candidate-independent provisional timing windows."""

import math

import pytest

from holypoly.inference.timing_reference import (
    TimingWindowConfig,
    extract_timing_reference,
    missing_timing_reference,
    timing_vector,
)


def test_ordinary_windows_available_without_any_poly_candidate() -> None:
    """Ordinary reads, including possible zero-nt reads, must not need an A/T run."""
    sequence = "ACGT" * 120
    row, windows = extract_timing_reference(
        sequence,
        [40] * len(sequence),
        [6] + [1, 0] * len(sequence),
        trim_samples=10,
        raw_samples=10 + 12 * len(sequence),
    )
    assert row["reference_available"] == 1
    assert row["reference_samples_per_base"] == 12
    assert row["reference_log_rate_sd"] == 0
    assert row["window_valid_count"] == 4
    assert sum(window["selected"] for window in windows) == 4
    assert timing_vector(row) == (math.log1p(12), 0, 1)
    for window in windows:
        assert window["start_sample"] == 10 + window["start_base"] * 12
        assert window["end_sample"] == 10 + window["end_base"] * 12


@pytest.mark.parametrize(
    "sequence,reason",
    [
        ("A" * 400, "insufficient_valid_windows"),
        ("T" * 400, "insufficient_valid_windows"),
        ("ACGT", "sequence_too_short"),
    ],
)
def test_low_complexity_or_short_reads_remain_missing_not_zero(
    sequence: str, reason: str
) -> None:
    row, _ = extract_timing_reference(
        sequence,
        [40] * len(sequence),
        [6] + [1] * len(sequence),
        trim_samples=0,
        raw_samples=6 * len(sequence),
    )
    assert row["reference_available"] == 0
    assert row["reference_status"] == reason
    assert timing_vector(row) == (0, 0, 0)


def test_qc_and_move_errors_produce_explicit_missing_evidence() -> None:
    sequence = "ACGT" * 100
    args = dict(trim_samples=0, raw_samples=2400)
    row, _ = extract_timing_reference(sequence, None, [6] + [1] * 400, **args)
    assert row["reference_status"] == "missing_or_mismatched_qualities"
    row, _ = extract_timing_reference(sequence, [0] * 400, [6] + [1] * 400, **args)
    assert row["reference_available"] == 0
    row, _ = extract_timing_reference(sequence, [40] * 400, [6, 1], **args)
    assert row["reference_status"] == "move_sequence_length_mismatch"
    row, _ = extract_timing_reference(sequence, [40] * 400, [6] + [1] * 401, **args)
    assert row["reference_status"] == "moves_outside_raw_signal"


def test_directions_use_same_quality_complexity_and_timing_rules() -> None:
    sequence = "A" * 30 + "ACGT" * 200
    reverse = sequence.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    first, _ = extract_timing_reference(
        sequence,
        [40] * len(sequence),
        [6] + [1] * len(sequence),
        trim_samples=0,
        raw_samples=6 * len(sequence),
    )
    second, _ = extract_timing_reference(
        reverse,
        [40] * len(sequence),
        [6] + [1] * len(sequence),
        trim_samples=0,
        raw_samples=6 * len(sequence),
    )
    assert first == second


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_rates_cannot_become_neural_inputs(value: float) -> None:
    row = {
        **missing_timing_reference("ok"),
        "reference_available": 1,
        "reference_samples_per_base": value,
        "reference_log_rate_sd": 0,
    }
    with pytest.raises(ValueError, match="invalid timing"):
        timing_vector(row)


def test_invalid_window_configuration_rejected() -> None:
    with pytest.raises(ValueError):
        TimingWindowConfig(min_valid_windows=5)
