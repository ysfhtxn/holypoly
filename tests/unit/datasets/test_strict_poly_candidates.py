"""Signal-only strict-rule tests; no real reads, models or calibration fitting."""

from __future__ import annotations

import numpy as np
import pytest

from holypoly.datasets.strict_poly_candidates import (
    StrictPolyConfig,
    _boundary_evidence,
    _extend,
    _interval_ious,
    _window_stats,
    detect_strict_poly_platforms,
    estimate_external_speed,
    interval_iou,
    restrict_platforms_to_interval,
    select_strict_poly_candidate,
)


def platform_signal(*, reverse=False, terminal=False):
    """Known synthetic interval in a fluctuating background, with a short spike."""
    rng = np.random.default_rng(17)
    signal = rng.normal(100, 20, 5000)
    end = 5000 if terminal else 4100
    signal[3600:end] = 60 + rng.normal(0, 0.8, end - 3600)
    signal[3800:3803] = 140
    return signal[::-1].copy() if reverse else signal


@pytest.mark.parametrize("signal", [[], [1, np.nan], [np.inf], [[1, 2]], [1j], [True]])
def test_invalid_signal_is_not_a_negative_label(signal):
    with pytest.raises(ValueError, match="signal"):
        detect_strict_poly_platforms(signal, 5000, "forward")


@pytest.mark.parametrize("fs", [0, -1, np.nan, np.inf, True, "5000"])
def test_invalid_sample_rate_errors(fs):
    with pytest.raises(ValueError, match="fs"):
        detect_strict_poly_platforms([1, 2], fs, "forward")


@pytest.mark.parametrize("reverse", [False, True])
def test_multiscale_platform_preserves_coordinates_and_short_spike(reverse):
    signal = platform_signal(reverse=reverse)
    orientation = "reverse" if reverse else "forward"
    config = StrictPolyConfig()
    result = detect_strict_poly_platforms(signal, 5000, orientation, config)
    speed = estimate_external_speed(
        np.arange(0, 5001, 10), result["regions"], 5000, config
    )
    selected = select_strict_poly_candidate(result, 50, speed, config)
    assert selected["status"] == "preferred_suspected_poly"
    region = selected["selected"]
    expected = (900, 1400) if reverse else (3600, 4100)
    assert abs(region["start"] - expected[0]) <= 30
    assert abs(region["end"] - expected[1]) <= 30
    spike = 1198 if reverse else 3801
    assert region["start"] <= spike < region["end"]
    assert len(region["support_window_samples"]) >= 2
    assert region["boundary_status"] == "two_sided_signal_support"
    assert result == detect_strict_poly_platforms(signal, 5000, orientation, config)


def test_wrong_direction_and_repeated_actual_scale_do_not_promote():
    signal = platform_signal()
    wrong = detect_strict_poly_platforms(signal, 5000, "reverse")
    assert wrong["regions"] and not any(
        r["morphology_supported"] for r in wrong["regions"]
    )
    duplicate_scales = StrictPolyConfig(window_ms=(0.1, 0.2, 0.3))
    result = detect_strict_poly_platforms(signal, 5000, "forward", duplicate_scales)
    assert not any(r["morphology_supported"] for r in result["regions"])
    assert all(len(r["support_window_samples"]) <= 1 for r in result["regions"])


def test_terminal_platform_has_unknown_completeness_and_missing_speed_defers():
    config = StrictPolyConfig()
    result = detect_strict_poly_platforms(
        platform_signal(terminal=True), 5000, "forward"
    )
    supported = [r for r in result["regions"] if r["morphology_supported"]]
    assert len(supported) == 1
    assert supported[0]["near_raw_edge"]
    assert supported[0]["boundary_status"] == "one_sided_or_edge_unknown"
    choice = select_strict_poly_candidate(
        result, 150, {"status": "unavailable"}, config
    )
    assert (
        choice["status"] == "review_timing_unavailable" and choice["selected"] is None
    )


def test_nominal_mismatch_zero_conflict_and_multiple_candidates_never_force_choice():
    config = StrictPolyConfig()
    result = detect_strict_poly_platforms(platform_signal(), 5000, "forward")
    speed = {"status": "available", "samples_per_nt": 10.0}
    assert (
        select_strict_poly_candidate(result, 0, speed, config)["status"]
        == "review_zero_construct_conflict"
    )
    assert (
        select_strict_poly_candidate(result, 500, speed, config)["status"]
        == "review_duration_prior_mismatch"
    )
    region = next(r for r in result["regions"] if r["morphology_supported"])
    ambiguous = {"regions": [region, {**region, "start": 4200, "end": 4300}]}
    assert (
        select_strict_poly_candidate(ambiguous, 50, speed, config)["status"]
        == "review_multiple_supported_candidates"
    )


def test_constant_signal_is_only_a_platform_and_two_separate_levels_are_not_joined():
    result = detect_strict_poly_platforms(np.ones(1000), 5000, "forward")
    assert not any(r["morphology_supported"] for r in result["regions"])
    config = StrictPolyConfig()
    assert (
        select_strict_poly_candidate(result, 0, {}, config)["status"]
        == "review_uninformative_signal"
    )
    signal = platform_signal()
    rng = np.random.default_rng(5)
    signal[4400:4700] = 140 + rng.normal(0, 0.8, 300)
    regions = detect_strict_poly_platforms(signal, 5000, "forward")["regions"]
    assert not any(r["start"] < 3900 and r["end"] > 4500 for r in regions)


def test_speed_excludes_platforms_and_margin_in_original_order():
    positions = np.arange(0, 2001, 10)
    result = estimate_external_speed(
        positions, [{"start": 200, "end": 1600}], 5000, StrictPolyConfig()
    )
    # [0,200), [1600,1800) overlap expanded exclusion; only [1800,2000) remains.
    assert result["window_count"] == 1
    assert result["status"] == "insufficient_external_timing_windows"


def test_vectorized_iou_matches_pairwise_for_nested_adjacent_and_random_intervals():
    rng = np.random.default_rng(73)
    starts = rng.integers(0, 1000, 500)
    ends = starts + rng.integers(1, 1000, 500)
    for region in [
        dict(start=0, end=100),
        dict(start=100, end=200),
        dict(start=0, end=2000),
        dict(start=1900, end=2000),
    ]:
        expected = [
            interval_iou(region, dict(start=int(a), end=int(b)))
            for a, b in zip(starts, ends, strict=True)
        ]
        np.testing.assert_array_equal(_interval_ious(region, starts, ends), expected)


@pytest.mark.parametrize("max_gap", [0, 1, 2, 5])
def test_local_extension_matches_full_read_fixed_level_mask(max_gap):
    """Compare optimized local traversal to the original full-mask definition."""
    rng = np.random.default_rng(73)
    medians = np.repeat(rng.normal(size=40), 11) + rng.normal(0, 0.05, 440)
    shape_ok = rng.random(440) > 0.15
    medians[30:40] = 1.25
    medians[40:50] = 1.0  # Exact threshold equality and a same-level gap.
    shape_ok[45:47] = False
    for seed in np.flatnonzero(shape_ok):
        full_mask = shape_ok & (np.abs(medians - medians[seed]) <= 0.25)
        for direction in [-1, 1]:
            expected, failures = int(seed), 0
            for index in range(
                int(seed) + direction, len(medians) if direction == 1 else -1, direction
            ):
                if full_mask[index]:
                    expected, failures = index, 0
                else:
                    failures += 1
                    if failures > max_gap:
                        break
            actual = _extend(medians, shape_ok, int(seed), direction, max_gap, 0.25)
            assert actual == expected


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("size", [12, 21, 50])
def test_binary_search_context_matches_full_window_masks(side, size):
    """Preserve full-window context, medians, contrast and continuation evidence."""
    config = StrictPolyConfig()
    stats = _window_stats(platform_signal(), size, 2)
    for region in [
        dict(start=2, end=70),
        dict(start=3600, end=4090),
        dict(start=4900, end=4999),
        dict(start=3613, end=4057),
    ]:
        inside = (stats["center"] >= region["start"]) & (
            stats["center"] < region["end"]
        )
        outside = (
            (stats["end"] <= region["start"])
            & (stats["start"] >= region["start"] - 100)
            if side == "left"
            else (stats["start"] >= region["end"])
            & (stats["end"] <= region["end"] + 100)
        )
        actual = _boundary_evidence(region, stats, side, 5000, config)
        if np.count_nonzero(outside) < 3:
            assert actual == {
                "supported": False,
                "status": "insufficient_context",
                "contrast": None,
            }
            continue
        level = np.median(stats["m"][inside])
        contrast = float(
            np.median(stats["s"][outside])
            / max(float(np.median(stats["s"][inside])), 1e-6)
        )
        shift = float(abs(np.median(stats["m"][outside]) - level))
        indices = np.flatnonzero(outside)
        nearest = indices[-3:] if side == "left" else indices[:3]
        continuation = bool(
            np.all(
                (stats["s"][nearest] <= 0.25)
                & (stats["d"][nearest] <= 0.25)
                & (np.abs(stats["m"][nearest] - level) <= 0.25)
            )
        )
        supported = not continuation and (contrast >= 2 or shift > 0.25)
        assert actual == {
            "supported": supported,
            "status": "same_level_continuation"
            if continuation
            else "transition_supported"
            if supported
            else "weak_transition",
            "contrast": contrast,
            "level_shift": shift,
        }


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("length", [0, 50])
def test_head_platforms_do_not_compete_or_conflict_with_weak_zero(reverse, length):
    """An eligible-strand platform in the head is not a Poly competitor."""
    rng = np.random.default_rng(17)
    signal = platform_signal() if length else rng.normal(100, 20, 5000)
    signal[2600:2900] = 140 + rng.normal(0, 0.8, 300)
    if reverse:
        signal = signal[::-1].copy()
    config = StrictPolyConfig()
    detection = detect_strict_poly_platforms(
        signal, 5000, "reverse" if reverse else "forward"
    )
    full_regions = detection["regions"].copy()
    assert sum(r["morphology_supported"] for r in full_regions) == (2 if length else 1)
    start, end = (0, 2000) if reverse else (3000, 5000)
    scoped = restrict_platforms_to_interval(detection, start, end)
    speed = {"status": "available", "samples_per_nt": 10.0}
    decision = select_strict_poly_candidate(scoped, length, speed, config)
    assert decision["status"] == (
        "preferred_suspected_poly" if length else "weak_zero_no_supported_candidate"
    )
    assert detection["regions"] == full_regions
    assert scoped["signal_informative"] == detection["signal_informative"]


def test_scope_preserves_inside_ambiguity_zero_conflicts_and_signal_quality():
    config = StrictPolyConfig()
    speed = {"status": "available", "samples_per_nt": 10.0}
    regions = [
        dict(start=3100, end=3500, morphology_supported=True),
        dict(start=4000, end=4500, morphology_supported=True),
    ]
    scoped = restrict_platforms_to_interval({"regions": regions}, 3000, 5000)
    assert (
        select_strict_poly_candidate(scoped, 50, speed, config)["status"]
        == "review_multiple_supported_candidates"
    )
    assert (
        select_strict_poly_candidate(scoped, 0, speed, config)["status"]
        == "review_zero_construct_conflict"
    )
    scoped["signal_informative"] = False
    assert (
        select_strict_poly_candidate(scoped, 0, speed, config)["status"]
        == "review_uninformative_signal"
    )


def test_scope_does_not_clip_crossing_candidates_or_change_raw_coordinates():
    regions = [
        dict(start=2900, end=3100),
        dict(start=4900, end=5100),
        dict(start=3000, end=5000),
    ]
    scoped = restrict_platforms_to_interval({"regions": regions}, 3000, 5000)
    assert scoped["regions"] == [regions[2]]
    assert scoped["regions"][0] is regions[2]
    assert not restrict_platforms_to_interval({"regions": regions[:2]}, 3000, 5000)[
        "has_platform"
    ]


@pytest.mark.parametrize(
    "bounds", [(-1, 20), (20, 20), (30, 20), (1.5, 20), (True, 20)]
)
def test_invalid_selection_interval_is_rejected(bounds):
    with pytest.raises(ValueError, match="selection interval"):
        restrict_platforms_to_interval({"regions": []}, *bounds)


@pytest.mark.parametrize(
    "kwargs",
    [{"min_scales": 1}, {"min_scales": 2.5}, {"hop_ms": 0}, {"contrast_min": np.nan}],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        StrictPolyConfig(**kwargs)
