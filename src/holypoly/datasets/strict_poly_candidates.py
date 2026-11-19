"""Strict signal-platform candidates, not verified Poly identity or nt truth.

All intervals are original-signal, 0-based, half-open sample coordinates.
The frozen initial rules intentionally prefer deferral over a forced selection.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class StrictPolyConfig:
    """Uncalibrated v1 settings; normalized statistics have no probability meaning."""

    window_ms: tuple[float, ...] = (2.0, 5.0, 10.0)
    min_window_samples: int = 12
    hop_ms: float = 0.5
    normalization_floor: float = 1e-6
    core_s_max: float = 0.15
    core_d_max: float = 0.15
    extend_s_max: float = 0.25
    extend_d_max: float = 0.25
    level_tolerance: float = 0.25
    max_gap_ms: float = 1.0
    min_duration_ms: float = 8.0
    max_bad_window_fraction: float = 0.10
    preferred_bad_window_fraction: float = 0.05
    max_median_spread: float = 0.35
    dedup_iou: float = 0.7
    min_scales: int = 2
    position_boundary_fraction: float = 0.5
    context_ms: float = 20.0
    min_context_windows: int = 3
    contrast_min: float = 2.0
    boundary_level_min: float = 0.25
    edge_margin_ms: float = 5.0
    timing_window_bases: int = 20
    timing_min_windows: int = 2
    timing_exclusion_ms: float = 5.0
    min_duration_ratio: float = 0.5
    max_duration_ratio: float = 2.0

    def __post_init__(self) -> None:
        """Reject invalid settings before opening any sequencing input."""
        for name in (
            "min_window_samples",
            "min_scales",
            "min_context_windows",
            "timing_window_bases",
            "timing_min_windows",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError(f"{name} must be an integer")
        for name, value in asdict(self).items():
            if name == "window_ms":
                if not value or any(not math.isfinite(v) or v <= 0 for v in value):
                    raise ValueError("window_ms must contain finite positive values")
            elif isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not (
            self.min_window_samples >= 2
            and self.min_scales >= 2
            and self.min_context_windows >= 1
            and self.timing_window_bases >= 1
            and self.timing_min_windows >= 2
            and 0 < self.position_boundary_fraction < 1
            and 0
            < self.preferred_bad_window_fraction
            <= self.max_bad_window_fraction
            <= 1
            and 0 < self.dedup_iou < 1
            and self.min_duration_ratio < self.max_duration_ratio
            and self.core_s_max <= self.extend_s_max
            and self.core_d_max <= self.extend_d_max
        ):
            raise ValueError("inconsistent strict candidate settings")


def interval_iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    """Return IoU without joining adjacent or different-level platforms."""
    overlap = max(
        0, min(first["end"], second["end"]) - max(first["start"], second["start"])
    )
    union = first["end"] - first["start"] + second["end"] - second["start"] - overlap
    return float(overlap / union) if union else 0.0


def _interval_ious(
    region: dict[str, Any], starts: NDArray[np.int64], ends: NDArray[np.int64]
) -> NDArray[np.float64]:
    """Vectorize positive-width interval IoUs without merging or changing ranking."""
    overlap = np.maximum(
        0, np.minimum(region["end"], ends) - np.maximum(region["start"], starts)
    )
    union = region["end"] - region["start"] + ends - starts - overlap
    return np.asarray(overlap / union, dtype=np.float64)


def _window_stats(values: NDArray[np.float64], size: int, hop: int) -> dict[str, Any]:
    """Return float64 robust statistics for unmodified strided raw windows."""
    windows = np.lib.stride_tricks.sliding_window_view(values, size)[::hop]
    median = np.median(windows, axis=1)
    half = size // 2
    return {
        "start": np.arange(len(windows)) * hop,
        "end": np.arange(len(windows)) * hop + size,
        "center": np.arange(len(windows)) * hop + half,
        "m": median,
        "s": 1.4826 * np.median(np.abs(windows - median[:, None]), axis=1),
        "d": np.abs(
            np.median(windows[:, :half], axis=1) - np.median(windows[:, half:], axis=1)
        ),
    }


def _extend(
    median: NDArray[np.float64],
    shape_ok: NDArray[np.bool_],
    seed: int,
    direction: int,
    max_gap: int,
    tolerance: float,
) -> int:
    """Evaluate only visited windows against the fixed seed, never the whole read."""
    last, failures, index = seed, 0, seed + direction
    seed_level = median[seed]
    while 0 <= index < len(median):
        if shape_ok[index] and abs(median[index] - seed_level) <= tolerance:
            last, failures = index, 0
        else:
            failures += 1
            if failures > max_gap:
                break
        index += direction
    return last


def _boundary_evidence(
    region: dict[str, Any],
    stats: dict[str, Any],
    side: str,
    fs: float,
    config: StrictPolyConfig,
) -> dict[str, Any]:
    """Inspect whole neighboring windows, with no cross-boundary sample leakage.

    Same-level low-variation context is a continuation warning even if its
    noise contrast exceeds two. Missing context is unknown, never negative.
    """
    inside_start = int(np.searchsorted(stats["center"], region["start"]))
    inside_end = int(np.searchsorted(stats["center"], region["end"]))
    inside = slice(inside_start, inside_end)
    context = max(1, round(config.context_ms * fs / 1000))
    if side == "left":
        outside_start = int(np.searchsorted(stats["start"], region["start"] - context))
        outside_end = int(np.searchsorted(stats["end"], region["start"], side="right"))
    else:
        outside_start = int(np.searchsorted(stats["start"], region["end"]))
        outside_end = int(
            np.searchsorted(stats["end"], region["end"] + context, side="right")
        )
    outside = slice(outside_start, outside_end)
    if (
        outside_end - outside_start < config.min_context_windows
        or inside_start >= inside_end
    ):
        return {"supported": False, "status": "insufficient_context", "contrast": None}
    level = float(np.median(stats["m"][inside]))
    contrast = float(
        np.median(stats["s"][outside])
        / max(float(np.median(stats["s"][inside])), config.normalization_floor)
    )
    shift = float(abs(np.median(stats["m"][outside]) - level))
    # Inspect the closest context windows, not just a distant background median.
    nearest = (
        slice(max(outside_start, outside_end - config.min_context_windows), outside_end)
        if side == "left"
        else slice(outside_start, outside_start + config.min_context_windows)
    )
    continuation = bool(
        np.all(
            (stats["s"][nearest] <= config.extend_s_max)
            & (stats["d"][nearest] <= config.extend_d_max)
            & (np.abs(stats["m"][nearest] - level) <= config.level_tolerance)
        )
    )
    supported = not continuation and (
        contrast >= config.contrast_min or shift > config.boundary_level_min
    )
    return {
        "supported": bool(supported),
        "status": "same_level_continuation"
        if continuation
        else "transition_supported"
        if supported
        else "weak_transition",
        "contrast": contrast,
        "level_shift": shift,
    }


def detect_strict_poly_platforms(
    signal: Any,
    fs: float,
    orientation: str,
    config: StrictPolyConfig | None = None,
) -> dict[str, Any]:
    """Find and grade platforms without nominal labels or sequence input.

    Signal has shape [raw_samples]. Multi-scale support is measured before
    deduplication, counting distinct actual sample widths, not repeated seeds.
    """
    config = config or StrictPolyConfig()
    values = np.asarray(signal)
    if (
        values.ndim != 1
        or not values.size
        or not np.issubdtype(values.dtype, np.number)
        or np.iscomplexobj(values)
        or not np.all(np.isfinite(values))
    ):
        raise ValueError("signal must be a nonempty finite real 1-D array")
    if (
        isinstance(fs, (bool, np.bool_))
        or not isinstance(fs, (int, float, np.integer, np.floating))
        or not math.isfinite(fs)
        or fs <= 0
    ):
        raise ValueError("fs must be finite positive Hz")
    if orientation not in {"forward", "reverse"}:
        raise ValueError("orientation must be forward or reverse")
    values = values.astype(np.float64)
    q25, q75 = np.percentile(values, [25, 75])
    raw_scale = float((q75 - q25) / 1.349)
    normalized = (values - np.median(values)) / max(
        raw_scale, config.normalization_floor
    )
    hop = max(1, round(config.hop_ms * fs / 1000))
    max_gap = math.floor(config.max_gap_ms * fs / (1000 * hop))
    all_regions: list[dict[str, Any]] = []
    scales: dict[int, dict[str, Any]] = {}
    for requested_ms in config.window_ms:
        size = max(config.min_window_samples, round(requested_ms * fs / 1000))
        if size > len(values) or size in scales:
            continue
        stats = _window_stats(normalized, size, hop)
        scales[size] = stats
        core = np.flatnonzero(
            (stats["s"] <= config.core_s_max) & (stats["d"] <= config.core_d_max)
        )
        seeds = core[np.argsort(stats["s"][core], kind="stable")]
        shape_ok = (stats["s"] <= config.extend_s_max) & (
            stats["d"] <= config.extend_d_max
        )
        covered = np.zeros(len(stats["m"]), dtype=np.bool_)
        accepted: list[dict[str, Any]] = []
        for seed_value in seeds:
            seed = int(seed_value)
            if covered[seed]:
                continue
            left, right = (
                _extend(
                    stats["m"], shape_ok, seed, -1, max_gap, config.level_tolerance
                ),
                _extend(stats["m"], shape_ok, seed, 1, max_gap, config.level_tolerance),
            )
            start, end = int(stats["center"][left]), int(stats["center"][right])
            duration_ms = (end - start) * 1000 / fs
            if duration_ms < config.min_duration_ms:
                continue
            good = shape_ok[left : right + 1] & (
                np.abs(stats["m"][left : right + 1] - stats["m"][seed])
                <= config.level_tolerance
            )
            bad = float(np.mean(~good))
            if bad > config.max_bad_window_fraction:
                continue
            q10, q90 = np.percentile(stats["m"][left : right + 1], [10, 90])
            if q90 - q10 > config.max_median_spread:
                continue
            # Coarse end is exclusive: the rightmost window's center remains
            # eligible as a future seed, matching the original interval test.
            covered[left:right] = True
            accepted.append(
                dict(
                    start=start,
                    end=end,
                    duration_ms=duration_ms,
                    window_samples=size,
                    window_ms=size * 1000 / fs,
                    bad_window_fraction=bad,
                    median_spread=float(q90 - q10),
                )
            )
        all_regions.extend(accepted)
    kept: list[dict[str, Any]] = []
    kept_starts = np.empty(len(all_regions), dtype=np.int64)
    kept_ends = np.empty(len(all_regions), dtype=np.int64)
    for region in sorted(
        all_regions,
        key=lambda item: (
            item["bad_window_fraction"],
            item["window_samples"],
            item["start"],
        ),
    ):
        if np.any(
            _interval_ious(region, kept_starts[: len(kept)], kept_ends[: len(kept)])
            > config.dedup_iou
        ):
            continue
        kept_starts[len(kept)] = region["start"]
        kept_ends[len(kept)] = region["end"]
        kept.append(region.copy())
    all_starts = np.asarray([r["start"] for r in all_regions], dtype=np.int64)
    all_ends = np.asarray([r["end"] for r in all_regions], dtype=np.int64)
    all_widths = np.asarray([r["window_samples"] for r in all_regions], dtype=np.int64)
    preferred = np.asarray(
        [
            r["bad_window_fraction"] <= config.preferred_bad_window_fraction
            for r in all_regions
        ],
        dtype=np.bool_,
    )
    for region in kept:
        # A poor-quality second scale cannot promote the representative.
        supported = (
            _interval_ious(region, all_starts, all_ends) > config.dedup_iou
        ) & preferred
        support = set(all_widths[supported].tolist())
        region["support_window_samples"] = sorted(support)
        fraction = (region["start"] + region["end"]) / (2 * len(values))
        position_ok = (
            fraction > config.position_boundary_fraction
            if orientation == "forward"
            else fraction < config.position_boundary_fraction
        )
        region["position_consistent"] = bool(position_ok)
        stats = scales[region["window_samples"]]
        left_evidence = _boundary_evidence(region, stats, "left", fs, config)
        right_evidence = _boundary_evidence(region, stats, "right", fs, config)
        region.update(left_boundary=left_evidence, right_boundary=right_evidence)
        sequence_side = left_evidence if orientation == "forward" else right_evidence
        reasons = []
        if not position_ok:
            reasons.append("strand_position_conflict")
        if region["bad_window_fraction"] > config.preferred_bad_window_fraction:
            reasons.append("bad_window_fraction_above_preferred")
        if len(support) < config.min_scales:
            reasons.append("insufficient_distinct_scale_support")
        if not sequence_side["supported"]:
            reasons.append("sequence_side_boundary_" + sequence_side["status"])
        edge = round(config.edge_margin_ms * fs / 1000)
        near_edge = region["start"] <= edge or len(values) - region["end"] <= edge
        region["boundary_status"] = (
            "two_sided_signal_support"
            if left_evidence["supported"]
            and right_evidence["supported"]
            and not near_edge
            else "one_sided_or_edge_unknown"
            if left_evidence["supported"] or right_evidence["supported"]
            else "boundary_undetermined"
        )
        region["near_raw_edge"] = near_edge
        region["morphology_supported"] = not reasons
        region["review_reasons"] = reasons
    kept.sort(key=lambda item: (item["start"], item["end"]))
    return {
        "has_platform": bool(kept),
        "regions": kept,
        "signal_informative": raw_scale > config.normalization_floor,
    }


def estimate_external_speed(
    move_positions: Any,
    regions: list[dict[str, Any]],
    fs: float,
    config: StrictPolyConfig,
) -> dict[str, Any]:
    """Median samples/nt from nonoverlapping 20-base windows outside all platforms.

    move_positions has shape [called_bases + 1], in raw acquisition order.
    This is an ordinary-sequence timing prior, not a calibrated Poly speed.
    """
    positions = np.asarray(move_positions, dtype=np.int64)
    if positions.ndim != 1 or len(positions) < 2 or np.any(np.diff(positions) < 0):
        raise ValueError("invalid acquisition-order move positions")
    margin = round(config.timing_exclusion_ms * fs / 1000)
    rates = []
    size = config.timing_window_bases
    # Merge exclusion coverage only for speed-window lookup, never candidates.
    intervals: list[list[int]] = []
    for region in sorted(regions, key=lambda item: item["start"]):
        start, end = region["start"] - margin, region["end"] + margin
        if intervals and start <= intervals[-1][1]:
            intervals[-1][1] = max(intervals[-1][1], end)
        else:
            intervals.append([start, end])
    cursor = 0
    for index in range(0, len(positions) - size, size):
        start, end = int(positions[index]), int(positions[index + size])
        while cursor < len(intervals) and intervals[cursor][1] <= start:
            cursor += 1
        overlaps = cursor < len(intervals) and intervals[cursor][0] < end
        if end > start and not overlaps:
            rates.append((end - start) / size)
    return {
        "status": "available"
        if len(rates) >= config.timing_min_windows
        else "insufficient_external_timing_windows",
        "window_count": len(rates),
        "samples_per_nt": float(np.median(rates))
        if len(rates) >= config.timing_min_windows
        else None,
    }


def restrict_platforms_to_interval(
    detection: dict[str, Any], start_sample: int, end_sample: int
) -> dict[str, Any]:
    """Return a selection view containing only wholly contained platforms.

    Bounds are raw-signal, 0-based [start_sample, end_sample). Do not clip or
    redetect platforms: full-signal normalization, strand positions and boundary
    evidence remain intact. Keep the original detection for timing exclusions
    and audit; this view must precede ambiguity and weak-zero conflict checks.
    """
    if (
        type(start_sample) is not int
        or type(end_sample) is not int
        or not 0 <= start_sample < end_sample
    ):
        raise ValueError("selection interval must be nonempty integer raw samples")
    regions = [
        region
        for region in detection["regions"]
        if start_sample <= region["start"] < region["end"] <= end_sample
    ]
    return {**detection, "regions": regions, "has_platform": bool(regions)}


def select_strict_poly_candidate(
    detection: dict[str, Any],
    nominal_length_nt: int,
    speed: dict[str, Any],
    config: StrictPolyConfig,
) -> dict[str, Any]:
    """Select only an unambiguous supported candidate; nominal length is weak.

    Multiple morphologically supported candidates remain ambiguous, even if
    the nominal-duration prior favors one. Zero constructs are not forced zero
    when a strong waveform candidate conflicts with their weak negative label.
    """
    if nominal_length_nt < 0:
        raise ValueError("nominal length must be nonnegative")
    if not detection.get("signal_informative", True):
        return {"status": "review_uninformative_signal", "selected": None}
    supported = [
        region for region in detection["regions"] if region["morphology_supported"]
    ]
    if nominal_length_nt == 0:
        return {
            "status": "review_zero_construct_conflict"
            if supported
            else "weak_zero_no_supported_candidate",
            "selected": None,
        }
    if not supported:
        return {
            "status": "review_platforms" if detection["regions"] else "no_candidate",
            "selected": None,
        }
    if len(supported) > 1:
        return {"status": "review_multiple_supported_candidates", "selected": None}
    if speed.get("status") != "available":
        return {"status": "review_timing_unavailable", "selected": None}
    expected_samples = nominal_length_nt * float(speed["samples_per_nt"])
    if not math.isfinite(expected_samples) or expected_samples <= 0:
        raise ValueError("invalid speed prior")
    region = supported[0]
    ratio = (region["end"] - region["start"]) / expected_samples
    if not config.min_duration_ratio <= ratio <= config.max_duration_ratio:
        return {
            "status": "review_duration_prior_mismatch",
            "selected": None,
            "duration_ratio": ratio,
        }
    return {
        "status": "preferred_suspected_poly",
        "selected": region,
        "duration_ratio": ratio,
    }
