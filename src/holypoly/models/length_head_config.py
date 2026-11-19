"""Torch-free configuration for opt-in length-head comparisons."""

import math

LENGTH_HEAD_MODES = ("mean", "sum", "duration_only", "mean_duration")
HEAD_TYPES = ("legacy", "tcn")
DEFAULT_TCN_LENGTH_SUM_SCALE = 0.001
TCN_LENGTH_FEATURES = ("sum", "sum_mean_duration")
DEFAULT_TCN_LENGTH_FEATURES = "sum_mean_duration"


def validate_tcn_length_features(value: object) -> str:
    """Select the TCN regression inputs while retaining historical sum mode."""
    if not isinstance(value, str) or value not in TCN_LENGTH_FEATURES:
        raise ValueError(f"tcn_length_features must be one of {TCN_LENGTH_FEATURES}")
    return value


def validate_tcn_length_sum_scale(value: object) -> float:
    """Require a fixed, finite, positive multiplier before the TCN length MLP."""
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ValueError("tcn_length_sum_scale must be finite and positive")
    return float(value)


def validate_head_type(value: object) -> str:
    """Select a head architecture; checkpoints without this setting remain legacy."""
    if not isinstance(value, str) or value not in HEAD_TYPES:
        raise ValueError(f"head_type must be one of {HEAD_TYPES}")
    return value


def validate_length_head_mode(value: object) -> str:
    """Validate the mode without changing the legacy mean-head default."""
    if not isinstance(value, str) or value not in LENGTH_HEAD_MODES:
        raise ValueError(f"length_head_mode must be one of {LENGTH_HEAD_MODES}")
    return value
