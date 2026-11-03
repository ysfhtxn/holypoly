"""Torch-free configuration for opt-in length-head comparisons."""

LENGTH_HEAD_MODES = ("mean", "sum", "duration_only", "mean_duration")


def validate_length_head_mode(value: object) -> str:
    """Validate the mode without changing the legacy mean-head default."""
    if not isinstance(value, str) or value not in LENGTH_HEAD_MODES:
        raise ValueError(f"length_head_mode must be one of {LENGTH_HEAD_MODES}")
    return value
