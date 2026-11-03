"""Torch-free validation and legacy defaults for auxiliary loss settings."""

from __future__ import annotations

import math
from collections.abc import Mapping

LENGTH_LOSS_MODES = ("log_smooth_l1", "mae", "huber")


def normalize_loss_config(values: Mapping[str, object]) -> dict[str, str | float]:
    """Validate loss settings, filling the original objective for old checkpoints.

    ``huber_delta_nt`` is the raw-nt transition between quadratic and linear
    Huber error. It is recorded for all modes and only used by ``huber``.
    """
    mode = values.get("length_loss", "log_smooth_l1")
    if not isinstance(mode, str) or mode not in LENGTH_LOSS_MODES:
        raise ValueError(f"length_loss must be one of {LENGTH_LOSS_MODES}")
    result: dict[str, str | float] = {"length_loss": mode}
    for name, default in (
        ("lambda_location", 1.0),
        ("lambda_length", 1.0),
        ("lambda_status", 1.0),
        ("huber_delta_nt", 10.0),
    ):
        value = values.get(name, default)
        if (
            isinstance(value, bool)
            or not isinstance(value, int | float)
            or not math.isfinite(value)
        ):
            raise ValueError(f"{name} must be a finite number")
        if name == "huber_delta_nt" and value <= 0:
            raise ValueError("huber_delta_nt must be greater than zero")
        if name != "huber_delta_nt" and value < 0:
            raise ValueError(f"{name} must be non-negative")
        result[name] = float(value)
    return result
