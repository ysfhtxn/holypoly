"""Independent auxiliary loss for frozen-encoder Poly training."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from holypoly.models.poly_heads import (
    BonitoMultiTaskOutput,
    ReadStatus,
    masked_status_loss,
)


@dataclass(frozen=True)
class PolyLossWeights:
    """Configurable weights for each recorded auxiliary loss."""

    lambda_location: float = 1.0
    lambda_length: float = 1.0
    lambda_status: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (
            ("lambda_location", self.lambda_location),
            ("lambda_length", self.lambda_length),
            ("lambda_status", self.lambda_status),
        ):
            if value < 0:
                raise ValueError(f"{name} must be non-negative")


class PolyAuxiliaryLoss(nn.Module):
    """Compute masked location, log-length, and shared status losses."""

    def __init__(
        self,
        weights: PolyLossWeights,
        *,
        location_ignore_index: int = -100,
    ) -> None:
        super().__init__()
        self.weights = weights
        self.location_ignore_index = location_ignore_index

    def forward(
        self,
        output: BonitoMultiTaskOutput,
        location_targets: Tensor,
        training_length_nt: Tensor,
        status_targets: Tensor,
    ) -> dict[str, Tensor]:
        """Return total loss and every unweighted and weighted component."""
        expected_locations = output.location.location_logits.shape[:2]
        if location_targets.shape != expected_locations:
            raise ValueError(
                "location_targets must match [batch, time], expected "
                f"{expected_locations}, received {tuple(location_targets.shape)}"
            )
        expected_reads = (output.location.location_logits.shape[0],)
        if training_length_nt.shape != expected_reads:
            raise ValueError(
                f"training_length_nt must have shape {expected_reads}, received "
                f"{tuple(training_length_nt.shape)}"
            )
        if status_targets.shape != expected_reads:
            raise ValueError(
                f"status_targets must have shape {expected_reads}, received "
                f"{tuple(status_targets.shape)}"
            )

        valid_locations = (~output.feature_padding_mask) & (
            location_targets != self.location_ignore_index
        )
        if torch.any(valid_locations):
            location_loss = functional.cross_entropy(
                output.location.location_logits[valid_locations],
                location_targets[valid_locations],
            )
        else:
            location_loss = output.location.location_logits.sum() * 0.0

        length_mask = status_targets != int(ReadStatus.FAILURE)
        valid_lengths = training_length_nt[length_mask]
        if torch.any(~torch.isfinite(valid_lengths)):
            raise ValueError("non-failed training lengths must be finite")
        if torch.any(valid_lengths < 0):
            raise ValueError("non-failed training lengths must be non-negative")
        if torch.any(length_mask):
            length_loss = functional.smooth_l1_loss(
                torch.log1p(output.length.length_nt[length_mask]),
                torch.log1p(valid_lengths),
            )
        else:
            length_loss = output.length.length_nt.sum() * 0.0

        shared_status_logits = (
            output.location.status_logits + output.length.status_logits
        ) / 2
        valid_status = status_targets != -100
        status_loss = masked_status_loss(shared_status_logits, status_targets)
        weighted_location_loss = self.weights.lambda_location * location_loss
        weighted_length_loss = self.weights.lambda_length * length_loss
        weighted_status_loss = self.weights.lambda_status * status_loss
        total_loss = (
            weighted_location_loss + weighted_length_loss + weighted_status_loss
        )
        return {
            "total_loss": total_loss,
            "location_loss": location_loss,
            "length_loss": length_loss,
            "status_loss": status_loss,
            "weighted_location_loss": weighted_location_loss,
            "weighted_length_loss": weighted_length_loss,
            "weighted_status_loss": weighted_status_loss,
            "location_examples": valid_locations.sum(),
            "length_examples": length_mask.sum(),
            "status_examples": valid_status.sum(),
        }
