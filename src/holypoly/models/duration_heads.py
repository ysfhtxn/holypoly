"""Optional raw-candidate-duration regressors; legacy PolyLengthHead is unchanged."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from holypoly.models.poly_heads import (
    PolyLengthHead,
    PolyLengthOutput,
    ReadStatus,
    _masked_mean,
)


class DurationLengthHead(PolyLengthHead):
    """Regress from duration alone or masked mean features plus duration.

    Duration is log1p(valid raw sample count), not padded input length or a
    verified Poly duration. No statistics are fitted using validation/test data.
    The inherited loss and length/status output contract remain unchanged.
    """

    def __init__(
        self,
        input_features: int,
        hidden_features: int = 128,
        *,
        mode: str,
        batch_first: bool = True,
        dropout: float = 0.0,
    ) -> None:
        if mode not in ("duration_only", "mean_duration"):
            raise ValueError("DurationLengthHead requires duration_only/mean_duration")
        super().__init__(
            input_features,
            hidden_features,
            batch_first=batch_first,
            dropout=dropout,
        )
        self.mode = mode
        if mode == "duration_only":
            # Remove unused trainable projection parameters from this baseline.
            self.projection = nn.Identity()
            regression_features = 1
        else:
            regression_features = hidden_features + 1
        self.length_regressor = nn.Sequential(
            nn.Linear(regression_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_features, 1),
        )
        self.status_classifier = nn.Linear(regression_features, len(ReadStatus))

    def forward(
        self,
        features: Tensor,
        padding_mask: Tensor | None = None,
        *,
        raw_lengths_samples: Tensor | None = None,
    ) -> PolyLengthOutput:
        """Use [B] integer raw sample counts and [B,T,C] encoder features."""
        batch_features, checked_mask = self._prepare_inputs(features, padding_mask)
        if raw_lengths_samples is None:
            raise ValueError("duration heads require valid raw_lengths_samples")
        if raw_lengths_samples.shape != (batch_features.shape[0],):
            raise ValueError("raw_lengths_samples must have shape [batch]")
        if raw_lengths_samples.device != features.device:
            raise ValueError("raw_lengths_samples and features must share a device")
        if raw_lengths_samples.dtype not in (torch.int32, torch.int64):
            raise ValueError("raw_lengths_samples must contain integer sample counts")
        if torch.any(raw_lengths_samples <= 0):
            raise ValueError("raw_lengths_samples must be positive")
        duration = torch.log1p(raw_lengths_samples.to(batch_features.dtype))[:, None]
        if self.mode == "duration_only":
            regression_input = duration
        else:
            pooled = _masked_mean(self.projection(batch_features), checked_mask)
            regression_input = torch.cat((pooled, duration), dim=-1)
        length_nt = functional.softplus(
            self.length_regressor(regression_input).squeeze(-1)
        )
        status_logits = self.status_classifier(regression_input)
        return PolyLengthOutput(
            length_nt=length_nt,
            status_logits=status_logits,
            predicted_length_nt=self.decode_lengths(length_nt, status_logits),
        )
