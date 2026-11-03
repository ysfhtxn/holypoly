"""Opt-in masked sum pooling with the original mean head's trainable layers."""

from __future__ import annotations

import torch.nn.functional as functional
from torch import Tensor

from holypoly.models.poly_heads import PolyLengthHead, PolyLengthOutput


class SumLengthHead(PolyLengthHead):
    """Sum projected valid timesteps without changing layers or initialization.

    Padding is excluded. The sum grows with valid candidate encoder timesteps,
    which are not an independently verified Poly interval or duration.
    """

    def forward(
        self, features: Tensor, padding_mask: Tensor | None = None
    ) -> PolyLengthOutput:
        """Pool [B,T,H] projected features to [B,H] before length/status layers."""
        batch_features, checked_mask = self._prepare_inputs(features, padding_mask)
        hidden = self.projection(batch_features)
        pooled = hidden.masked_fill(checked_mask.unsqueeze(-1), 0.0).sum(dim=1)
        length_nt = functional.softplus(self.length_regressor(pooled).squeeze(-1))
        status_logits = self.status_classifier(pooled)
        return PolyLengthOutput(
            length_nt=length_nt,
            status_logits=status_logits,
            predicted_length_nt=self.decode_lengths(length_nt, status_logits),
        )
