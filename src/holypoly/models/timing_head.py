"""Opt-in timing fusion with an exact original sum-head path for missing evidence."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from holypoly.models.poly_heads import PolyLengthOutput
from holypoly.models.pooling_heads import SumLengthHead


class TimingReferenceLengthHead(SumLengthHead):
    """Fuse pooled features, timing rate/dispersion and provisional duration ratio.

    The residual acts on the pre-softplus length logit only. Status remains the
    original classifier. Missing references bypass the residual exactly; a zero
    initialized final layer preserves the original prediction at initialization.
    """

    def __init__(
        self,
        input_features: int,
        hidden_features: int = 128,
        *,
        batch_first: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            input_features, hidden_features, batch_first=batch_first, dropout=dropout
        )
        self.timing_residual = nn.Sequential(
            nn.Linear(hidden_features + 4, hidden_features),
            nn.GELU(),
            nn.Linear(hidden_features, 1),
        )
        nn.init.zeros_(self.timing_residual[-1].weight)
        nn.init.zeros_(self.timing_residual[-1].bias)

    def forward(
        self,
        features: Tensor,
        padding_mask: Tensor | None = None,
        *,
        raw_lengths_samples: Tensor | None = None,
        timing_reference: Tensor | None = None,
    ) -> PolyLengthOutput:
        """Timing is float [B,3]: log1p(samples/base), log-rate SD, availability."""
        if timing_reference is None:
            return super().forward(features, padding_mask)
        batch, checked_mask = self._prepare_inputs(features, padding_mask)
        if (
            timing_reference.shape != (batch.shape[0], 3)
            or timing_reference.device != batch.device
            or not timing_reference.is_floating_point()
        ):
            raise ValueError(
                "timing_reference must be float [batch,3] on features device"
            )
        if not torch.isfinite(timing_reference).all():
            raise ValueError("timing reference must be finite; missing rows use zeros")
        available = timing_reference[:, 2]
        if (
            torch.any((available != 0) & (available != 1))
            or torch.any(timing_reference[:, :2] < 0)
            or torch.any((available == 1) & (timing_reference[:, 0] <= 0))
        ):
            raise ValueError("invalid timing rate, dispersion or availability")
        if (
            raw_lengths_samples is None
            or raw_lengths_samples.shape != (batch.shape[0],)
            or raw_lengths_samples.device != batch.device
            or not torch.isfinite(raw_lengths_samples).all()
            or torch.any(raw_lengths_samples <= 0)
        ):
            raise ValueError("timing fusion needs positive raw_lengths_samples [batch]")
        hidden = self.projection(batch)
        pooled = hidden.masked_fill(checked_mask.unsqueeze(-1), 0).sum(1)
        metadata = timing_reference.to(pooled.dtype).clone()
        metadata[available == 0] = 0
        rate = metadata[:, 0].expm1().clamp_min(torch.finfo(pooled.dtype).eps)
        if not torch.isfinite(rate).all():
            raise ValueError("timing rate exceeds model numeric range")
        ratio = torch.log1p(raw_lengths_samples.to(pooled.dtype) / rate) * available
        extra = torch.cat((metadata, ratio[:, None]), -1)
        residual = self.timing_residual(torch.cat((pooled, extra), -1)).squeeze(-1)
        original = self.length_regressor(pooled).squeeze(-1)
        value = F.softplus(original + residual * available)
        status = self.status_classifier(pooled)
        return PolyLengthOutput(value, status, self.decode_lengths(value, status))
