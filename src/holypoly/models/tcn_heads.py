"""Independent residual TCN branches for length and global read status only."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from holypoly.models.length_head_config import (
    DEFAULT_TCN_LENGTH_FEATURES,
    DEFAULT_TCN_LENGTH_SUM_SCALE,
    validate_tcn_length_features,
    validate_tcn_length_sum_scale,
)
from holypoly.models.poly_heads import (
    PolyClass,
    PolyLengthHead,
    PolyLengthOutput,
    PolyLocationHead,
    PolyLocationOutput,
    ReadStatus,
)

CHANNELS = 128
DILATIONS = (1, 2, 4, 8)
TCN_DROPOUT = 0.1


class ResidualTCNBlock(nn.Module):
    """Preserve [B,T,128]; True padding is zeroed before and after convolutions."""

    def __init__(self, dilation: int) -> None:
        super().__init__()
        if dilation not in DILATIONS:
            raise ValueError(f"TCN dilation must be one of {DILATIONS}")
        self.norm1 = nn.LayerNorm(CHANNELS)
        self.conv1 = nn.Conv1d(
            CHANNELS, CHANNELS, kernel_size=3, dilation=dilation, padding=dilation
        )
        self.norm2 = nn.LayerNorm(CHANNELS)
        self.conv2 = nn.Conv1d(
            CHANNELS, CHANNELS, kernel_size=3, dilation=dilation, padding=dilation
        )
        self.dropout1 = nn.Dropout(TCN_DROPOUT)
        self.dropout2 = nn.Dropout(TCN_DROPOUT)

    def forward(self, hidden: Tensor, padding_mask: Tensor) -> Tensor:
        """Apply channel LayerNorm and symmetric convolutions in encoder frame order."""
        mask = padding_mask.unsqueeze(-1)
        residual = hidden.masked_fill(mask, 0.0)
        branch = self.norm1(residual).masked_fill(mask, 0.0)
        branch = self.conv1(branch.transpose(1, 2)).transpose(1, 2)
        branch = branch.masked_fill(mask, 0.0)
        branch = self.dropout1(functional.gelu(branch))
        branch = self.norm2(branch).masked_fill(mask, 0.0)
        branch = self.conv2(branch.transpose(1, 2)).transpose(1, 2)
        branch = self.dropout2(branch.masked_fill(mask, 0.0))
        return (residual + branch).masked_fill(mask, 0.0)


def _projection(input_features: int) -> nn.Sequential:
    """Project each encoder frame independently to the fixed TCN channel count."""
    return nn.Sequential(
        nn.LayerNorm(input_features), nn.Linear(input_features, CHANNELS), nn.GELU()
    )


def _classifier(outputs: int, input_features: int = CHANNELS) -> nn.Sequential:
    """Create one independent input_features -> 64 -> outputs read-level MLP."""
    return nn.Sequential(
        nn.Linear(input_features, 64), nn.GELU(), nn.Linear(64, outputs)
    )


def _encode(
    projection: nn.Module,
    blocks: nn.ModuleList,
    features: Tensor,
    padding_mask: Tensor,
) -> Tensor:
    """Isolate padded frames, including nonfinite sentinels, throughout the TCN."""
    mask = padding_mask.unsqueeze(-1)
    hidden = projection(features.masked_fill(mask, 0.0)).masked_fill(mask, 0.0)
    for block in blocks:
        hidden = block(hidden, padding_mask)
    return hidden


class TCNLengthHead(PolyLengthHead):
    """Masked-sum TCN with independent length and three-state MLPs."""

    def __init__(
        self,
        input_features: int,
        *,
        tcn_length_sum_scale: float = DEFAULT_TCN_LENGTH_SUM_SCALE,
        tcn_length_features: str = DEFAULT_TCN_LENGTH_FEATURES,
    ) -> None:
        self.tcn_length_sum_scale = validate_tcn_length_sum_scale(tcn_length_sum_scale)
        self.tcn_length_features = validate_tcn_length_features(tcn_length_features)
        super().__init__(input_features, CHANNELS)
        self.projection = _projection(input_features)
        self.blocks = nn.ModuleList(ResidualTCNBlock(d) for d in DILATIONS)
        regression_features = (
            2 * CHANNELS + 1
            if self.tcn_length_features == "sum_mean_duration"
            else CHANNELS
        )
        self.length_regressor = _classifier(1, regression_features)
        self.status_classifier = _classifier(len(ReadStatus))

    def forward(
        self, features: Tensor, padding_mask: Tensor | None = None
    ) -> PolyLengthOutput:
        """Return [B] length and raw [B,3] logits with legacy decoding fields."""
        batch, checked_mask = self._prepare_inputs(features, padding_mask)
        hidden = _encode(self.projection, self.blocks, batch, checked_mask)
        pooled = hidden.sum(dim=1)
        # [B,128] sum stays cumulative. Optional [B,128] mean and [B,1]
        # log1p(valid encoder frames) describe the candidate, not a true Poly span.
        length_features = pooled * self.tcn_length_sum_scale
        if self.tcn_length_features == "sum_mean_duration":
            counts = (~checked_mask).sum(dim=1, keepdim=True).to(hidden.dtype)
            length_features = torch.cat(
                (length_features, pooled / counts, torch.log1p(counts)), dim=-1
            )
        length_nt = functional.softplus(
            self.length_regressor(length_features).squeeze(-1)
        )
        status = self.status_classifier(pooled)
        return PolyLengthOutput(
            length_nt, status, self.decode_lengths(length_nt, status)
        )


class TCNLocationHead(PolyLocationHead):
    """Masked-mean TCN for global status; no per-frame location classifier."""

    def __init__(self, input_features: int) -> None:
        super().__init__(input_features, CHANNELS)
        del self.location_classifier
        self.projection = _projection(input_features)
        self.blocks = nn.ModuleList(ResidualTCNBlock(d) for d in DILATIONS)
        self.status_classifier = _classifier(len(ReadStatus))

    def forward(
        self, features: Tensor, padding_mask: Tensor | None = None
    ) -> PolyLocationOutput:
        """Keep the output schema with zero location placeholders, never coordinates."""
        batch, checked_mask = self._prepare_inputs(features, padding_mask)
        hidden = _encode(self.projection, self.blocks, batch, checked_mask)
        counts = (~checked_mask).sum(dim=1, keepdim=True).to(hidden.dtype)
        pooled = hidden.sum(dim=1) / counts
        locations = hidden.new_zeros((*hidden.shape[:2], len(PolyClass)))
        return PolyLocationOutput(
            locations, self.status_classifier(pooled), checked_mask
        )
