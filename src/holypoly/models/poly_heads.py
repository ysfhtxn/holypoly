"""Minimal multitask heads for features before Bonito's basecalling head."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

BONITO_V5_STRIDE = 6


def masked_status_loss(logits: Tensor, targets: Tensor) -> Tensor:
    """Ignore -100 status targets, including batches with no status labels."""
    valid = targets != -100
    if torch.any(valid):
        return functional.cross_entropy(logits[valid], targets[valid])
    return logits.sum() * 0.0


class PolyClass(IntEnum):
    """Per-timestep labels produced by :class:`PolyLocationHead`."""

    BACKGROUND = 0
    POLY_A = 1
    POLY_T = 2


class ReadStatus(IntEnum):
    """Read-level outcomes shared by both prototype heads."""

    POLY_PRESENT = 0
    ZERO_NT = 1
    FAILURE = 2


@dataclass(frozen=True)
class PolyLocationOutput:
    """Per-timestep location logits and read-level status logits."""

    location_logits: Tensor
    status_logits: Tensor
    padding_mask: Tensor


@dataclass(frozen=True)
class PolyLengthOutput:
    """Non-negative length estimate and explicit read-level status."""

    length_nt: Tensor
    status_logits: Tensor
    predicted_length_nt: Tensor


@dataclass(frozen=True)
class PolyInterval:
    """One decoded Poly interval in encoder and raw-signal coordinates."""

    poly_class: PolyClass
    encoder_start: int
    encoder_end: int
    raw_start_sample: int
    raw_end_sample: int


@dataclass(frozen=True)
class BonitoMultiTaskOutput:
    """Unchanged Bonito scores plus auxiliary head outputs."""

    basecall_scores: Tensor
    encoder_features: Tensor
    feature_padding_mask: Tensor
    location: PolyLocationOutput
    length: PolyLengthOutput


class _FeatureHead(nn.Module):
    """Shared validation and projection for explicit Bonito feature layouts."""

    def __init__(
        self,
        input_features: int,
        hidden_features: int,
        *,
        batch_first: bool,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_features < 1:
            raise ValueError("input_features must be at least 1")
        if hidden_features < 1:
            raise ValueError("hidden_features must be at least 1")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")
        self.input_features = input_features
        self.batch_first = batch_first
        self.projection = nn.Sequential(
            nn.LayerNorm(input_features),
            nn.Linear(input_features, hidden_features),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def _prepare_inputs(
        self, features: Tensor, padding_mask: Tensor | None
    ) -> tuple[Tensor, Tensor]:
        if features.ndim != 3:
            raise ValueError(
                "features must have three dimensions: [batch, time, features] "
                "or [time, batch, features]"
            )
        if features.shape[-1] != self.input_features:
            raise ValueError(
                f"expected {self.input_features} input features, "
                f"received {features.shape[-1]}"
            )
        batch_features = features if self.batch_first else features.transpose(0, 1)
        batch_size, time_steps, _ = batch_features.shape
        if batch_size == 0:
            raise ValueError("features must contain at least one read")
        if time_steps == 0:
            raise ValueError("features must contain at least one timestep")
        if padding_mask is None:
            padding_mask = torch.zeros(
                (batch_size, time_steps),
                dtype=torch.bool,
                device=features.device,
            )
        if padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must have boolean dtype")
        if padding_mask.shape != (batch_size, time_steps):
            raise ValueError(
                "padding_mask must have shape [batch, time], expected "
                f"{(batch_size, time_steps)}, received {tuple(padding_mask.shape)}"
            )
        if padding_mask.device != features.device:
            raise ValueError("padding_mask and features must be on the same device")
        if torch.any(padding_mask.all(dim=1)):
            raise ValueError("each read must contain at least one non-padding timestep")
        return batch_features, padding_mask


def _masked_mean(features: Tensor, padding_mask: Tensor) -> Tensor:
    valid_mask = ~padding_mask
    masked_features = features.masked_fill(padding_mask.unsqueeze(-1), 0.0)
    valid_counts = valid_mask.sum(dim=1, keepdim=True).to(features.dtype)
    return masked_features.sum(dim=1) / valid_counts


class PolyLocationHead(_FeatureHead):
    """Predict poly(A/T) locations and present/zero/failure read status.

    The input must be the feature tensor immediately before Bonito's existing
    CRF basecalling head. ``batch_first`` is explicit because current Bonito
    transformer models use ``[batch, time, features]`` there, while older RNN
    models use ``[time, batch, features]``.
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
            input_features,
            hidden_features,
            batch_first=batch_first,
            dropout=dropout,
        )
        self.location_classifier = nn.Linear(hidden_features, len(PolyClass))
        self.status_classifier = nn.Linear(hidden_features, len(ReadStatus))

    def forward(
        self, features: Tensor, padding_mask: Tensor | None = None
    ) -> PolyLocationOutput:
        """Return batch-first logits while zeroing padded timestep logits."""
        batch_features, checked_mask = self._prepare_inputs(features, padding_mask)
        hidden = self.projection(batch_features)
        location_logits = self.location_classifier(hidden)
        location_logits = location_logits.masked_fill(checked_mask.unsqueeze(-1), 0.0)
        status_logits = self.status_classifier(_masked_mean(hidden, checked_mask))
        return PolyLocationOutput(
            location_logits=location_logits,
            status_logits=status_logits,
            padding_mask=checked_mask,
        )

    def loss(
        self,
        output: PolyLocationOutput,
        location_targets: Tensor,
        status_targets: Tensor,
        *,
        ignore_index: int = -100,
        status_weight: float = 1.0,
    ) -> dict[str, Tensor]:
        """Compute masked timestep cross-entropy and read-status loss."""
        expected_location_shape = output.location_logits.shape[:2]
        if location_targets.shape != expected_location_shape:
            raise ValueError(
                "location_targets must match [batch, time], expected "
                f"{expected_location_shape}, received {tuple(location_targets.shape)}"
            )
        expected_status_shape = (output.status_logits.shape[0],)
        if status_targets.shape != expected_status_shape:
            raise ValueError(
                "status_targets must match [batch], expected "
                f"{expected_status_shape}, received {tuple(status_targets.shape)}"
            )
        valid_locations = (~output.padding_mask) & (location_targets != ignore_index)
        if torch.any(valid_locations):
            location_loss = functional.cross_entropy(
                output.location_logits[valid_locations],
                location_targets[valid_locations],
            )
        else:
            location_loss = output.location_logits.sum() * 0.0
        status_loss = masked_status_loss(output.status_logits, status_targets)
        total_loss = location_loss + status_weight * status_loss
        return {
            "total_loss": total_loss,
            "location_loss": location_loss,
            "status_loss": status_loss,
        }


class PolyLengthHead(_FeatureHead):
    """Predict non-negative poly length plus present/zero/failure status."""

    def __init__(
        self,
        input_features: int,
        hidden_features: int = 128,
        *,
        batch_first: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__(
            input_features,
            hidden_features,
            batch_first=batch_first,
            dropout=dropout,
        )
        self.length_regressor = nn.Linear(hidden_features, 1)
        self.status_classifier = nn.Linear(hidden_features, len(ReadStatus))

    def forward(
        self, features: Tensor, padding_mask: Tensor | None = None
    ) -> PolyLengthOutput:
        """Return a non-negative regression value and decoded status-aware length."""
        batch_features, checked_mask = self._prepare_inputs(features, padding_mask)
        hidden = self.projection(batch_features)
        pooled = _masked_mean(hidden, checked_mask)
        length_nt = functional.softplus(self.length_regressor(pooled).squeeze(-1))
        status_logits = self.status_classifier(pooled)
        predicted_length_nt = self.decode_lengths(length_nt, status_logits)
        return PolyLengthOutput(
            length_nt=length_nt,
            status_logits=status_logits,
            predicted_length_nt=predicted_length_nt,
        )

    @staticmethod
    def decode_lengths(length_nt: Tensor, status_logits: Tensor) -> Tensor:
        """Map zero-nt reads to 0 and failed reads to NaN."""
        if length_nt.ndim != 1:
            raise ValueError("length_nt must have shape [batch]")
        if status_logits.shape != (length_nt.shape[0], len(ReadStatus)):
            raise ValueError(
                "status_logits must have shape [batch, 3], received "
                f"{tuple(status_logits.shape)}"
            )
        status = status_logits.argmax(dim=-1)
        decoded = length_nt.clone()
        decoded = torch.where(
            status == int(ReadStatus.ZERO_NT), torch.zeros_like(decoded), decoded
        )
        return torch.where(
            status == int(ReadStatus.FAILURE),
            torch.full_like(decoded, torch.nan),
            decoded,
        )

    def loss(
        self,
        output: PolyLengthOutput,
        true_length_nt: Tensor,
        status_targets: Tensor,
        *,
        status_weight: float = 1.0,
    ) -> dict[str, Tensor]:
        """Compute Smooth L1 for non-failed reads and read-status loss."""
        expected_shape = (output.length_nt.shape[0],)
        if true_length_nt.shape != expected_shape:
            raise ValueError(
                f"true_length_nt must have shape {expected_shape}, received "
                f"{tuple(true_length_nt.shape)}"
            )
        if status_targets.shape != expected_shape:
            raise ValueError(
                f"status_targets must have shape {expected_shape}, received "
                f"{tuple(status_targets.shape)}"
            )
        regression_mask = status_targets != int(ReadStatus.FAILURE)
        regression_targets = true_length_nt[regression_mask]
        if torch.any(~torch.isfinite(regression_targets)):
            raise ValueError("non-failed true lengths must be finite")
        if torch.any(regression_targets < 0):
            raise ValueError("non-failed true lengths must be non-negative")
        if torch.any(regression_mask):
            length_loss = functional.smooth_l1_loss(
                torch.log1p(output.length_nt[regression_mask]),
                torch.log1p(regression_targets),
            )
        else:
            length_loss = output.length_nt.sum() * 0.0
        status_loss = masked_status_loss(output.status_logits, status_targets)
        total_loss = length_loss + status_weight * status_loss
        return {
            "total_loss": total_loss,
            "length_loss": length_loss,
            "status_loss": status_loss,
        }


def raw_interval_to_encoder_interval(
    start_sample: int,
    end_sample: int,
    *,
    raw_length_samples: int,
    feature_steps: int,
    stride: int = BONITO_V5_STRIDE,
) -> tuple[int, int]:
    """Map a 0-based half-open raw-signal interval to encoder timesteps."""
    _validate_raw_interval(start_sample, end_sample, raw_length_samples)
    if feature_steps < 1:
        raise ValueError("feature_steps must be at least 1")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    encoder_start = min(start_sample // stride, feature_steps)
    encoder_end = min((end_sample + stride - 1) // stride, feature_steps)
    return encoder_start, encoder_end


def encoder_interval_to_raw_interval(
    start_timestep: int,
    end_timestep: int,
    *,
    raw_length_samples: int,
    feature_steps: int,
    stride: int = BONITO_V5_STRIDE,
) -> tuple[int, int]:
    """Map a 0-based half-open encoder interval back to raw samples."""
    if not 0 <= start_timestep <= end_timestep <= feature_steps:
        raise ValueError(
            "encoder interval must satisfy "
            "0 <= start_timestep <= end_timestep <= feature_steps"
        )
    if raw_length_samples < 0:
        raise ValueError("raw_length_samples must be non-negative")
    if stride < 1:
        raise ValueError("stride must be at least 1")
    return (
        min(start_timestep * stride, raw_length_samples),
        min(end_timestep * stride, raw_length_samples),
    )


def make_location_targets(
    start_sample: int,
    end_sample: int,
    *,
    raw_length_samples: int,
    feature_steps: int,
    poly_class: PolyClass,
    stride: int = BONITO_V5_STRIDE,
    device: torch.device | None = None,
) -> Tensor:
    """Create encoder-timestep labels from one canonical raw-signal interval.

    Non-target interruptions remain ``BACKGROUND`` and must be supplied as
    separate label edits by the label-generation pipeline. This function does
    not merge interruptions.
    """
    if poly_class is PolyClass.BACKGROUND:
        raise ValueError("poly_class must be POLY_A or POLY_T")
    encoder_start, encoder_end = raw_interval_to_encoder_interval(
        start_sample,
        end_sample,
        raw_length_samples=raw_length_samples,
        feature_steps=feature_steps,
        stride=stride,
    )
    targets = torch.full(
        (feature_steps,),
        int(PolyClass.BACKGROUND),
        dtype=torch.long,
        device=device,
    )
    targets[encoder_start:encoder_end] = int(poly_class)
    return targets


def decode_location_intervals(
    location_output: PolyLocationOutput,
    status_logits: Tensor,
    raw_lengths_samples: Tensor,
    *,
    stride: int = BONITO_V5_STRIDE,
) -> tuple[PolyInterval | None, ...]:
    """Decode the longest contiguous canonical Poly run for each read."""
    batch_size, feature_steps, classes = location_output.location_logits.shape
    if classes != len(PolyClass):
        raise ValueError("location logits must contain three Poly classes")
    if status_logits.shape != (batch_size, len(ReadStatus)):
        raise ValueError("status_logits must have shape [batch, 3]")
    if raw_lengths_samples.shape != (batch_size,):
        raise ValueError("raw_lengths_samples must have shape [batch]")
    predictions = location_output.location_logits.argmax(dim=-1)
    statuses = status_logits.argmax(dim=-1)
    intervals: list[PolyInterval | None] = []
    for batch_index in range(batch_size):
        if statuses[batch_index].item() != int(ReadStatus.POLY_PRESENT):
            intervals.append(None)
            continue
        valid_steps = (~location_output.padding_mask[batch_index]).sum().item()
        best = _longest_poly_run(predictions[batch_index, :valid_steps])
        if best is None:
            intervals.append(None)
            continue
        encoder_start, encoder_end, poly_class = best
        raw_length_samples = int(raw_lengths_samples[batch_index].item())
        raw_start, raw_end = encoder_interval_to_raw_interval(
            encoder_start,
            encoder_end,
            raw_length_samples=raw_length_samples,
            feature_steps=feature_steps,
            stride=stride,
        )
        intervals.append(
            PolyInterval(
                poly_class=poly_class,
                encoder_start=encoder_start,
                encoder_end=encoder_end,
                raw_start_sample=raw_start,
                raw_end_sample=raw_end,
            )
        )
    return tuple(intervals)


class BonitoV5MultiTaskModel(nn.Module):
    """Attach auxiliary heads to Bonito v5 immediately before its CRF head."""

    def __init__(
        self,
        basecaller: nn.Module,
        *,
        hidden_features: int = 128,
        dropout: float = 0.0,
        lambda_location: float = 1.0,
        lambda_length: float = 1.0,
        lambda_status: float = 1.0,
    ) -> None:
        super().__init__()
        self.basecaller = basecaller
        encoder = _require_v5_encoder(basecaller)
        input_features = int(encoder.crf.linear.in_features)
        self.location_head = PolyLocationHead(
            input_features,
            hidden_features,
            batch_first=True,
            dropout=dropout,
        )
        self.length_head = PolyLengthHead(
            input_features,
            hidden_features,
            batch_first=True,
            dropout=dropout,
        )
        self.lambda_location = _validate_loss_weight(lambda_location, "lambda_location")
        self.lambda_length = _validate_loss_weight(lambda_length, "lambda_length")
        self.lambda_status = _validate_loss_weight(lambda_status, "lambda_status")

    def forward(
        self, signal: Tensor, raw_padding_mask: Tensor | None = None
    ) -> BonitoMultiTaskOutput:
        """Return original CRF scores plus auxiliary predictions."""
        checked_signal, raw_lengths = _prepare_signal(signal, raw_padding_mask)
        encoder = _require_v5_encoder(self.basecaller)
        features = encoder.conv(checked_signal)
        features = encoder.transformer_encoder(features)
        features = encoder.upsample(features)
        feature_mask = _feature_padding_mask(
            raw_lengths,
            feature_steps=features.shape[1],
            stride=BONITO_V5_STRIDE,
        )
        basecall_scores = encoder.crf(features)
        location = self.location_head(features, feature_mask)
        length = self.length_head(features, feature_mask)
        return BonitoMultiTaskOutput(
            basecall_scores=basecall_scores,
            encoder_features=features,
            feature_padding_mask=feature_mask,
            location=location,
            length=length,
        )

    def auxiliary_loss(
        self,
        output: BonitoMultiTaskOutput,
        location_targets: Tensor,
        true_length_nt: Tensor,
        status_targets: Tensor,
    ) -> dict[str, Tensor]:
        """Compute auxiliary loss with one shared read-status term."""
        location_losses = self.location_head.loss(
            output.location,
            location_targets,
            status_targets,
            status_weight=0.0,
        )
        length_losses = self.length_head.loss(
            output.length,
            true_length_nt,
            status_targets,
            status_weight=0.0,
        )
        shared_status_logits = (
            output.location.status_logits + output.length.status_logits
        ) / 2
        status_loss = masked_status_loss(shared_status_logits, status_targets)
        total_loss = (
            self.lambda_location * location_losses["location_loss"]
            + self.lambda_length * length_losses["length_loss"]
            + self.lambda_status * status_loss
        )
        return {
            "total_loss": total_loss,
            "location_loss": location_losses["location_loss"],
            "length_loss": length_losses["length_loss"],
            "status_loss": status_loss,
        }

    def basecall_loss(self, *args: Any, **kwargs: Any) -> Any:
        """Delegate unchanged basecalling loss computation to Bonito."""
        loss_method = getattr(self.basecaller, "loss", None)
        if loss_method is None:
            raise TypeError("basecaller must provide a loss() method")
        return loss_method(*args, **kwargs)


def _validate_raw_interval(
    start_sample: int, end_sample: int, raw_length_samples: int
) -> None:
    if raw_length_samples < 0:
        raise ValueError("raw_length_samples must be non-negative")
    if not 0 <= start_sample <= end_sample <= raw_length_samples:
        raise ValueError(
            "raw interval must satisfy "
            "0 <= start_sample <= end_sample <= raw_length_samples"
        )


def _longest_poly_run(labels: Tensor) -> tuple[int, int, PolyClass] | None:
    best: tuple[int, int, PolyClass] | None = None
    start = 0
    while start < labels.shape[0]:
        label = int(labels[start].item())
        if label == int(PolyClass.BACKGROUND):
            start += 1
            continue
        end = start + 1
        while end < labels.shape[0] and int(labels[end].item()) == label:
            end += 1
        candidate = (start, end, PolyClass(label))
        if best is None or end - start > best[1] - best[0]:
            best = candidate
        start = end
    return best


def _require_v5_encoder(basecaller: nn.Module) -> Any:
    encoder = getattr(basecaller, "encoder", None)
    required_layers = ("conv", "transformer_encoder", "upsample", "crf")
    if encoder is None or any(not hasattr(encoder, name) for name in required_layers):
        raise TypeError(
            "basecaller must expose Bonito v5 encoder layers: "
            "conv, transformer_encoder, upsample, and crf"
        )
    crf = encoder.crf
    if not hasattr(crf, "linear") or not hasattr(crf.linear, "in_features"):
        raise TypeError("Bonito v5 CRF head must expose linear.in_features")
    return encoder


def _prepare_signal(
    signal: Tensor, raw_padding_mask: Tensor | None
) -> tuple[Tensor, Tensor]:
    if signal.ndim != 3:
        raise ValueError("signal must have shape [batch, channels, raw_samples]")
    batch_size, _, raw_samples = signal.shape
    if batch_size == 0:
        raise ValueError("signal must contain at least one read")
    if raw_samples == 0:
        raise ValueError("signal must contain at least one raw sample")
    if raw_padding_mask is None:
        raw_lengths = torch.full(
            (batch_size,), raw_samples, dtype=torch.long, device=signal.device
        )
        return signal, raw_lengths
    if raw_padding_mask.dtype != torch.bool:
        raise ValueError("raw_padding_mask must have boolean dtype")
    if raw_padding_mask.shape != (batch_size, raw_samples):
        raise ValueError("raw_padding_mask must have shape [batch, raw_samples]")
    if raw_padding_mask.device != signal.device:
        raise ValueError("raw_padding_mask and signal must be on the same device")
    if torch.any(raw_padding_mask.all(dim=1)):
        raise ValueError("each read must contain at least one non-padding raw sample")
    if torch.any(raw_padding_mask[:, :-1] & ~raw_padding_mask[:, 1:]):
        raise ValueError("raw_padding_mask must contain suffix padding only")
    raw_lengths = (~raw_padding_mask).sum(dim=1)
    checked_signal = signal.masked_fill(raw_padding_mask.unsqueeze(1), 0.0)
    return checked_signal, raw_lengths


def _feature_padding_mask(
    raw_lengths: Tensor, *, feature_steps: int, stride: int
) -> Tensor:
    feature_lengths = torch.div(
        raw_lengths + stride - 1, stride, rounding_mode="floor"
    ).clamp(max=feature_steps)
    positions = torch.arange(feature_steps, device=raw_lengths.device)
    return positions.unsqueeze(0) >= feature_lengths.unsqueeze(1)


def _validate_loss_weight(value: float, name: str) -> float:
    converted = float(value)
    if converted < 0:
        raise ValueError(f"{name} must be non-negative")
    return converted
