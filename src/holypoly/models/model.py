"""Frozen Bonito v5 wrapper for Poly auxiliary-head training."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Self, cast

from torch import Tensor, nn

from holypoly.models.length_head_config import (
    DEFAULT_TCN_LENGTH_FEATURES,
    DEFAULT_TCN_LENGTH_SUM_SCALE,
    validate_head_type,
    validate_length_head_mode,
    validate_tcn_length_features,
    validate_tcn_length_sum_scale,
)
from holypoly.models.poly_heads import (
    BonitoMultiTaskOutput,
    BonitoV5MultiTaskModel,
    PolyAuxiliaryOutput,
)


class HolypolyModel(nn.Module):
    """Keep Bonito basecalling intact while training only the two Poly heads."""

    def __init__(
        self,
        basecaller: nn.Module,
        *,
        hidden_features: int = 128,
        dropout: float = 0.0,
        freeze_encoder: bool = True,
        freeze_basecalling_head: bool = True,
        length_head_mode: str = "mean",
        use_timing_reference: bool = False,
        head_type: str = "legacy",
        tcn_length_sum_scale: float = DEFAULT_TCN_LENGTH_SUM_SCALE,
        tcn_length_features: str = DEFAULT_TCN_LENGTH_FEATURES,
    ) -> None:
        super().__init__()
        self.head_config = {
            "length_head_mode": validate_length_head_mode(length_head_mode),
            "hidden_features": hidden_features,
            "dropout": dropout,
        }
        if validate_head_type(head_type) != "legacy":
            self.head_config["head_type"] = head_type
            self.head_config["tcn_length_sum_scale"] = validate_tcn_length_sum_scale(
                tcn_length_sum_scale
            )
            self.head_config["tcn_length_features"] = validate_tcn_length_features(
                tcn_length_features
            )
        if use_timing_reference:
            self.head_config["use_timing_reference"] = True
        self.multitask = BonitoV5MultiTaskModel(
            basecaller,
            hidden_features=hidden_features,
            dropout=dropout,
            length_head_mode=length_head_mode,
            use_timing_reference=use_timing_reference,
            head_type=head_type,
            tcn_length_sum_scale=tcn_length_sum_scale,
            tcn_length_features=tcn_length_features,
        )
        self.freeze_encoder = freeze_encoder
        self.freeze_basecalling_head = freeze_basecalling_head
        self._apply_freezing()

    @property
    def basecaller(self) -> nn.Module:
        """Return the unchanged Bonito basecaller."""
        return self.multitask.basecaller

    @property
    def location_head(self) -> nn.Module:
        """Return the trainable Poly location head."""
        return self.multitask.location_head

    @property
    def length_head(self) -> nn.Module:
        """Return the trainable Poly length head."""
        return self.multitask.length_head

    def forward(
        self,
        signal: Tensor,
        raw_padding_mask: Tensor | None = None,
        *,
        timing_reference: Tensor | None = None,
    ) -> BonitoMultiTaskOutput:
        """Return unchanged basecall scores plus location and length outputs."""
        if timing_reference is not None:
            return cast(
                BonitoMultiTaskOutput,
                self.multitask(
                    signal, raw_padding_mask, timing_reference=timing_reference
                ),
            )
        return cast(BonitoMultiTaskOutput, self.multitask(signal, raw_padding_mask))

    def forward_poly(
        self,
        signal: Tensor,
        raw_padding_mask: Tensor | None = None,
        *,
        timing_reference: Tensor | None = None,
        measure: Any = None,
    ) -> PolyAuxiliaryOutput:
        """Return only auxiliary predictions; keep the full forward unchanged."""
        return self.multitask.forward_poly(
            signal, raw_padding_mask, timing_reference=timing_reference, measure=measure
        )

    def train(self, mode: bool = True) -> Self:
        """Train Poly heads while keeping all frozen Bonito modules in eval mode."""
        super().train(mode)
        if self.freeze_encoder or self.freeze_basecalling_head:
            self.basecaller.eval()
        return self

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield only parameters owned by the two Poly heads."""
        yield from self.location_head.parameters()
        yield from self.length_head.parameters()

    def parameter_counts(self) -> dict[str, int]:
        """Return frozen and trainable parameter counts."""
        frozen = sum(
            parameter.numel()
            for parameter in self.parameters()
            if not parameter.requires_grad
        )
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return {"frozen": frozen, "trainable": trainable, "total": frozen + trainable}

    def _apply_freezing(self) -> None:
        encoder = getattr(self.basecaller, "encoder", None)
        if encoder is None:
            raise TypeError("Bonito basecaller must expose encoder")
        if self.freeze_encoder:
            for name in ("conv", "transformer_encoder", "upsample"):
                module = getattr(encoder, name, None)
                if module is None:
                    raise TypeError(f"Bonito v5 encoder must expose {name}")
                module.requires_grad_(False)
        if self.freeze_basecalling_head:
            crf = getattr(encoder, "crf", None)
            if crf is None:
                raise TypeError("Bonito v5 encoder must expose crf")
            crf.requires_grad_(False)


def head_config_from_checkpoint(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Restore explicit head architecture, defaulting legacy checkpoints to mean."""
    if "head_config" in payload:
        config = dict(payload["head_config"])
        validate_head_type(config.get("head_type", "legacy"))
        validate_length_head_mode(config["length_head_mode"])
        if config.get("head_type") == "tcn":
            # Old TCN checkpoints used the unscaled sum. Do not reinterpret them
            # using the smaller default introduced for new training runs.
            config["tcn_length_sum_scale"] = validate_tcn_length_sum_scale(
                config.get("tcn_length_sum_scale", 1.0)
            )
            config["tcn_length_features"] = validate_tcn_length_features(
                config.get("tcn_length_features", "sum")
            )
        return config
    state = payload["model_state_dict"]
    return {
        "length_head_mode": "mean",
        "hidden_features": state["multitask.length_head.length_regressor.weight"].shape[
            1
        ],
        "dropout": 0.0,
    }


def load_bonito_v5_model(
    model_directory: Path,
    *,
    device: str,
) -> nn.Module:
    """Load an external Bonito v5 model without copying it into the repository."""
    if not model_directory.is_dir():
        raise FileNotFoundError(
            f"Bonito model directory does not exist: {model_directory}"
        )
    if not (model_directory / "config.toml").is_file():
        raise FileNotFoundError(
            f"Bonito model config does not exist: {model_directory / 'config.toml'}"
        )

    from bonito.util import load_model

    model: Any = load_model(
        str(model_directory),
        device=device,
        half=False,
        compile=False,
    )
    return model
