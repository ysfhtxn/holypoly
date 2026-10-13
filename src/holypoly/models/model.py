"""Frozen Bonito v5 wrapper for Poly auxiliary-head training."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self, cast

from torch import Tensor, nn

from holypoly.models.poly_heads import BonitoMultiTaskOutput, BonitoV5MultiTaskModel


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
    ) -> None:
        super().__init__()
        self.multitask = BonitoV5MultiTaskModel(
            basecaller,
            hidden_features=hidden_features,
            dropout=dropout,
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
        self, signal: Tensor, raw_padding_mask: Tensor | None = None
    ) -> BonitoMultiTaskOutput:
        """Return unchanged basecall scores plus location and length outputs."""
        return cast(BonitoMultiTaskOutput, self.multitask(signal, raw_padding_mask))

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
