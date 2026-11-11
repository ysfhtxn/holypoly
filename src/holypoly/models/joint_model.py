"""Standard joint model with selected final Transformer blocks unfrozen."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from torch import nn

from holypoly.models.model import HolypolyModel


class JointHolypolyModel(HolypolyModel):
    """Unfreeze only the final N v5 Transformer blocks, retaining the original CRF.

    Frozen modules stay in eval mode, including batch normalization buffers.
    The basecalling output and parameter names are unchanged.
    """

    def __init__(
        self, basecaller: nn.Module, *, unfreeze_last_layers: int = 1, **kwargs: Any
    ) -> None:
        if any(
            kwargs.get(name, True) is not True
            for name in ("freeze_encoder", "freeze_basecalling_head")
        ):
            raise ValueError(
                "joint mode only unfreezes selected final Transformer blocks"
            )
        super().__init__(basecaller, **kwargs)
        self.basecaller.requires_grad_(False)
        stack = self.basecaller.encoder.transformer_encoder
        layers = list(stack.children())
        if unfreeze_last_layers < 1 or unfreeze_last_layers > len(layers):
            raise ValueError(
                "unfreeze_last_layers must select existing final Transformer blocks"
            )
        self.unfreeze_last_layers = unfreeze_last_layers
        for layer in layers[-unfreeze_last_layers:]:
            layer.requires_grad_(True)

    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        """Yield exactly the unfrozen encoder blocks and auxiliary-head parameters."""
        yield from (
            parameter for parameter in self.parameters() if parameter.requires_grad
        )


def create_joint_optimizer(
    model: JointHolypolyModel,
    *,
    head_learning_rate: float,
    encoder_learning_rate: float,
    weight_decay: float,
) -> Any:
    """Use separately recorded learning rates without changing the frozen CRF."""
    import math

    import torch

    if (
        not all(
            math.isfinite(value)
            for value in (head_learning_rate, encoder_learning_rate, weight_decay)
        )
        or head_learning_rate <= 0
        or encoder_learning_rate <= 0
        or weight_decay < 0
    ):
        raise ValueError("invalid optimizer learning rates or weight decay")
    encoder = [p for p in model.basecaller.parameters() if p.requires_grad]
    heads = [*model.location_head.parameters(), *model.length_head.parameters()]
    optimizer = torch.optim.AdamW(
        [
            {"params": heads, "lr": head_learning_rate, "group_name": "poly_heads"},
            {
                "params": encoder,
                "lr": encoder_learning_rate,
                "group_name": "encoder_tail",
            },
        ],
        weight_decay=weight_decay,
    )
    optimized = {id(p) for group in optimizer.param_groups for p in group["params"]}
    if optimized != {id(p) for p in model.trainable_parameters()}:
        raise ValueError("optimizer and trainable model parameters differ")
    return optimizer
