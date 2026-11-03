"""Numerical objectives, masking and gradients without real data or training."""

from __future__ import annotations

import math

import pytest
import torch

from holypoly.models.poly_heads import (
    BonitoMultiTaskOutput,
    PolyLengthOutput,
    PolyLocationOutput,
    ReadStatus,
)
from holypoly.training.loss_config import LENGTH_LOSS_MODES
from holypoly.training.losses import PolyAuxiliaryLoss, PolyLossWeights


def _output(predictions: torch.Tensor) -> BonitoMultiTaskOutput:
    """Build a minimal [batch, time=2] output; final timestep is padding."""
    count = len(predictions)
    padding = torch.tensor([[False, True]]).expand(count, 2)
    return BonitoMultiTaskOutput(
        basecall_scores=torch.zeros(2, count, 3),
        encoder_features=torch.zeros(count, 2, 4),
        feature_padding_mask=padding,
        location=PolyLocationOutput(
            location_logits=torch.zeros(count, 2, 3, requires_grad=True),
            status_logits=torch.zeros(count, 3, requires_grad=True),
            padding_mask=padding,
        ),
        length=PolyLengthOutput(
            length_nt=predictions,
            status_logits=torch.zeros(count, 3, requires_grad=True),
            predicted_length_nt=predictions,
        ),
    )


@pytest.mark.parametrize("mode", LENGTH_LOSS_MODES)
def test_length_objective_values_and_gradients_including_zero_and_failure(
    mode: str,
) -> None:
    predictions = torch.tensor([0.0, 30.0, 100.0, 777.0], requires_grad=True)
    output = _output(predictions)
    criterion = PolyAuxiliaryLoss(
        PolyLossWeights(lambda_location=0, lambda_status=0, length_loss=mode)
    )
    losses = criterion(
        output,
        torch.full((4, 2), -100),
        torch.tensor([0.0, 0.0, 150.0, math.nan]),
        torch.tensor([-100, -100, -100, ReadStatus.FAILURE]),
    )
    if mode == "mae":
        expected = 80 / 3
        gradient = [0, 1 / 3, -1 / 3, 0]
    elif mode == "huber":
        # delta=10: H(30)=10*(30-5), H(50)=10*(50-5).
        expected = (250 + 450) / 3
        gradient = [0, 10 / 3, -10 / 3, 0]
    else:
        difference = math.log1p(150) - math.log1p(100)
        expected = (math.log1p(30) - 0.5 + 0.5 * difference**2) / 3
        gradient = [0, 1 / (3 * 31), -difference / (3 * 101), 0]
    assert losses["length_loss"].item() == pytest.approx(expected)
    assert losses["length_examples"].item() == 3
    losses["total_loss"].backward()
    torch.testing.assert_close(predictions.grad, torch.tensor(gradient))


def test_huber_transition_and_custom_delta_use_raw_nt() -> None:
    predictions = torch.tensor([2.0, 4.0, 8.0], requires_grad=True)
    criterion = PolyAuxiliaryLoss(
        PolyLossWeights(length_loss="huber", huber_delta_nt=4)
    )
    losses = criterion(
        _output(predictions),
        torch.full((3, 2), -100),
        torch.zeros(3),
        torch.full((3,), -100),
    )
    # e=2 (quadratic), e=4 (transition), e=8 (linear), without log transforms.
    assert losses["length_loss"].item() == pytest.approx((2 + 8 + 24) / 3)
    losses["length_loss"].backward()
    torch.testing.assert_close(predictions.grad, torch.tensor([2 / 3, 4 / 3, 4 / 3]))


@pytest.mark.parametrize("mode", LENGTH_LOSS_MODES)
def test_all_failure_batch_keeps_differentiable_zero_length_loss(mode: str) -> None:
    predictions = torch.tensor([10.0, 150.0], requires_grad=True)
    losses = PolyAuxiliaryLoss(PolyLossWeights(length_loss=mode))(
        _output(predictions),
        torch.full((2, 2), -100),
        torch.full((2,), math.nan),
        torch.full((2,), int(ReadStatus.FAILURE)),
    )
    assert losses["length_loss"].item() == 0
    assert losses["length_examples"].item() == 0
    losses["length_loss"].backward()
    torch.testing.assert_close(predictions.grad, torch.zeros(2))


@pytest.mark.parametrize("mode", LENGTH_LOSS_MODES)
@pytest.mark.parametrize("invalid", [math.nan, math.inf, -1.0])
def test_invalid_length_only_targets_are_rejected(mode: str, invalid: float) -> None:
    with pytest.raises(ValueError, match="non-failed training lengths"):
        PolyAuxiliaryLoss(PolyLossWeights(length_loss=mode))(
            _output(torch.tensor([20.0])),
            torch.full((1, 2), -100),
            torch.tensor([invalid]),
            torch.tensor([-100]),
        )
