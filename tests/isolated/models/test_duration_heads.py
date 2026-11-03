"""Synthetic head comparisons: duration isolation, padding and legacy compatibility."""

from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from holypoly.models.duration_heads import DurationLengthHead
from holypoly.models.model import HolypolyModel, head_config_from_checkpoint
from holypoly.models.poly_heads import PolyLengthHead, ReadStatus
from holypoly.training.losses import PolyAuxiliaryLoss, PolyLossWeights
from tests.isolated.helpers import tiny_v5_basecaller


def test_duration_only_is_independent_of_waveform_but_uses_exact_raw_counts() -> None:
    """Two counts within one encoder stride must still be distinguishable."""
    head = DurationLengthHead(4, 6, mode="duration_only").eval()
    with torch.no_grad():
        head.length_regressor[0].weight.fill_(1)
        head.length_regressor[0].bias.zero_()
        head.length_regressor[3].weight.fill_(1)
        head.length_regressor[3].bias.zero_()
    features = torch.randn(2, 4, 4)
    lengths = torch.tensor([13, 14])
    original = head(features, raw_lengths_samples=lengths).length_nt
    changed = head(features * 100 + 10, raw_lengths_samples=lengths).length_nt
    torch.testing.assert_close(original, changed, rtol=0, atol=0)
    assert original[0] < original[1]
    assert isinstance(head.projection, nn.Identity)


@pytest.mark.parametrize("mode", ["duration_only", "mean_duration"])
@pytest.mark.parametrize("batch_first", [True, False])
def test_duration_heads_ignore_padding_and_have_finite_masked_loss(
    mode: str, batch_first: bool
) -> None:
    torch.manual_seed(13)
    head = DurationLengthHead(4, 6, mode=mode, batch_first=batch_first).eval()
    features = torch.randn(3, 5, 4, requires_grad=True)
    mask = torch.arange(5)[None, :] >= torch.tensor([5, 3, 2])[:, None]
    lengths = torch.tensor([30, 13, 7])
    inputs = features if batch_first else features.transpose(0, 1)
    output = head(inputs, mask, raw_lengths_samples=lengths)
    padded = features.detach().masked_fill(mask[..., None], 10000)
    padded = torch.cat((padded, torch.full((3, 2, 4), -999.0)), dim=1)
    extended_mask = torch.cat((mask, torch.ones(3, 2, dtype=torch.bool)), dim=1)
    extended = head(
        padded if batch_first else padded.transpose(0, 1),
        extended_mask,
        raw_lengths_samples=lengths,
    )
    torch.testing.assert_close(output.length_nt, extended.length_nt)
    targets = torch.tensor([0.0, 150.0, torch.nan])
    statuses = torch.tensor([-100, -100, int(ReadStatus.FAILURE)])
    losses = head.loss(output, targets, statuses)
    assert torch.isfinite(losses["total_loss"])
    losses["total_loss"].backward()
    assert head.length_regressor[0].weight.grad is not None
    assert torch.count_nonzero(head.length_regressor[0].weight.grad) > 0
    if mode == "mean_duration":
        assert features.grad is not None
        assert torch.count_nonzero(features.grad[mask]) == 0
    else:
        assert features.grad is None


@pytest.mark.parametrize("bad", [None, torch.tensor([0]), torch.tensor([1.5])])
def test_duration_heads_reject_missing_empty_or_fractional_raw_counts(bad) -> None:
    head = DurationLengthHead(4, 6, mode="duration_only")
    with pytest.raises(ValueError, match="raw_lengths_samples"):
        head(torch.randn(1, 3, 4), raw_lengths_samples=bad)


@pytest.mark.parametrize("mode", ["mean", "sum", "duration_only", "mean_duration"])
@pytest.mark.parametrize("length_loss", ["log_smooth_l1", "mae", "huber"])
def test_model_preserves_basecalling_and_roundtrips_head_configuration(
    mode: str,
    length_loss: str,
) -> None:
    """CPU unit backward uses a tiny synthetic encoder, never real model weights."""
    torch.manual_seed(17)
    basecaller = tiny_v5_basecaller().eval()
    model = HolypolyModel(basecaller, hidden_features=6, length_head_mode=mode).eval()
    signal = torch.randn(3, 1, 120)
    mask = torch.arange(120)[None, :] >= torch.tensor([120, 61, 13])[:, None]
    with torch.no_grad():
        expected = basecaller(signal.masked_fill(mask[:, None, :], 0))
    output = model(signal, mask)
    torch.testing.assert_close(output.basecall_scores, expected, rtol=0, atol=0)
    if mode == "duration_only":
        with torch.no_grad():
            isolated = model.length_head(
                output.encoder_features,
                output.feature_padding_mask,
                raw_lengths_samples=torch.tensor([120, 61, 13]),
            )
        torch.testing.assert_close(output.length.length_nt, isolated.length_nt)
    criterion = PolyAuxiliaryLoss(PolyLossWeights(length_loss=length_loss))
    losses = criterion(
        output,
        torch.full(output.feature_padding_mask.shape, -100),
        torch.tensor([0.0, 20.0, 150.0]),
        torch.full((3,), -100),
    )
    losses["total_loss"].backward()
    assert all(p.grad is None for p in model.basecaller.parameters())
    assert any(
        p.grad is not None and torch.count_nonzero(p.grad) > 0
        for p in model.length_head.parameters()
    )
    buffer = io.BytesIO()
    torch.save(
        {"model_state_dict": model.state_dict(), "head_config": model.head_config},
        buffer,
    )
    buffer.seek(0)
    payload = torch.load(buffer, weights_only=False)
    restored = HolypolyModel(
        tiny_v5_basecaller(), **head_config_from_checkpoint(payload)
    ).eval()
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    torch.testing.assert_close(
        restored(signal, mask).length.length_nt, output.length.length_nt
    )


def test_default_mean_retains_original_head_and_legacy_state_dict() -> None:
    model = HolypolyModel(tiny_v5_basecaller(), hidden_features=6).eval()
    assert type(model.length_head) is PolyLengthHead
    assert type(model.length_head.length_regressor) is nn.Linear
    payload = {"model_state_dict": model.state_dict()}
    config = head_config_from_checkpoint(payload)
    assert config == {"length_head_mode": "mean", "hidden_features": 6, "dropout": 0.0}
    restored = HolypolyModel(tiny_v5_basecaller(), **config).eval()
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    signal = torch.randn(2, 1, 120)
    torch.testing.assert_close(
        model(signal).length.length_nt, restored(signal).length.length_nt
    )
