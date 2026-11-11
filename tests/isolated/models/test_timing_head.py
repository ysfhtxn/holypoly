"""Synthetic timing-fusion behavior, exact fallback, and unchanged basecalling."""

import io
import math

import pytest
import torch

from holypoly.models.model import HolypolyModel, head_config_from_checkpoint
from holypoly.models.pooling_heads import SumLengthHead
from holypoly.models.timing_head import TimingReferenceLengthHead
from tests.isolated.helpers import tiny_v5_basecaller


def test_default_model_has_identical_state_and_predictions() -> None:
    torch.manual_seed(9)
    legacy = HolypolyModel(
        tiny_v5_basecaller(), hidden_features=6, length_head_mode="sum"
    )
    torch.manual_seed(9)
    disabled = HolypolyModel(
        tiny_v5_basecaller(),
        hidden_features=6,
        length_head_mode="sum",
        use_timing_reference=False,
    )
    assert legacy.head_config == disabled.head_config
    assert "use_timing_reference" not in legacy.head_config
    assert legacy.state_dict().keys() == disabled.state_dict().keys()
    for key, value in legacy.state_dict().items():
        torch.testing.assert_close(value, disabled.state_dict()[key], rtol=0, atol=0)
    x = torch.randn(2, 1, 132)
    legacy.eval()
    disabled.eval()
    torch.testing.assert_close(
        legacy(x).length.length_nt, disabled(x).length.length_nt, rtol=0, atol=0
    )


def test_missing_reference_bypasses_trained_residual_exactly() -> None:
    head = TimingReferenceLengthHead(4, 6)
    original = SumLengthHead(4, 6)
    original.load_state_dict(
        {
            key: value
            for key, value in head.state_dict().items()
            if not key.startswith("timing_residual.")
        }
    )
    with torch.no_grad():
        head.timing_residual[-1].bias.fill_(3)
    x = torch.randn(3, 7, 4)
    mask = torch.tensor([[False] * 7, [False] * 5 + [True] * 2, [False] * 7])
    metadata = torch.tensor([[0, 0, 0], [math.log1p(12), 0.1, 1], [0, 0, 0]])
    output = head(
        x,
        mask,
        raw_lengths_samples=torch.tensor([42, 30, 42]),
        timing_reference=metadata,
    )
    expected = original(x, mask)
    torch.testing.assert_close(
        output.length_nt[[0, 2]], expected.length_nt[[0, 2]], rtol=0, atol=0
    )
    assert output.length_nt[1] > expected.length_nt[1]
    torch.testing.assert_close(
        output.status_logits, expected.status_logits, rtol=0, atol=0
    )


def test_enabled_checkpoint_roundtrip_and_original_crf_output() -> None:
    torch.manual_seed(13)
    model = HolypolyModel(
        tiny_v5_basecaller(),
        hidden_features=6,
        length_head_mode="sum",
        use_timing_reference=True,
    ).eval()
    signal = torch.randn(2, 1, 132)
    metadata = torch.tensor([[math.log1p(12), 0.1, 1], [0, 0, 0]])
    actual = model(signal, timing_reference=metadata)
    without_reference = model(signal)
    torch.testing.assert_close(
        actual.length.length_nt, without_reference.length.length_nt, rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual.basecall_scores, model.basecaller(signal), rtol=0, atol=0
    )
    stream = io.BytesIO()
    torch.save(
        {"head_config": model.head_config, "model_state_dict": model.state_dict()},
        stream,
    )
    stream.seek(0)
    payload = torch.load(stream, weights_only=False)
    restored = HolypolyModel(
        tiny_v5_basecaller(), **head_config_from_checkpoint(payload)
    ).eval()
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    torch.testing.assert_close(
        restored(signal, timing_reference=metadata).length.length_nt,
        actual.length.length_nt,
        rtol=0,
        atol=0,
    )


def test_reference_gradient_and_frozen_basecaller() -> None:
    model = HolypolyModel(
        tiny_v5_basecaller(),
        hidden_features=6,
        length_head_mode="sum",
        use_timing_reference=True,
    )
    x = torch.randn(2, 1, 132)
    metadata = torch.tensor([[math.log1p(12), 0.1, 1], [0, 0, 0]])
    output = model(x, timing_reference=metadata)
    loss = torch.nn.functional.huber_loss(
        output.length.length_nt, torch.tensor([150.0, 0.0]), delta=10
    )
    loss.backward()
    assert model.length_head.timing_residual[-1].weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.basecaller.parameters())
    assert model.length_head.length_regressor.weight.grad is not None


def test_disabled_or_unsupported_architecture_rejects_timing() -> None:
    with pytest.raises(ValueError, match="sum pooling"):
        HolypolyModel(tiny_v5_basecaller(), use_timing_reference=True)
    model = HolypolyModel(tiny_v5_basecaller())
    with pytest.raises(ValueError, match="does not enable"):
        model(torch.randn(1, 1, 132), timing_reference=torch.zeros(1, 3))
    head = TimingReferenceLengthHead(4, 6)
    with pytest.raises(ValueError, match="finite"):
        head(
            torch.randn(1, 4, 4),
            raw_lengths_samples=torch.tensor([24]),
            timing_reference=torch.full((1, 3), float("nan")),
        )
