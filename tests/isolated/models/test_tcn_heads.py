"""Synthetic TCN architecture, padding isolation and checkpoint compatibility."""

from __future__ import annotations

import io
from copy import deepcopy

import pytest
import torch
from torch import nn

from holypoly.models.length_head_config import DEFAULT_TCN_LENGTH_SUM_SCALE
from holypoly.models.model import HolypolyModel, head_config_from_checkpoint
from holypoly.models.poly_heads import ReadStatus
from holypoly.models.tcn_heads import (
    DILATIONS,
    TCNLengthHead,
    TCNLocationHead,
)
from holypoly.training.losses import PolyAuxiliaryLoss, PolyLossWeights
from tests.isolated.helpers import tiny_v5_basecaller


@pytest.mark.parametrize("batch,time", [(1, 1), (2, 7), (3, 31)])
def test_tcn_outputs_and_independent_branches(batch: int, time: int) -> None:
    """Both branches preserve T and emit raw logits with the legacy field shapes."""
    length, location = TCNLengthHead(8).eval(), TCNLocationHead(8).eval()
    features = torch.randn(batch, time, 8)
    mask = torch.zeros(batch, time, dtype=torch.bool)
    mask[:, max(1, time // 2) :] = True
    observed = []
    hooks = [
        block.register_forward_hook(lambda m, args, out: observed.append(out))
        for head in (length, location)
        for block in head.blocks
    ]
    out, state = length(features, mask), location(features, mask)
    for hook in hooks:
        hook.remove()
    assert out.length_nt.shape == (batch,)
    assert out.status_logits.shape == state.status_logits.shape == (batch, 3)
    assert (out.length_nt >= 0).all()
    assert state.location_logits.shape == (batch, time, 3)
    assert state.location_logits.count_nonzero() == 0
    assert len(observed) == 8
    for hidden in observed:
        assert hidden.shape == (batch, time, 128)
        assert hidden[mask].count_nonzero() == 0
    assert not {id(p) for p in length.parameters()} & {
        id(p) for p in location.parameters()
    }
    for head in (length, location):
        assert [block.conv1.dilation[0] for block in head.blocks] == list(DILATIONS)
        for block in head.blocks:
            for conv in (block.conv1, block.conv2):
                assert conv.kernel_size == (3,) and conv.stride == (1,)
                assert conv.padding == conv.dilation
            assert block.dropout1.p == block.dropout2.p == 0.1
    # Biases demonstrate that status values remain logits, without a softmax.
    with torch.no_grad():
        length.status_classifier[-1].weight.zero_()
        length.status_classifier[-1].bias.copy_(torch.tensor([-4.0, 2.0, 7.0]))
    torch.testing.assert_close(
        length(features, mask).status_logits[0], torch.tensor([-4.0, 2.0, 7.0])
    )


@pytest.mark.parametrize("head_class", [TCNLengthHead, TCNLocationHead])
def test_padding_values_extensions_and_gradients_are_isolated(head_class) -> None:
    """Affine LN/conv biases cannot send padded activations back into valid frames."""
    torch.manual_seed(71)
    head = head_class(5).eval()
    with torch.no_grad():
        for name, parameter in head.named_parameters():
            if name.endswith("bias"):
                parameter.uniform_(-0.4, 0.4)
    mask = torch.tensor([[False] * 7 + [True] * 6, [False] * 13])
    features = torch.randn(2, 13, 5, requires_grad=True)
    original = head(features, mask)
    contaminated = features.detach().clone()
    contaminated[mask] = torch.nan
    changed = head(contaminated, mask)
    torch.testing.assert_close(
        original.status_logits, changed.status_logits, rtol=0, atol=0
    )
    extended_mask = torch.cat([mask, torch.ones(2, 9, dtype=torch.bool)], 1)
    extended = torch.cat([contaminated, torch.full((2, 9, 5), torch.nan)], 1)
    padded = head(extended, extended_mask)
    torch.testing.assert_close(
        original.status_logits, padded.status_logits, rtol=1e-5, atol=1e-5
    )
    if isinstance(head, TCNLengthHead):
        torch.testing.assert_close(
            original.length_nt, padded.length_nt, rtol=1e-5, atol=1e-5
        )
        objective = original.length_nt.sum() + original.status_logits.square().mean()
    else:
        objective = original.status_logits.square().mean()
    objective.backward()
    assert features.grad[mask].count_nonzero() == 0
    assert features.grad[~mask].abs().sum() > 0
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in head.parameters()
    )


def test_tcn_masked_sum_and_mean() -> None:
    """Repeated frames double the length sum but preserve the global status mean."""
    features = torch.tensor([[[1.0, 2.0]] * 4, [[1.0, 2.0]] * 4])
    mask = torch.tensor([[False] * 4, [False, False, True, True]])
    for head, expected in (
        (
            TCNLengthHead(2, tcn_length_sum_scale=0.25, tcn_length_features="sum"),
            torch.tensor([12.0, 6.0]),
        ),
        (TCNLocationHead(2), torch.tensor([3.0, 3.0])),
    ):
        head.projection = nn.Identity()
        head.blocks = nn.ModuleList()
        head.status_classifier = nn.Linear(2, 3)
        with torch.no_grad():
            head.status_classifier.weight.fill_(1)
            head.status_classifier.bias.zero_()
        if isinstance(head, TCNLengthHead):
            head.length_regressor = nn.Linear(2, 1)
            with torch.no_grad():
                head.length_regressor.weight.fill_(1)
                head.length_regressor.bias.zero_()
        output = head(features, mask)
        torch.testing.assert_close(output.status_logits[:, 0], expected)
        if isinstance(head, TCNLengthHead):
            torch.testing.assert_close(
                output.length_nt,
                torch.nn.functional.softplus(expected * head.tcn_length_sum_scale),
            )


def test_tcn_length_concatenates_sum_mean_and_valid_frame_duration() -> None:
    """Different valid lengths change sum/duration but not constant-frame means."""
    head = TCNLengthHead(128, tcn_length_sum_scale=0.25).eval()
    head.projection = nn.Identity()
    head.blocks = nn.ModuleList()
    counts = torch.tensor([[4.0], [2.0], [1.0]])
    mask = torch.arange(6)[None] >= counts.long()
    features = torch.ones(3, 6, 128).masked_fill(mask.unsqueeze(-1), torch.nan)
    features.requires_grad_()
    captured = []
    hook = head.length_regressor.register_forward_pre_hook(
        lambda _, args: captured.append(args[0].detach().clone())
    )
    output = head(features, mask)
    hook.remove()
    assert captured[0].shape == (3, 257)
    assert head.length_regressor[0].in_features == 257
    assert head.status_classifier[0].in_features == 128
    torch.testing.assert_close(captured[0][:, :128], (0.25 * counts).expand(-1, 128))
    torch.testing.assert_close(captured[0][:, 128:256], torch.ones(3, 128))
    torch.testing.assert_close(captured[0][:, 256:], torch.log1p(counts))
    output.length_nt.sum().backward()
    assert torch.isfinite(features.grad).all()
    assert features.grad[mask].count_nonzero() == 0
    assert features.grad[~mask].abs().sum() > 0


def test_additional_length_inputs_leave_status_predictions_unchanged() -> None:
    """Changing regression inputs alone never changes either status definition."""
    old = TCNLengthHead(4, tcn_length_features="sum").eval()
    new = TCNLengthHead(4).eval()
    state = new.state_dict()
    state.update(
        {
            name: value
            for name, value in old.state_dict().items()
            if value.shape == state[name].shape
        }
    )
    new.load_state_dict(state, strict=True)
    features = torch.randn(2, 13, 4)
    mask = torch.arange(13)[None] >= torch.tensor([13, 7])[:, None]
    torch.testing.assert_close(
        old(features, mask).status_logits,
        new(features, mask).status_logits,
        rtol=0,
        atol=0,
    )


def test_fixed_sum_scale_prevents_softplus_underflow_and_preserves_gradients() -> None:
    """Reproduce a large negative pre-Softplus value with the actual length MLP."""
    unscaled = TCNLengthHead(
        1, tcn_length_sum_scale=1.0, tcn_length_features="sum"
    ).eval()
    # Isolate the regression failure with known per-frame activations; use the
    # unchanged 128->64->1 GELU MLP and 1000 valid encoder frames.
    unscaled.projection = nn.Linear(1, 128, bias=False)
    unscaled.blocks = nn.ModuleList()
    with torch.no_grad():
        unscaled.projection.weight.fill_(1.0)
        unscaled.length_regressor[0].weight.fill_(1.0 / 128)
        unscaled.length_regressor[0].bias.zero_()
        unscaled.length_regressor[2].weight.fill_(-1.0 / 64)
        unscaled.length_regressor[2].bias.zero_()
    scaled = deepcopy(unscaled)
    scaled.tcn_length_sum_scale = DEFAULT_TCN_LENGTH_SUM_SCALE
    features = torch.ones(1, 1200, 1, requires_grad=True)
    mask = torch.arange(1200)[None] >= 1000
    pre_softplus = []
    hook = unscaled.length_regressor.register_forward_hook(
        lambda _, args, output: pre_softplus.append(float(output.detach().squeeze()))
    )
    before = unscaled(features, mask)
    hook.remove()
    before.length_nt.sum().backward()
    assert pre_softplus[0] == pytest.approx(-1000)
    assert before.length_nt.item() == 0
    assert unscaled.length_regressor[-1].weight.grad.count_nonzero() == 0
    assert features.grad.count_nonzero() == 0
    features.grad = None
    after = scaled(features, mask)
    after.length_nt.sum().backward()
    assert after.length_nt.item() > 0.1
    assert scaled.length_regressor[-1].weight.grad.abs().max() > 0.1
    assert torch.isfinite(features.grad).all()
    assert features.grad[~mask].abs().sum() > 1e-3
    assert features.grad[mask].count_nonzero() == 0
    # Length scaling must not change either the status path or state_dict keys.
    torch.testing.assert_close(
        before.status_logits, after.status_logits, rtol=0, atol=0
    )
    assert unscaled.state_dict().keys() == scaled.state_dict().keys()


@pytest.mark.parametrize("scale", [0.0, -1.0, float("inf"), float("nan"), True])
def test_tcn_sum_scale_rejects_nonpositive_or_nonfinite_values(scale: float) -> None:
    """Invalid multipliers fail before any model or real data is loaded."""
    with pytest.raises(ValueError, match="finite and positive"):
        TCNLengthHead(4, tcn_length_sum_scale=scale)


@pytest.mark.parametrize("scale", [1.0, 0.001])
def test_old_tcn_checkpoint_defaults_to_original_sum_features(scale: float) -> None:
    """Restore both pre-scaling and scaled historical 128-input TCN checkpoints."""
    model = HolypolyModel(
        tiny_v5_basecaller(),
        head_type="tcn",
        length_head_mode="sum",
        tcn_length_sum_scale=scale,
        tcn_length_features="sum",
    ).eval()
    old_config = dict(model.head_config)
    if scale == 1.0:
        del old_config["tcn_length_sum_scale"]
    del old_config["tcn_length_features"]
    payload = {"head_config": old_config, "model_state_dict": model.state_dict()}
    restored = HolypolyModel(
        tiny_v5_basecaller(), **head_config_from_checkpoint(payload)
    ).eval()
    restored.load_state_dict(payload["model_state_dict"], strict=True)
    assert "tcn_length_features" not in old_config
    assert restored.head_config["tcn_length_sum_scale"] == scale
    assert restored.length_head.tcn_length_sum_scale == scale
    assert restored.head_config["tcn_length_features"] == "sum"
    assert restored.length_head.length_regressor[0].in_features == 128
    features = torch.randn(2, 17, 8)
    torch.testing.assert_close(
        model.length_head(features).length_nt,
        restored.length_head(features).length_nt,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("head_class", [TCNLengthHead, TCNLocationHead])
def test_tcn_rejects_empty_or_invalid_feature_masks(head_class) -> None:
    """Each sample requires at least one valid encoder frame."""
    head = head_class(4)
    with pytest.raises(ValueError, match="at least one non-padding"):
        head(torch.randn(2, 4, 4), torch.ones(2, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="boolean"):
        head(torch.randn(2, 4, 4), torch.zeros(2, 4))
    with pytest.raises(ValueError, match="at least one timestep"):
        head(torch.randn(2, 0, 4))


@pytest.mark.parametrize("head_type", ["legacy", "tcn"])
def test_checkpoint_roundtrip_frozen_encoder_and_original_basecall(
    head_type: str,
) -> None:
    """Synthetic frozen models exercise unchanged loss, old loading and all labels."""
    torch.manual_seed(53)
    model = HolypolyModel(
        tiny_v5_basecaller(),
        length_head_mode="sum",
        head_type=head_type,
        **({"tcn_length_sum_scale": 0.002} if head_type == "tcn" else {}),
    ).eval()
    signal = torch.randn(3, 1, 132)
    raw_mask = torch.arange(132)[None] >= torch.tensor([132, 80, 12])[:, None]
    output = model(signal, raw_mask)
    with torch.no_grad():
        full = model(signal)
        torch.testing.assert_close(
            full.basecall_scores, model.basecaller(signal), rtol=0, atol=0
        )
    criterion = PolyAuxiliaryLoss(
        PolyLossWeights(lambda_location=0, length_loss="huber")
    )
    losses = criterion(
        output,
        torch.full((3, 22), -100),
        torch.tensor([40.0, 0.0, torch.nan]),
        torch.tensor([ReadStatus.POLY_PRESENT, ReadStatus.ZERO_NT, ReadStatus.FAILURE]),
    )
    losses["total_loss"].backward()
    assert torch.isfinite(losses["total_loss"])
    assert all(p.grad is None for p in model.basecaller.parameters())
    for branch in (model.length_head, model.location_head):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in branch.parameters()
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
    actual = restored(signal, raw_mask)
    torch.testing.assert_close(
        output.length.length_nt, actual.length.length_nt, rtol=0, atol=0
    )
    torch.testing.assert_close(
        output.location.status_logits, actual.location.status_logits, rtol=0, atol=0
    )
    if head_type == "legacy":
        assert "head_type" not in model.head_config
    else:
        assert model.head_config["head_type"] == "tcn"
        assert restored.head_config["tcn_length_sum_scale"] == 0.002
        assert restored.head_config["tcn_length_features"] == "sum_mean_duration"
        assert restored.length_head.length_regressor[0].in_features == 257


def test_explicit_legacy_is_exactly_the_default() -> None:
    """New selection preserves legacy RNG order, checkpoint keys and predictions."""
    torch.manual_seed(17)
    default = HolypolyModel(tiny_v5_basecaller(), length_head_mode="sum").eval()
    torch.manual_seed(17)
    explicit = HolypolyModel(
        tiny_v5_basecaller(), length_head_mode="sum", head_type="legacy"
    ).eval()
    assert default.head_config == explicit.head_config
    for key, value in default.state_dict().items():
        torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
    old = head_config_from_checkpoint({"model_state_dict": default.state_dict()})
    assert old["length_head_mode"] == "mean" and "head_type" not in old


@pytest.mark.parametrize(
    "options",
    [
        {"head_type": "unknown"},
        {"head_type": "tcn", "hidden_features": 64, "length_head_mode": "sum"},
        {"head_type": "tcn", "length_head_mode": "sum", "use_timing_reference": True},
        {"head_type": "tcn", "length_head_mode": "mean"},
        {"head_type": "tcn", "length_head_mode": "sum", "tcn_length_features": "bad"},
    ],
)
def test_unsupported_head_options_fail_explicitly(options: dict) -> None:
    """No silent alteration of TCN width, pooling, or reference fusion."""
    with pytest.raises(ValueError):
        HolypolyModel(tiny_v5_basecaller(), **options)
    with pytest.raises(ValueError, match="head_type"):
        head_config_from_checkpoint({"head_config": {"head_type": "unknown"}})
