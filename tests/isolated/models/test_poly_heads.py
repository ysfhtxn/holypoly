from __future__ import annotations

import io
from pathlib import Path
from types import MethodType

import pytest
import toml
import torch
import torch.nn.functional as functional
from bonito.crf.model import SeqdistModel
from bonito.transformer import Model as TransformerModel
from torch import nn

from holypoly.models.poly_heads import (
    BONITO_V5_STRIDE,
    BonitoV5MultiTaskModel,
    PolyClass,
    PolyLengthHead,
    PolyLocationHead,
    ReadStatus,
    decode_location_intervals,
    encoder_interval_to_raw_interval,
    make_location_targets,
    raw_interval_to_encoder_interval,
)
from tests.isolated.helpers import tiny_v5_basecaller


def _assert_nonzero_gradients(module: nn.Module) -> None:
    for parameter in module.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.count_nonzero(parameter.grad) > 0


def test_poly_location_head_masks_padding_and_backpropagates() -> None:
    torch.manual_seed(7)
    features = torch.randn(3, 6, 8, requires_grad=True)
    padding_mask = torch.tensor(
        [
            [False, False, False, False, False, False],
            [False, False, False, True, True, True],
            [False, False, True, True, True, True],
        ]
    )
    location_targets = torch.tensor(
        [
            [0, 1, 1, 1, 0, 0],
            [0, 2, 0, -100, -100, -100],
            [0, 0, -100, -100, -100, -100],
        ]
    )
    status_targets = torch.tensor(
        [ReadStatus.POLY_PRESENT, ReadStatus.POLY_PRESENT, ReadStatus.ZERO_NT]
    )
    head = PolyLocationHead(input_features=8, hidden_features=6)

    output = head(features, padding_mask)
    losses = head.loss(output, location_targets, status_targets)
    losses["total_loss"].backward()

    assert output.location_logits.shape == (3, 6, len(PolyClass))
    assert output.status_logits.shape == (3, len(ReadStatus))
    assert torch.count_nonzero(output.location_logits[padding_mask]) == 0
    assert torch.isfinite(losses["total_loss"])
    assert features.grad is not None
    assert torch.count_nonzero(features.grad[padding_mask]) == 0
    _assert_nonzero_gradients(head)


def test_poly_length_head_uses_log_loss_and_supports_failure() -> None:
    torch.manual_seed(11)
    features = torch.randn(5, 3, 10, requires_grad=True)
    padding_mask = torch.tensor(
        [
            [False, False, False, False, False],
            [False, False, False, True, True],
            [False, True, True, True, True],
        ]
    )
    true_length_nt = torch.tensor([20.0, 0.0, torch.nan])
    status_targets = torch.tensor(
        [ReadStatus.POLY_PRESENT, ReadStatus.ZERO_NT, ReadStatus.FAILURE]
    )
    head = PolyLengthHead(10, hidden_features=7, batch_first=False)

    output = head(features, padding_mask)
    losses = head.loss(output, true_length_nt, status_targets)
    expected_length_loss = functional.smooth_l1_loss(
        torch.log1p(output.length_nt[:2]), torch.log1p(true_length_nt[:2])
    )
    losses["total_loss"].backward()

    assert output.length_nt.shape == (3,)
    assert torch.all(output.length_nt >= 0)
    assert torch.equal(losses["length_loss"], expected_length_loss)
    assert features.grad is not None
    assert torch.count_nonzero(features.grad.transpose(0, 1)[padding_mask]) == 0
    _assert_nonzero_gradients(head)
    decoded = PolyLengthHead.decode_lengths(
        torch.tensor([12.5, 8.0, 4.0]),
        torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]]),
    )
    assert decoded[0].item() == pytest.approx(12.5)
    assert decoded[1].item() == 0.0
    assert torch.isnan(decoded[2])


def test_empty_batch_time_and_all_padding_raise_clear_errors() -> None:
    head = PolyLocationHead(4)
    with pytest.raises(ValueError, match="at least one read"):
        head(torch.randn(0, 3, 4))
    with pytest.raises(ValueError, match="at least one timestep"):
        head(torch.randn(2, 0, 4))
    with pytest.raises(ValueError, match="at least one non-padding"):
        head(torch.randn(2, 3, 4), torch.ones(2, 3, dtype=torch.bool))


@pytest.mark.parametrize("batch_size", [1, 2, 5])
@pytest.mark.parametrize("time_steps", [1, 17, 4096])
def test_heads_support_different_batch_and_time_sizes(
    batch_size: int, time_steps: int
) -> None:
    features = torch.randn(batch_size, time_steps, 4)
    mask = torch.zeros(batch_size, time_steps, dtype=torch.bool)

    location = PolyLocationHead(4, hidden_features=3)(features, mask)
    length = PolyLengthHead(4, hidden_features=3)(features, mask)

    assert location.location_logits.shape == (batch_size, time_steps, 3)
    assert length.length_nt.shape == (batch_size,)


def test_padding_values_do_not_change_valid_outputs_or_loss() -> None:
    torch.manual_seed(13)
    features = torch.randn(2, 100, 5)
    mask = torch.tensor([[False] * 100, [False] + [True] * 99], dtype=torch.bool)
    altered = features.clone()
    altered[mask] = torch.randn_like(altered[mask]) * 1000
    location_head = PolyLocationHead(5, hidden_features=4).eval()
    length_head = PolyLengthHead(5, hidden_features=4).eval()
    status_targets = torch.tensor([ReadStatus.POLY_PRESENT, ReadStatus.ZERO_NT])
    location_targets = torch.tensor([[0] * 40 + [1] * 20 + [0] * 40, [0] + [-100] * 99])
    lengths = torch.tensor([20.0, 0.0])

    first_location = location_head(features, mask)
    second_location = location_head(altered, mask)
    first_length = length_head(features, mask)
    second_length = length_head(altered, mask)
    first_loss = (
        location_head.loss(first_location, location_targets, status_targets)[
            "total_loss"
        ]
        + length_head.loss(first_length, lengths, status_targets)["total_loss"]
    )
    second_loss = (
        location_head.loss(second_location, location_targets, status_targets)[
            "total_loss"
        ]
        + length_head.loss(second_length, lengths, status_targets)["total_loss"]
    )

    valid = ~mask
    assert torch.equal(
        first_location.location_logits[valid],
        second_location.location_logits[valid],
    )
    assert torch.equal(first_location.status_logits, second_location.status_logits)
    assert torch.equal(first_length.length_nt, second_length.length_nt)
    assert torch.equal(first_loss, second_loss)


def test_raw_encoder_coordinate_mapping_and_validation() -> None:
    assert raw_interval_to_encoder_interval(
        0, 12, raw_length_samples=30, feature_steps=5
    ) == (0, 2)
    assert raw_interval_to_encoder_interval(
        18, 30, raw_length_samples=30, feature_steps=5
    ) == (3, 5)
    assert encoder_interval_to_raw_interval(
        3, 5, raw_length_samples=29, feature_steps=5
    ) == (18, 29)
    zero_targets = make_location_targets(
        12,
        12,
        raw_length_samples=30,
        feature_steps=5,
        poly_class=PolyClass.POLY_A,
    )
    assert torch.count_nonzero(zero_targets) == 0
    poly_t_targets = make_location_targets(
        0,
        12,
        raw_length_samples=30,
        feature_steps=5,
        poly_class=PolyClass.POLY_T,
    )
    assert torch.equal(poly_t_targets, torch.tensor([2, 2, 0, 0, 0]))
    with pytest.raises(ValueError, match="0 <= start_sample"):
        raw_interval_to_encoder_interval(13, 12, raw_length_samples=30, feature_steps=5)
    with pytest.raises(ValueError, match="0 <= start_sample"):
        raw_interval_to_encoder_interval(0, 31, raw_length_samples=30, feature_steps=5)


def test_location_decoder_keeps_background_interruptions_separate() -> None:
    head = PolyLocationHead(4, hidden_features=3)
    features = torch.randn(1, 7, 4)
    output = head(features)
    logits = torch.full_like(output.location_logits, -10.0)
    labels = torch.tensor([[1, 1, 0, 1, 1, 1, 0]])
    logits.scatter_(2, labels.unsqueeze(-1), 10.0)
    output = type(output)(logits, output.status_logits, output.padding_mask)
    intervals = decode_location_intervals(
        output,
        torch.tensor([[10.0, 0.0, 0.0]]),
        torch.tensor([42]),
    )
    assert intervals[0] is not None
    assert intervals[0].encoder_start == 3
    assert intervals[0].encoder_end == 6
    assert intervals[0].raw_start_sample == 18
    assert intervals[0].raw_end_sample == 36


def test_v5_adapter_preserves_basecall_scores_and_returns_heads() -> None:
    torch.manual_seed(23)
    basecaller = tiny_v5_basecaller().eval()
    adapter = BonitoV5MultiTaskModel(basecaller, hidden_features=6).eval()
    signal = torch.randn(2, 1, 24)

    with torch.no_grad():
        original_scores = basecaller(signal)
        output = adapter(signal)

    assert basecaller.stride == BONITO_V5_STRIDE
    assert torch.equal(output.basecall_scores, original_scores)
    assert output.encoder_features.shape == (2, 4, 8)
    assert output.location.location_logits.shape == (2, 4, 3)
    assert output.length.length_nt.shape == (2,)


def test_official_v5_config_uses_expected_adapter_tap() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "bonito"
        / "bonito"
        / "models"
        / "configs"
        / "dna_r10.4.1@v5.0.toml"
    )
    basecaller = TransformerModel(toml.load(config_path))
    adapter = BonitoV5MultiTaskModel(basecaller, hidden_features=8)

    assert basecaller.stride == BONITO_V5_STRIDE
    assert adapter.location_head.input_features == 512
    assert adapter.length_head.input_features == 512


def test_v5_adapter_auxiliary_loss_has_one_shared_status_term() -> None:
    torch.manual_seed(29)
    adapter = BonitoV5MultiTaskModel(
        tiny_v5_basecaller(),
        hidden_features=6,
        lambda_location=2.0,
        lambda_length=3.0,
        lambda_status=4.0,
    )
    output = adapter(torch.randn(2, 1, 24))
    location_targets = torch.tensor([[0, 1, 1, 0], [0, 0, 0, 0]])
    true_length_nt = torch.tensor([12.0, 0.0])
    status_targets = torch.tensor([ReadStatus.POLY_PRESENT, ReadStatus.ZERO_NT])

    losses = adapter.auxiliary_loss(
        output, location_targets, true_length_nt, status_targets
    )
    shared_logits = (output.location.status_logits + output.length.status_logits) / 2
    expected_status = functional.cross_entropy(shared_logits, status_targets)
    expected_total = (
        2.0 * losses["location_loss"]
        + 3.0 * losses["length_loss"]
        + 4.0 * expected_status
    )

    assert torch.equal(losses["status_loss"], expected_status)
    assert torch.equal(losses["total_loss"], expected_total)


def test_v5_adapter_freeze_unfreeze_and_state_dict_roundtrip() -> None:
    torch.manual_seed(31)
    adapter = BonitoV5MultiTaskModel(tiny_v5_basecaller(), hidden_features=6)
    signal = torch.randn(2, 1, 24)
    location_targets = torch.tensor([[0, 1, 1, 0], [2, 2, 0, 0]])
    lengths = torch.tensor([12.0, 12.0])
    statuses = torch.tensor([ReadStatus.POLY_PRESENT, ReadStatus.POLY_PRESENT])

    for parameter in adapter.basecaller.parameters():
        parameter.requires_grad_(False)
    frozen_output = adapter(signal)
    adapter.auxiliary_loss(frozen_output, location_targets, lengths, statuses)[
        "total_loss"
    ].backward()
    assert all(parameter.grad is None for parameter in adapter.basecaller.parameters())
    _assert_nonzero_gradients(adapter.location_head)
    _assert_nonzero_gradients(adapter.length_head)

    unfrozen = BonitoV5MultiTaskModel(tiny_v5_basecaller(), hidden_features=6)
    unfrozen_output = unfrozen(signal)
    unfrozen.auxiliary_loss(unfrozen_output, location_targets, lengths, statuses)[
        "total_loss"
    ].backward()
    shared_parameters = [
        parameter
        for name, parameter in unfrozen.basecaller.named_parameters()
        if "crf" not in name
    ]
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in shared_parameters
    )
    assert all(
        parameter.grad is None
        for name, parameter in unfrozen.basecaller.named_parameters()
        if "crf" in name
    )

    unfrozen.eval()
    expected = unfrozen(signal)
    buffer = io.BytesIO()
    torch.save(unfrozen.state_dict(), buffer)
    buffer.seek(0)
    restored = BonitoV5MultiTaskModel(tiny_v5_basecaller(), hidden_features=6)
    restored.load_state_dict(torch.load(buffer, weights_only=True))
    restored.eval()
    actual = restored(signal)
    assert torch.equal(expected.basecall_scores, actual.basecall_scores)
    assert torch.equal(
        expected.location.location_logits, actual.location.location_logits
    )
    assert torch.equal(expected.length.length_nt, actual.length.length_nt)


def test_basecall_loss_delegates_to_bonito_model() -> None:
    basecaller = tiny_v5_basecaller()

    def fake_loss(self: SeqdistModel, value: torch.Tensor) -> dict[str, torch.Tensor]:
        del self
        return {"total_loss": value.sum()}

    basecaller.loss = MethodType(fake_loss, basecaller)
    adapter = BonitoV5MultiTaskModel(basecaller)
    delegated = adapter.basecall_loss(torch.tensor([2.0, 3.0]))
    assert delegated["total_loss"].item() == 5.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_v5_adapter_gpu_forward_backward() -> None:
    device = torch.device("cuda:0")
    adapter = BonitoV5MultiTaskModel(tiny_v5_basecaller(), hidden_features=6).to(device)
    signal = torch.randn(2, 1, 24, device=device)
    output = adapter(signal)
    losses = adapter.auxiliary_loss(
        output,
        torch.tensor([[0, 1, 1, 0], [0, 0, 0, 0]], device=device),
        torch.tensor([12.0, 0.0], device=device),
        torch.tensor([ReadStatus.POLY_PRESENT, ReadStatus.ZERO_NT], device=device),
    )
    losses["total_loss"].backward()
    assert torch.isfinite(losses["total_loss"])


def test_adapter_rejects_non_v5_backbone_and_invalid_signal_masks() -> None:
    with pytest.raises(TypeError, match="Bonito v5 encoder layers"):
        BonitoV5MultiTaskModel(nn.Linear(2, 2))

    adapter = BonitoV5MultiTaskModel(tiny_v5_basecaller())
    with pytest.raises(ValueError, match="at least one read"):
        adapter(torch.randn(0, 1, 24))
    with pytest.raises(ValueError, match="suffix padding"):
        adapter(
            torch.randn(1, 1, 6),
            torch.tensor([[False, True, False, True, True, True]]),
        )


def test_raw_padding_mask_supports_different_read_lengths() -> None:
    adapter = BonitoV5MultiTaskModel(tiny_v5_basecaller())
    signal = torch.randn(2, 1, 24)
    mask = torch.tensor([[False] * 24, [False] * 12 + [True] * 12], dtype=torch.bool)
    output = adapter(signal, mask)
    assert torch.equal(
        output.feature_padding_mask,
        torch.tensor([[False, False, False, False], [False, False, True, True]]),
    )
