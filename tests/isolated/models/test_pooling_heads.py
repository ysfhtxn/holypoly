"""Controlled pooling comparisons using synthetic encoder features only."""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as functional
from torch import nn

from holypoly.models.model import HolypolyModel
from holypoly.models.poly_heads import PolyLengthHead
from holypoly.models.pooling_heads import SumLengthHead
from tests.isolated.helpers import tiny_v5_basecaller


def test_pooling_modes_have_identical_initialization_and_parameter_counts() -> None:
    """A fixed seed produces exactly the same weights in the two full models."""
    models = []
    for mode in ("mean", "sum"):
        torch.manual_seed(17)
        models.append(
            HolypolyModel(
                tiny_v5_basecaller(), hidden_features=6, length_head_mode=mode
            )
        )
    mean, summed = models
    assert type(mean.length_head) is PolyLengthHead
    assert type(summed.length_head) is SumLengthHead
    assert mean.parameter_counts() == summed.parameter_counts()
    assert mean.state_dict().keys() == summed.state_dict().keys()
    for key, tensor in mean.state_dict().items():
        torch.testing.assert_close(tensor, summed.state_dict()[key], rtol=0, atol=0)


@pytest.mark.parametrize("batch_first", [True, False])
def test_sum_uses_valid_duration_and_excludes_padding_from_values_and_gradients(
    batch_first: bool,
) -> None:
    """Repeated features must scale the sum, while padding never contributes."""
    mean = PolyLengthHead(2, 2, batch_first=batch_first).eval()
    summed = SumLengthHead(2, 2, batch_first=batch_first).eval()
    for head in (mean, summed):
        # Isolate pooling from nonlinear projection for this algebraic check.
        head.projection = nn.Identity()
        with torch.no_grad():
            head.length_regressor.weight.fill_(1)
            head.length_regressor.bias.fill_(0.5)
            head.status_classifier.weight.fill_(1)
            head.status_classifier.bias.zero_()
    features = torch.tensor(
        [[[1.0, 2.0]] * 4, [[1.0, 2.0]] * 2 + [[999.0, 999.0]] * 2],
        requires_grad=True,
    )
    mask = torch.tensor([[False] * 4, [False, False, True, True]])
    inputs = features if batch_first else features.transpose(0, 1)
    mean_output = mean(inputs, mask)
    output = summed(inputs, mask)
    torch.testing.assert_close(
        mean_output.length_nt, functional.softplus(torch.tensor([3.5, 3.5]))
    )
    torch.testing.assert_close(
        output.length_nt, functional.softplus(torch.tensor([12.5, 6.5]))
    )
    torch.testing.assert_close(output.status_logits[:, 0], torch.tensor([12.0, 6.0]))
    extended = torch.cat((features.detach(), torch.full((2, 3, 2), -999.0)), dim=1)
    extended_mask = torch.cat((mask, torch.ones(2, 3, dtype=torch.bool)), dim=1)
    padded = summed(
        extended if batch_first else extended.transpose(0, 1), extended_mask
    )
    torch.testing.assert_close(output.length_nt, padded.length_nt, rtol=0, atol=0)
    torch.testing.assert_close(
        output.status_logits, padded.status_logits, rtol=0, atol=0
    )
    output.length_nt.sum().backward()
    assert features.grad is not None
    assert torch.count_nonzero(features.grad[mask]) == 0
    assert torch.all(features.grad[~mask] > 0)


def test_sum_without_padding_mask_and_rejection_of_empty_reads() -> None:
    """The sum head shares the original input validation contract."""
    head = SumLengthHead(4, 6).eval()
    features = torch.randn(2, 5, 4)
    full = head(features)
    explicit = head(features, torch.zeros(2, 5, dtype=torch.bool))
    torch.testing.assert_close(full.length_nt, explicit.length_nt, rtol=0, atol=0)
    with pytest.raises(ValueError, match="at least one non-padding"):
        head(features, torch.ones(2, 5, dtype=torch.bool))
