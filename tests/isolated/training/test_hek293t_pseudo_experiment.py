"""Synthetic sampling, provenance, weighting and evaluation safeguards."""

import importlib.util
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from holypoly.models.joint_model import JointHolypolyModel
from holypoly.training.data import PolyTrainingExample
from holypoly.training.losses import PolyAuxiliaryLoss, PolyLossWeights
from tests.isolated.training.test_reference_joint import joint_basecaller

SPEC = importlib.util.spec_from_file_location(
    "pseudo_experiment",
    Path(__file__).resolve().parents[3]
    / "diagnostics_hek293t_pseudo_label/run_experiment.py",
)
EXPERIMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPERIMENT)


def candidate(read_id="synthetic", length="1.2", status="candidate_predicted"):
    """Return metadata for a synthetic 75-sample candidate."""
    return {
        "read_id": read_id,
        "predicted_length_nt": length,
        "status": status,
        "start_sample": "25",
        "end_sample": "100",
        "raw_signal_length": "150",
        "pod5_path": "/synthetic/reads.pod5",
        "poly_type": "polyA",
    }


def test_sampling_filters_failures_boundary_nonfinite_and_historical_holdout():
    rows = [candidate(str(i)) for i in range(300)]
    rows += [
        candidate("five", "5"),
        candidate("nan", "nan"),
        candidate("negative", "-1"),
        candidate("fail", "0", "input_failure"),
    ]
    excluded = {str(i) for i in range(50)}
    chosen = EXPERIMENT.select_candidates(rows, excluded, 17)
    assert len(chosen) == 200
    assert not {r["read_id"] for r in chosen} & excluded
    assert all(float(r["predicted_length_nt"]) == 1.2 for r in chosen)
    assert chosen == EXPERIMENT.select_candidates(list(reversed(rows)), excluded, 17)
    assert chosen != EXPERIMENT.select_candidates(rows, excluded, 18)
    with pytest.raises(ValueError, match="duplicate"):
        EXPERIMENT.select_candidates([rows[0], rows[0]], set(), 17)
    assert EXPERIMENT.select_candidates([], set(), 17) == []


def test_original_subset_never_uses_heldout_and_keeps_fields():
    rows = [
        {
            "read_id": f"{length}-{i}",
            "split": "train",
            "nominal_length_nt": str(length),
            "original": "untouched",
        }
        for length in [0, 20, 40, 60, 80, 100, 150]
        for i in range(210)
    ]
    rows.append({"read_id": "heldout", "split": "validation"})
    chosen = EXPERIMENT.select_original(rows, 17)
    assert len(chosen) == 1400
    assert all(r["split"] == "train" and r["original"] == "untouched" for r in chosen)
    assert chosen == EXPERIMENT.select_original(list(reversed(rows)), 17)


def test_experimental_record_preserves_unknowns_and_ignores_location():
    row = candidate()
    original = row.copy()
    record = EXPERIMENT.pseudo_record(row)
    assert row == original
    assert record.label_source == "model_pseudo_label"
    assert record.training_length_nt == 0 and record.label_status.value == "zero_nt"
    assert math.isnan(record.nominal_length_nt)
    assert math.isnan(record.observed_sequence_length_nt)
    assert record.poly_is_complete is None and record.poly_start_sample is None
    record = replace(record, model_input_samples=120)
    batch = EXPERIMENT.collate([PolyTrainingExample(torch.randn(120), record)])
    assert (batch.location_targets == -100).all()
    assert batch.status_targets.tolist() == [1]


def test_point_one_weight_scales_actual_length_and_status_gradients():
    torch.manual_seed(17)
    model = JointHolypolyModel(
        joint_basecaller(), hidden_features=6, length_head_mode="sum"
    )
    record = replace(EXPERIMENT.pseudo_record(candidate()), model_input_samples=120)
    batch = EXPERIMENT.collate([PolyTrainingExample(torch.randn(120), record)])
    criterion = PolyAuxiliaryLoss(
        PolyLossWeights(lambda_location=0, length_loss="huber")
    )
    output = model(batch.signal, batch.raw_padding_mask)
    loss = criterion(
        output, batch.location_targets, batch.true_length_nt, batch.status_targets
    )
    parameters = [
        model.length_head.length_regressor.weight,
        model.length_head.status_classifier.weight,
    ]
    gradients = torch.autograd.grad(loss["total_loss"], parameters, retain_graph=True)
    weighted = torch.autograd.grad(
        EXPERIMENT.pseudo_scale(1, 1) * loss["total_loss"], parameters
    )
    for full, low in zip(gradients, weighted, strict=True):
        assert full.abs().sum() > 0
        torch.testing.assert_close(low, full * 0.1)
    assert EXPERIMENT.pseudo_scale(2, 4) == 0.05


def test_metrics_use_ungated_lengths_and_retain_failures():
    rows = [
        {
            "prediction_status": "ok",
            "nominal_length_nt": 20,
            "regression_length_nt": 15,
            "shared_status": "zero_nt",
        },
        {"prediction_status": "input_failure"},
    ]
    result = EXPERIMENT.metrics(rows)
    assert result["failed"] == 1
    group = result["by_nominal_length_nt"]["20"]
    assert group["shared_zero_nt_fraction"] == 1
    assert group["nominal_bias_nt"] == -5
    assert group["regression_lt5_fraction"] == 0
