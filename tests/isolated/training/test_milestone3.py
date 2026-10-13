from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import toml
import torch
from torch.utils.data import DataLoader

from holypoly.models.model import HolypolyModel, load_bonito_v5_model
from holypoly.models.poly_heads import PolyClass, ReadStatus
from holypoly.training.data import (
    ManifestSplit,
    Pod5SignalReader,
    PolyTrainingBatch,
    PolyTrainingDataset,
    collate_poly_training_examples,
    load_training_manifest,
)
from holypoly.training.engine import (
    CheckpointContext,
    _MetricAccumulator,
    append_loss_log,
    create_optimizer,
    load_checkpoint,
    save_checkpoint,
    set_random_seed,
    train_one_epoch,
    validate_one_epoch,
)
from holypoly.training.losses import PolyAuxiliaryLoss, PolyLossWeights
from tests.isolated.helpers import tiny_v5_basecaller


def _synthetic_batch() -> PolyTrainingBatch:
    raw_samples = 120
    feature_steps = 20
    signal = torch.zeros(6, 1, raw_samples)
    signal[0, 0, 24:60] = 4.0
    signal[1, 0, 60:108] = 6.0
    signal[2, 0, 12:48] = -4.0
    signal[3, 0, 72:114] = -6.0
    signal[4, 0] = torch.linspace(-0.1, 0.1, raw_samples)
    signal[5, 0] = 9.0

    location_targets = torch.zeros(6, feature_steps, dtype=torch.long)
    location_targets[0, 4:10] = int(PolyClass.POLY_A)
    location_targets[1, 10:18] = int(PolyClass.POLY_A)
    location_targets[2, 2:8] = int(PolyClass.POLY_T)
    location_targets[3, 12:19] = int(PolyClass.POLY_T)
    location_targets[5] = -100
    return PolyTrainingBatch(
        signal=signal,
        raw_padding_mask=torch.zeros(6, raw_samples, dtype=torch.bool),
        raw_lengths_samples=torch.full((6,), raw_samples, dtype=torch.long),
        location_targets=location_targets,
        true_length_nt=torch.tensor([12.0, 24.0, 15.0, 30.0, 0.0, math.nan]),
        status_targets=torch.tensor(
            [
                ReadStatus.POLY_PRESENT,
                ReadStatus.POLY_PRESENT,
                ReadStatus.POLY_PRESENT,
                ReadStatus.POLY_PRESENT,
                ReadStatus.ZERO_NT,
                ReadStatus.FAILURE,
            ]
        ),
        read_ids=(
            "poly-a-short",
            "poly-a-long",
            "poly-t-short",
            "poly-t-long",
            "zero",
            "failure",
        ),
    )


def _losses_on_batch(
    model: HolypolyModel,
    criterion: PolyAuxiliaryLoss,
    batch: PolyTrainingBatch,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], object]:
    output = model(batch.signal.to(device), batch.raw_padding_mask.to(device))
    losses = criterion(
        output,
        batch.location_targets.to(device),
        batch.true_length_nt.to(device),
        batch.status_targets.to(device),
    )
    return losses, output


def test_holypoly_model_freezes_bonito_and_optimizer_uses_only_heads() -> None:
    torch.manual_seed(20260923)
    basecaller = tiny_v5_basecaller()
    signal = torch.randn(2, 1, 120)
    original_scores = basecaller(signal)
    model = HolypolyModel(basecaller, hidden_features=16)
    optimizer = create_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    output = model(signal)

    assert output.basecall_scores.shape == original_scores.shape
    assert torch.allclose(
        output.basecall_scores,
        original_scores,
        rtol=0.0,
        atol=1e-6,
    )
    assert all(
        not parameter.requires_grad for parameter in model.basecaller.parameters()
    )
    assert all(
        parameter.requires_grad for parameter in model.location_head.parameters()
    )
    assert all(parameter.requires_grad for parameter in model.length_head.parameters())
    optimized = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    assert optimized == {
        id(parameter) for parameter in model.parameters() if parameter.requires_grad
    }
    counts = model.parameter_counts()
    assert counts["frozen"] > 0
    assert counts["trainable"] > 0
    assert counts["total"] == counts["frozen"] + counts["trainable"]


def test_independent_loss_masks_failure_padding_and_invalid_lengths() -> None:
    model = HolypolyModel(tiny_v5_basecaller(), hidden_features=16)
    criterion = PolyAuxiliaryLoss(PolyLossWeights(2.0, 3.0, 4.0))
    batch = _synthetic_batch()
    losses, _ = _losses_on_batch(model, criterion, batch, torch.device("cpu"))

    assert losses["location_examples"].item() == 5 * 20
    assert losses["length_examples"].item() == 5
    assert torch.equal(
        losses["total_loss"],
        losses["weighted_location_loss"]
        + losses["weighted_length_loss"]
        + losses["weighted_status_loss"],
    )
    invalid_lengths = batch.true_length_nt.clone()
    invalid_lengths[4] = torch.nan
    with pytest.raises(ValueError, match="non-failed training lengths must be finite"):
        output = model(batch.signal, batch.raw_padding_mask)
        criterion(
            output,
            batch.location_targets,
            invalid_lengths,
            batch.status_targets,
        )


def test_length_only_batch_has_finite_loss_and_only_length_gradients() -> None:
    model = HolypolyModel(tiny_v5_basecaller(), hidden_features=16)
    criterion = PolyAuxiliaryLoss(PolyLossWeights())
    batch = _synthetic_batch()
    batch.location_targets.fill_(-100)
    batch.status_targets.fill_(-100)
    batch.true_length_nt[4:] = 20  # All six now have positive weak length targets.
    losses, output = _losses_on_batch(model, criterion, batch, torch.device("cpu"))
    assert losses["location_examples"].item() == 0
    assert losses["status_examples"].item() == 0
    assert losses["length_examples"].item() == 6
    assert losses["location_loss"].item() == losses["status_loss"].item() == 0
    assert torch.isfinite(losses["total_loss"])
    adapter_losses = model.multitask.auxiliary_loss(
        output, batch.location_targets, batch.true_length_nt, batch.status_targets
    )
    assert torch.equal(adapter_losses["total_loss"], losses["total_loss"])
    losses["total_loss"].backward()
    assert all(parameter.grad is None for parameter in model.basecaller.parameters())
    assert all(
        parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
        for parameter in model.location_head.parameters()
    )
    assert torch.count_nonzero(model.length_head.length_regressor.weight.grad) > 0
    accumulator = _MetricAccumulator()
    accumulator.update(output, batch, losses)
    assert math.isnan(accumulator.result(0).status_accuracy)
    # Adding one supervised status must use one, rather than six, in its denominator.
    batch.status_targets[0] = int(
        ((output.location.status_logits + output.length.status_logits) / 2).argmax(-1)[
            0
        ]
    )
    accumulator = _MetricAccumulator()
    accumulator.update(output, batch, losses)
    assert accumulator.result(0).status_accuracy == 1.0
    batch.location_targets[0].fill_(0)
    mixed_losses = criterion(
        output, batch.location_targets, batch.true_length_nt, batch.status_targets
    )
    assert mixed_losses["status_examples"].item() == 1
    assert mixed_losses["location_examples"].item() == 20
    assert torch.equal(
        mixed_losses["location_loss"],
        torch.nn.functional.cross_entropy(
            output.location.location_logits[0], batch.location_targets[0]
        ),
    )
    shared_logits = (output.location.status_logits + output.length.status_logits) / 2
    assert torch.equal(
        mixed_losses["status_loss"],
        torch.nn.functional.cross_entropy(shared_logits[:1], batch.status_targets[:1]),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_synthetic_overfit_and_checkpoint_roundtrip(tmp_path: Path) -> None:
    device = torch.device("cuda:0")
    set_random_seed(20260923)
    batch = _synthetic_batch()
    model = HolypolyModel(tiny_v5_basecaller(), hidden_features=32).to(device)
    criterion = PolyAuxiliaryLoss(PolyLossWeights())
    optimizer = create_optimizer(model, learning_rate=2e-2, weight_decay=0.0)
    model.eval()
    with torch.no_grad():
        initial_losses, _ = _losses_on_batch(model, criterion, batch, device)
    initial = {
        name: float(initial_losses[name])
        for name in ("total_loss", "location_loss", "length_loss")
    }

    result = train_one_epoch(
        model,
        [batch] * 500,
        criterion,
        optimizer,
        device=device,
        gradient_clip_norm=2.0,
    )
    model.eval()
    with torch.no_grad():
        final_losses, expected_output = _losses_on_batch(
            model, criterion, batch, device
        )
    final = {
        name: float(final_losses[name])
        for name in ("total_loss", "location_loss", "length_loss")
    }

    assert torch.isfinite(final_losses["total_loss"])
    assert final["location_loss"] < initial["location_loss"]
    assert final["length_loss"] < initial["length_loss"]
    assert final["total_loss"] <= initial["total_loss"] * 0.1
    shared_status = (
        expected_output.location.status_logits + expected_output.length.status_logits
    ) / 2
    assert shared_status.argmax(dim=-1)[4].item() == int(ReadStatus.ZERO_NT)
    assert expected_output.length.predicted_length_nt[4].item() == 0.0
    assert final_losses["length_examples"].item() == 5
    assert all(parameter.grad is None for parameter in model.basecaller.parameters())

    checkpoint_path = tmp_path / "milestone3.pt"
    context = CheckpointContext(
        random_seed=20260923,
        bonito_commit="b98e6187365b57b249d829df4dae15905da4ad3f",
        model_config={"model": "tiny-v5-test"},
        manifest_schema_version="synthetic-v1",
        loss_weights=criterion.weights,
        git_commit="test-commit",
    )
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        global_step=result.global_step,
        context=context,
    )
    append_loss_log(tmp_path / "losses.jsonl", result, epoch=3)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["bonito_commit"] == context.bonito_commit
    assert payload["manifest_schema_version"] == "synthetic-v1"
    assert payload["loss_config"] == {
        "lambda_location": 1.0,
        "lambda_length": 1.0,
        "lambda_status": 1.0,
    }
    assert payload["python_version"]
    assert payload["pytorch_version"]
    assert payload["cuda_version"]

    restored = HolypolyModel(tiny_v5_basecaller(), hidden_features=32).to(device)
    restored_optimizer = create_optimizer(
        restored, learning_rate=2e-2, weight_decay=0.0
    )
    resume = load_checkpoint(
        checkpoint_path,
        model=restored,
        optimizer=restored_optimizer,
        device=device,
    )
    restored.eval()
    with torch.no_grad():
        _, actual_output = _losses_on_batch(restored, criterion, batch, device)

    assert resume.epoch == 3
    assert resume.global_step == 500
    assert resume.random_seed == 20260923
    assert torch.equal(expected_output.basecall_scores, actual_output.basecall_scores)
    assert torch.equal(
        expected_output.location.location_logits,
        actual_output.location.location_logits,
    )
    assert torch.equal(expected_output.length.length_nt, actual_output.length.length_nt)


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_2p5per_pipeline_smoke() -> None:
    manifest_value = os.environ.get("HOLYPOLY_M3_MANIFEST")
    model_value = os.environ.get("HOLYPOLY_BONITO_MODEL")
    checkpoint_value = os.environ.get("HOLYPOLY_M3_CHECKPOINT")
    if not manifest_value or not model_value or not checkpoint_value:
        pytest.skip("Milestone 3 integration paths are not configured")

    manifest_path = Path(manifest_value)
    model_path = Path(model_value)
    checkpoint_path = Path(checkpoint_value)
    records = load_training_manifest(manifest_path)
    assert len(records) == 16
    assert {record.manifest_version for record in records} == {"milestone3-2p5per-v1"}
    assert all(record.flank_status == "provisional" for record in records)
    assert all(record.nominal_length_nt == 20 for record in records)
    assert all(record.training_length_nt == 20 for record in records)
    assert {record.observed_sequence_length_nt for record in records} <= {
        17.0,
        18.0,
        19.0,
        20.0,
        21.0,
    }

    config = toml.load(model_path / "config.toml")
    reader = Pod5SignalReader(
        signal_shift=float(config["standardisation"]["mean"]),
        signal_scale=float(config["standardisation"]["stdev"]),
    )
    train_dataset = PolyTrainingDataset(
        records, reader, split=ManifestSplit.TRAIN, chunk_size_samples=12000
    )
    validation_dataset = PolyTrainingDataset(
        records, reader, split=ManifestSplit.VALIDATION, chunk_size_samples=12000
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_poly_training_examples,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_poly_training_examples,
    )
    device = torch.device("cuda:0")
    basecaller = load_bonito_v5_model(model_path, device=str(device))
    model = HolypolyModel(basecaller, hidden_features=128).to(device)
    criterion = PolyAuxiliaryLoss(PolyLossWeights())
    optimizer = create_optimizer(model, learning_rate=1e-3, weight_decay=0.0)

    train_result = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device=device,
        gradient_clip_norm=2.0,
    )
    validation_result = validate_one_epoch(
        model,
        validation_loader,
        criterion,
        device=device,
        global_step=train_result.global_step,
    )
    assert train_result.examples == 12
    assert validation_result.examples == 4
    assert math.isfinite(train_result.losses["total_loss"])
    assert math.isfinite(validation_result.losses["total_loss"])
    assert all(parameter.grad is None for parameter in model.basecaller.parameters())
    assert any(
        parameter.grad is not None for parameter in model.location_head.parameters()
    )
    assert any(
        parameter.grad is not None for parameter in model.length_head.parameters()
    )

    context = CheckpointContext(
        random_seed=20260923,
        bonito_commit="b98e6187365b57b249d829df4dae15905da4ad3f",
        model_config=config,
        manifest_schema_version="milestone3-2p5per-v1",
        loss_weights=criterion.weights,
        git_commit="1a895a6f3db36e39c8e994798961888a087a93b1",
    )
    save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        global_step=train_result.global_step,
        context=context,
        data_root=Path("/data/biolab-backup-hdd3/zhaoxy/holypoly_data/fixed_len_data"),
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert payload["manifest_schema_version"] == "milestone3-2p5per-v1"
    assert payload["bonito_commit"] == context.bonito_commit
    assert payload["model_config"]["model"]["package"] == "bonito.transformer"
    resume = load_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        device=device,
    )
    assert resume.epoch == 1
    assert resume.global_step == 12
