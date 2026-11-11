"""Explicit integration for the original 16-read Milestone 3 dataset only."""

from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import toml
import torch
from torch.utils.data import DataLoader

from holypoly.models.model import HolypolyModel, load_bonito_v5_model
from holypoly.training.data import (
    ManifestSplit,
    Pod5SignalReader,
    PolyTrainingDataset,
    collate_poly_training_examples,
    load_training_manifest,
)
from holypoly.training.engine import (
    CheckpointContext,
    create_optimizer,
    load_checkpoint,
    save_checkpoint,
    train_one_epoch,
    validate_one_epoch,
)
from holypoly.training.losses import PolyAuxiliaryLoss, PolyLossWeights


@pytest.mark.integration
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_real_2p5per_pipeline_smoke() -> None:
    manifest_value = os.environ.get("HOLYPOLY_M3_MANIFEST")
    model_value = os.environ.get("HOLYPOLY_BONITO_MODEL")
    checkpoint_value = os.environ.get("HOLYPOLY_M3_CHECKPOINT")
    data_value = os.environ.get("HOLYPOLY_DATA_ROOT")
    work_value = os.environ.get("HOLYPOLY_WORK_ROOT")
    if not all((manifest_value, model_value, checkpoint_value, data_value, work_value)):
        pytest.skip("Milestone 3 integration paths are not configured")

    data_root = Path(str(data_value)).resolve()
    work_root = Path(str(work_value)).resolve()
    manifest_path = Path(str(manifest_value)).resolve()
    model_path = Path(str(model_value)).resolve()
    checkpoint_path = Path(str(checkpoint_value)).resolve()
    assert not work_root.is_relative_to(data_root)
    assert not work_root.is_relative_to(Path(__file__).resolve().parents[3])
    assert manifest_path.is_relative_to(work_root)
    assert checkpoint_path.is_relative_to(work_root)
    assert not checkpoint_path.exists(), "legacy smoke requires a new output checkpoint"
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
        data_root=data_root,
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
