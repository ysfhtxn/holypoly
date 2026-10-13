"""Opt-in checks on a prepared manifest and a trained real-model checkpoint."""

from __future__ import annotations

import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
import toml
import torch

from holypoly.models.model import HolypolyModel, load_bonito_v5_model
from holypoly.training.data import (
    ManifestSplit,
    Pod5SignalReader,
    PolyTrainingDataset,
    collate_poly_training_examples,
    load_training_manifest,
)
from holypoly.training.engine import (
    create_optimizer,
    load_checkpoint,
    require_single_cuda_device,
    set_random_seed,
)
from holypoly.training.losses import PolyAuxiliaryLoss, PolyLossWeights


@pytest.mark.integration
def test_prepared_manifest_signal_backward_and_checkpoint() -> None:
    """Read explicit external inputs on one GPU without modifying any input."""
    values = {
        name: os.environ.get(name)
        for name in (
            "HOLYPOLY_M3_MANIFEST",
            "HOLYPOLY_BONITO_MODEL",
            "HOLYPOLY_M3_CHECKPOINT",
            "HOLYPOLY_DATA_ROOT",
            "HOLYPOLY_WORK_ROOT",
        )
    }
    if not all(values.values()):
        pytest.skip("prepared-pipeline external paths are not configured")
    if not torch.cuda.is_available():
        pytest.skip("prepared-pipeline CUDA is unavailable")
    assert torch.cuda.device_count() == 1, "select exactly one physical GPU"
    device = torch.device("cuda:0")
    require_single_cuda_device(device)
    data_root = Path(str(values["HOLYPOLY_DATA_ROOT"])).resolve()
    work_root = Path(str(values["HOLYPOLY_WORK_ROOT"])).resolve()
    manifest = Path(str(values["HOLYPOLY_M3_MANIFEST"])).resolve()
    checkpoint = Path(str(values["HOLYPOLY_M3_CHECKPOINT"])).resolve()
    model_dir = Path(str(values["HOLYPOLY_BONITO_MODEL"])).resolve()
    assert not work_root.is_relative_to(data_root)
    assert manifest.is_relative_to(work_root)
    assert checkpoint.is_relative_to(work_root)
    assert not work_root.is_relative_to(Path(__file__).resolve().parents[3])
    started = datetime.now(UTC).isoformat()
    set_random_seed(20260923)
    records = load_training_manifest(manifest)
    ids_by_split = {
        split: {record.read_id for record in records if record.split == split}
        for split in ManifestSplit
    }
    assert all(ids_by_split.values())
    assert len(set.union(*ids_by_split.values())) == sum(
        map(len, ids_by_split.values())
    )
    chunk_size = records[0].input_length_samples
    assert all(record.input_length_samples == chunk_size for record in records)
    inputs = {manifest, checkpoint, model_dir / "config.toml"}
    inputs.update(model_dir.glob("weights_*.tar"))
    inputs.update(record.pod5_path for record in records)
    before = {path: (path.stat().st_size, path.stat().st_mtime_ns) for path in inputs}
    config = toml.load(model_dir / "config.toml")
    reader = Pod5SignalReader(
        signal_shift=float(config["standardisation"]["mean"]),
        signal_scale=float(config["standardisation"]["stdev"]),
    )
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = HolypolyModel(
        load_bonito_v5_model(model_dir, device=str(device)), hidden_features=128
    ).to(device)
    original_basecaller = {
        name: value.detach().cpu().clone()
        for name, value in model.basecaller.state_dict().items()
    }
    optimizer = create_optimizer(model, learning_rate=1e-3, weight_decay=0.0)
    restored = load_checkpoint(
        checkpoint, model=model, optimizer=optimizer, device=device
    )
    assert restored.epoch >= 1 and restored.global_step > 0
    assert restored.random_seed == 20260923
    assert (
        optimizer.state_dict()["param_groups"]
        == payload["optimizer_state_dict"]["param_groups"]
    )
    assert payload["manifest_schema_version"] == records[0].manifest_version
    assert all(
        torch.equal(value.cpu(), original_basecaller[name])
        for name, value in model.basecaller.state_dict().items()
    )
    criterion = PolyAuxiliaryLoss(PolyLossWeights())
    model.eval()
    checks = []
    for split in ManifestSplit:
        dataset = PolyTrainingDataset(
            records, reader, split=split, chunk_size_samples=chunk_size
        )
        indices = [0]
        first_mode = dataset.records[0].supervision_mode
        alternative = next(
            (
                index
                for index, record in enumerate(dataset.records)
                if record.supervision_mode != first_mode
            ),
            None,
        )
        if alternative is not None:
            indices.append(alternative)
        batch = collate_poly_training_examples([dataset[index] for index in indices])
        assert torch.isfinite(batch.signal).all()
        signal = batch.signal.to(device)
        mask = batch.raw_padding_mask.to(device)
        with torch.no_grad():
            expected_basecall = model.basecaller(signal)
            expected = model(signal, mask)
            repeated = model(signal, mask)
        assert torch.allclose(
            expected.basecall_scores, expected_basecall, rtol=0, atol=1e-6
        )
        assert torch.equal(expected.length.length_nt, repeated.length.length_nt)
        assert (
            expected.location.location_logits.shape[:2] == batch.location_targets.shape
        )
        model.zero_grad(set_to_none=True)
        output = model(signal, mask)
        losses = criterion(
            output,
            batch.location_targets.to(device),
            batch.true_length_nt.to(device),
            batch.status_targets.to(device),
        )
        assert torch.isfinite(losses["total_loss"])
        losses["total_loss"].backward()
        assert all(
            parameter.grad is None for parameter in model.basecaller.parameters()
        )
        assert torch.count_nonzero(model.length_head.length_regressor.weight.grad) > 0
        if losses["location_examples"].item() == 0:
            assert losses["location_loss"].item() == 0
        if losses["status_examples"].item() == 0:
            assert losses["status_loss"].item() == 0
        checks.append(
            {
                "split": split.value,
                "supervision_modes": [
                    dataset.records[index].supervision_mode.value for index in indices
                ],
                "total_loss": float(losses["total_loss"].detach()),
                "location_examples": int(losses["location_examples"]),
                "status_examples": int(losses["status_examples"]),
            }
        )
    # Export raw regression and status separately; weak nominal targets are not truth.
    predictions = []
    test_dataset = PolyTrainingDataset(
        records, reader, split=ManifestSplit.TEST, chunk_size_samples=chunk_size
    )
    with torch.no_grad():
        for index, record in enumerate(test_dataset.records):
            batch = collate_poly_training_examples([test_dataset[index]])
            output = model(batch.signal.to(device), batch.raw_padding_mask.to(device))
            assert torch.isfinite(output.length.length_nt).all()
            predictions.append(
                {
                    "read_id": record.read_id,
                    "nominal_length_nt": record.nominal_length_nt,
                    "regression_length_nt": float(output.length.length_nt[0]),
                    "decoded_length_nt": float(output.length.predicted_length_nt[0]),
                    "status_class": int(output.length.status_logits.argmax(-1)[0]),
                    "supervision_mode": record.supervision_mode.value,
                    "label_source": record.label_source,
                }
            )
    assert before == {
        path: (path.stat().st_size, path.stat().st_mtime_ns) for path in inputs
    }
    output_dir = (work_root / "results" / checkpoint.parent.name).resolve()
    assert output_dir.is_relative_to(work_root) and not output_dir.is_relative_to(
        data_root
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "test_predictions.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(predictions[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(predictions)
    report = {
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "manifest": str(manifest),
        "checkpoint": str(checkpoint),
        "epoch": restored.epoch,
        "global_step": restored.global_step,
        "split_counts": {split.value: len(ids) for split, ids in ids_by_split.items()},
        "checks": checks,
        "test_prediction_rows": len(predictions),
        "basecaller_weights_unchanged": True,
        "basecall_output_regression_passed": True,
        "frozen_parameters_without_gradients": True,
        "input_files_unchanged": True,
        "checkpoint_optimizer_restored": True,
        "inference_repeat_equal": True,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "manifest_schema_version": payload["manifest_schema_version"],
        "git_commit": payload["git_commit"],
        "bonito_commit": payload["bonito_commit"],
        "random_seed": restored.random_seed,
        "loss_config": payload["loss_config"],
        "limitations": (
            "Small pipeline smoke test with nominal weak labels; "
            "no formal accuracy claims."
        ),
    }
    (output_dir / "pipeline_checks.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
