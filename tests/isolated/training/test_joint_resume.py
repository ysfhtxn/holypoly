"""Synthetic metadata and checkpoint tests for length batch changes on resume."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from holypoly.datasets import prepare_sequence
from holypoly.models.joint_model import JointHolypolyModel, create_joint_optimizer
from holypoly.training import engine, joint_command
from holypoly.training.engine import (
    CheckpointContext,
    epoch_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    validate_joint_config,
)
from holypoly.training.losses import PolyLossWeights
from tests.isolated.datasets.test_prepare_sequence import sequence_setup
from tests.isolated.training.test_reference_joint import joint_basecaller


def _config(batch_size: int = 8) -> dict[str, Any]:
    """Use representative immutable joint settings with a variable length batch."""
    return {
        "schema": "reference-head-joint-v1",
        "length_batch_size": batch_size,
        "sequence_batch_size": 1,
        "head_learning_rate": 0.001,
        "encoder_learning_rate": 1e-5,
        "lambda_basecall": 0.1,
        "length_manifest_sha256": "a" * 64,
        "sequence_manifest_sha256": "b" * 64,
        "status_supervision": "construct_nominal_weak_supervision",
    }


@pytest.mark.parametrize(
    ("updates", "allowed", "accepted"),
    [
        ({}, False, True),
        ({"length_batch_size": 64}, False, False),
        ({"length_batch_size": 64}, True, True),
        ({"length_batch_size": 0}, True, False),
        ({"length_batch_size": True}, True, False),
        ({"length_batch_size": 64, "sequence_batch_size": 2}, True, False),
        ({"length_batch_size": 64, "head_learning_rate": 1e-4}, True, False),
        ({"lambda_basecall": 0.2}, True, False),
        ({"length_manifest_sha256": "c" * 64}, True, False),
        ({"sequence_manifest_sha256": "c" * 64}, True, False),
        ({"status_supervision": "disabled"}, True, False),
    ],
)
def test_only_length_batch_change_is_allowed(
    updates: dict[str, Any], allowed: bool, accepted: bool
) -> None:
    saved = _config()
    requested = {**saved, **updates}
    if accepted:
        validate_joint_config(saved, requested, allow_batch_size_change=allowed)
    else:
        with pytest.raises(ValueError, match="joint training configuration"):
            validate_joint_config(saved, requested, allow_batch_size_change=allowed)
    assert saved == _config()
    with pytest.raises(ValueError, match="joint training configuration"):
        validate_joint_config(None, requested, allow_batch_size_change=True)


def _history_row(epoch: int, split: str, config: dict[str, Any]) -> dict[str, Any]:
    """Represent a completed epoch with its actual runtime configuration."""
    return {
        "epoch": epoch,
        "split": split,
        "global_step": epoch * 3,
        "joint_config": config,
    }


def _resume_files(tmp_path: Path) -> tuple[Path, Path, dict[str, Any]]:
    """Create a minimal consistent completed run using synthetic metadata only."""
    log = tmp_path / "train.jsonl"
    log.write_text(
        "".join(
            json.dumps(_history_row(1, split, _config())) + "\n"
            for split in ("train", "validation")
        )
    )
    result = tmp_path / "results"
    result.mkdir()
    (result / "basecalling_validation_baseline.json").write_text("{}\n")
    (result / "run_summary.json").write_text(json.dumps({"joint_config": _config()}))
    payload = {"epoch": 1, "global_step": 3, "joint_config": _config()}
    return log, result, payload


def test_mixed_batch_history_supports_repeated_and_interrupted_resumes(
    tmp_path: Path,
) -> None:
    log, result, payload = _resume_files(tmp_path)
    before = log.read_bytes()
    arguments = dict(
        log=log,
        result_dir=result,
        joint_config=_config(64),
        target_epochs=3,
        recorded_utc="2026-01-01T00:00:00+00:00",
    )
    with pytest.raises(ValueError, match="length_batch_size"):
        joint_command._validate_resume_metadata(
            payload, **arguments, allow_batch_size_change=False
        )
    changes = joint_command._validate_resume_metadata(
        payload, **arguments, allow_batch_size_change=True
    )
    assert changes == [
        {
            "after_epoch": 1,
            "global_step": 3,
            "from_batch_size": 8,
            "to_batch_size": 64,
            "recorded_utc": arguments["recorded_utc"],
        }
    ]
    assert log.read_bytes() == before
    assert "length_batch_size_changes" not in payload
    with log.open("a") as stream:
        for split in ("train", "validation"):
            stream.write(json.dumps(_history_row(2, split, _config(64))) + "\n")
    payload.update(
        epoch=2,
        global_step=6,
        joint_config=_config(64),
        length_batch_size_changes=changes,
    )
    # Summary still says batch=8, as when interrupted after a saved resumed epoch.
    assert (
        joint_command._validate_resume_metadata(
            payload, **arguments, allow_batch_size_change=False
        )
        == changes
    )
    arguments["joint_config"] = _config(16)
    again = joint_command._validate_resume_metadata(
        payload, **arguments, allow_batch_size_change=True
    )
    assert again[:-1] == changes
    assert again[-1]["from_batch_size"] == 64
    assert again[-1]["to_batch_size"] == 16
    assert again[-1]["after_epoch"] == 2
    assert log.read_bytes().startswith(before)


@pytest.mark.parametrize(
    "corruption", ["step", "missing_epoch", "pair", "immutable", "summary", "baseline"]
)
def test_batch_override_still_rejects_inconsistent_run_history(
    tmp_path: Path, corruption: str
) -> None:
    log, result, payload = _resume_files(tmp_path)
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    if corruption == "step":
        rows[-1]["global_step"] += 1
    elif corruption == "missing_epoch":
        rows.pop(0)
    elif corruption == "pair":
        rows[0]["joint_config"]["length_batch_size"] = 4
    elif corruption == "immutable":
        for row in rows:
            row["joint_config"]["head_learning_rate"] = 1e-4
    elif corruption == "summary":
        (result / "run_summary.json").write_text(
            json.dumps({"joint_config": {**_config(), "lambda_basecall": 0.2}})
        )
    else:
        (result / "basecalling_validation_baseline.json").unlink()
    log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        joint_command._validate_resume_metadata(
            payload,
            log=log,
            result_dir=result,
            joint_config=_config(64),
            target_epochs=3,
            recorded_utc="synthetic",
            allow_batch_size_change=True,
        )


def test_checkpoint_restores_optimizer_and_records_batch_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real serialization restores synthetic Adam state and preserves old snapshots."""
    for name, function in (
        ("current_device", lambda: 0),
        ("get_device_name", lambda _: "synthetic"),
        ("get_rng_state_all", lambda: []),
        ("set_rng_state_all", lambda _: None),
    ):
        monkeypatch.setattr(torch.cuda, name, function)
    monkeypatch.setattr(engine, "require_single_cuda_device", lambda _: None)
    model = JointHolypolyModel(
        joint_basecaller(), hidden_features=6, length_head_mode="sum"
    )
    optimizer = create_joint_optimizer(
        model, head_learning_rate=0.001, encoder_learning_rate=1e-5, weight_decay=0
    )
    # Populate Adam moments without native CRF, real data, or a training loop.
    for group in optimizer.param_groups:
        for parameter in group["params"]:
            parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    optimizer_before = deepcopy(optimizer.state_dict())
    weights = PolyLossWeights(length_loss="huber")
    context = CheckpointContext(
        17, "synthetic", {}, "synthetic", weights, "synthetic", joint_config=_config()
    )
    path = tmp_path / "heads.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=1,
        global_step=3,
        context=context,
        archive_epoch=True,
    )
    previous = epoch_checkpoint_path(path, 1).read_bytes()
    restored_model = JointHolypolyModel(
        joint_basecaller(), hidden_features=6, length_head_mode="sum"
    )
    restored_optimizer = create_joint_optimizer(
        restored_model,
        head_learning_rate=0.001,
        encoder_learning_rate=1e-5,
        weight_decay=0,
    )
    restore_args = dict(
        model=restored_model,
        optimizer=restored_optimizer,
        device=torch.device("cpu"),
        loss_weights=weights,
        joint_config=_config(64),
    )
    with pytest.raises(ValueError, match="length_batch_size"):
        load_checkpoint(path, **restore_args)
    restored = load_checkpoint(path, **restore_args, allow_batch_size_change=True)
    assert (restored.epoch, restored.global_step) == (1, 3)
    with pytest.raises(ValueError, match="loss configuration differs"):
        load_checkpoint(
            path,
            **{**restore_args, "loss_weights": replace(weights, lambda_status=2)},
            allow_batch_size_change=True,
        )
    with pytest.raises(ValueError, match="data order seed differs"):
        load_checkpoint(
            path,
            **restore_args,
            data_order_seed=123,
            allow_batch_size_change=True,
        )
    for key, value in model.state_dict().items():
        assert torch.equal(value, restored_model.state_dict()[key])
    assert (
        restored_optimizer.state_dict()["param_groups"]
        == optimizer_before["param_groups"]
    )
    for key, values in optimizer_before["state"].items():
        for name, value in values.items():
            assert torch.equal(
                value, restored_optimizer.state_dict()["state"][key][name]
            )
    changes = ({"after_epoch": 1, "from_batch_size": 8, "to_batch_size": 64},)
    save_checkpoint(
        path,
        model=restored_model,
        optimizer=restored_optimizer,
        epoch=2,
        global_step=4,
        context=replace(
            context,
            joint_config=_config(64),
            length_batch_size_changes=changes,
            input_pipeline={"prefetch_batches": 2, "pod5_reader_cache_size": 8},
        ),
        archive_epoch=True,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["joint_config"]["length_batch_size"] == 64
    assert payload["length_batch_size_changes"] == list(changes)
    assert payload["input_pipeline"] == {
        "prefetch_batches": 2,
        "pod5_reader_cache_size": 8,
    }
    assert epoch_checkpoint_path(path, 1).read_bytes() == previous
    assert load_checkpoint(path, **restore_args).epoch == 2


def test_resume_dry_run_checks_checkpoint_without_signals_or_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the actual resume CLI against a synthetic metadata-only checkpoint."""
    args, _, _, work, manifest = sequence_setup(tmp_path)
    prepare_sequence.prepare(args)
    model_dir = work / "model"
    model_dir.mkdir()
    (model_dir / "config.toml").write_text(
        "[model.seqdist]\nstate_len=1\n[model.encoder.transformer_encoder]\ndepth=2\n"
    )
    arguments = [
        "--config",
        str(args.config),
        "--manifest",
        str(manifest),
        "--sequence-manifest",
        str(work / args.output_dir / "sequence_manifest.tsv"),
        "--model-dir",
        str(model_dir),
        "--chunk-size-samples",
        "252",
        "--run-id",
        "synthetic-resume",
        "--epochs",
        "3",
        "--batch-size",
        "8",
        "--dry-run",
    ]
    summary = joint_command.run(joint_command.build_parser().parse_args(arguments))
    log = work / "logs/synthetic-resume/train.jsonl"
    log.parent.mkdir(parents=True)
    config = summary["joint_config"]
    log.write_text(
        "".join(
            json.dumps(_history_row(1, split, config)) + "\n"
            for split in ("train", "validation")
        )
    )
    result = Path(summary["result_dir"])
    result.mkdir(parents=True)
    (result / "run_summary.json").write_text(json.dumps(summary))
    (result / "basecalling_validation_baseline.json").write_text("{}\n")
    checkpoint = work / "models/synthetic-resume/heads.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save({"epoch": 1, "global_step": 3, "joint_config": config}, checkpoint)

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("resume dry-run must not initialize model or GPU")

    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    monkeypatch.setattr(joint_command, "load_bonito_v5_model", forbidden)
    before = {path: path.read_bytes() for path in work.rglob("*") if path.is_file()}
    resume_args = joint_command.build_parser().parse_args(
        [*arguments, "--resume", "--batch-size", "64"]
    )
    resumed_summary = joint_command.run(resume_args)
    assert resumed_summary["joint_config"]["length_batch_size"] == 64
    assert resumed_summary["length_batch_size_changes"][0]["from_batch_size"] == 8
    assert resumed_summary["input_pipeline"]["prefetch_batches"] == 2
    assert resumed_summary["input_pipeline"]["pod5_reader_cache_size"] == 8
    resume_args.prefetch_batches = 0
    resume_args.pod5_reader_cache_size = 0
    synchronous = joint_command.run(resume_args)
    assert synchronous["joint_config"] == resumed_summary["joint_config"]
    assert synchronous["input_pipeline"]["prefetch_batches"] == 0
    strict_args = joint_command.build_parser().parse_args(
        [*arguments, "--resume", "--batch-size", "64", "--no-allow-batch-size-change"]
    )
    with pytest.raises(ValueError, match="length_batch_size"):
        joint_command.run(strict_args)
    after = {path: path.read_bytes() for path in work.rglob("*") if path.is_file()}
    assert after == before
