"""Epoch snapshot safety and single-GPU synthetic training/resume checks."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch

from holypoly.models.model import HolypolyModel
from holypoly.training import command, engine
from holypoly.training.command import (
    _validate_epoch_checkpoint_paths,
    _validate_output_state,
)
from holypoly.training.engine import (
    CheckpointContext,
    EpochResult,
    append_loss_log,
    create_optimizer,
    epoch_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from holypoly.training.losses import PolyLossWeights
from tests.isolated.helpers import tiny_v5_basecaller
from tests.isolated.training.test_command import FIELDNAMES, _row


def test_epoch_snapshots_are_independent_full_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Serialize synthetic CPU weights; mock GPU metadata without CPU training."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "synthetic-device")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    model = HolypolyModel(tiny_v5_basecaller(), hidden_features=16)
    optimizer = create_optimizer(model, learning_rate=0.001, weight_decay=0)
    context = CheckpointContext(
        random_seed=17,
        bonito_commit="synthetic-commit",
        model_config={"source": "synthetic"},
        manifest_schema_version="synthetic-v1",
        loss_weights=PolyLossWeights(),
        git_commit="synthetic-commit",
    )
    latest = tmp_path / "work" / "heads.pt"
    for epoch in (1, 2):
        with torch.no_grad():
            model.length_head.length_regressor.bias.fill_(float(epoch))
        save_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=epoch * 3,
            context=context,
            work_root=tmp_path / "work",
            archive_epoch=True,
        )
    first = epoch_checkpoint_path(latest, 1)
    second = epoch_checkpoint_path(latest, 2)
    assert first.read_bytes() != second.read_bytes()
    assert latest.read_bytes() == second.read_bytes()
    for epoch in (1, 2):
        payload = torch.load(
            epoch_checkpoint_path(latest, epoch), map_location="cpu", weights_only=False
        )
        assert payload["epoch"] == epoch
        assert payload["global_step"] == epoch * 3
        assert payload["manifest_schema_version"] == "synthetic-v1"
        assert "optimizer_state_dict" in payload
        assert "python_random_state" in payload
        assert "torch_random_state" in payload
        assert (
            payload["model_state_dict"][
                "multitask.length_head.length_regressor.bias"
            ].item()
            == epoch
        )
    before = latest.read_bytes()
    with pytest.raises(FileExistsError, match="epoch checkpoint already exists"):
        save_checkpoint(
            latest,
            model=model,
            optimizer=optimizer,
            epoch=2,
            global_step=999,
            context=context,
            archive_epoch=True,
        )
    assert latest.read_bytes() == before
    assert not list(latest.parent.glob("*.tmp"))


def test_fresh_runs_protect_archives_and_explicit_overwrite_clears_only_owned_files(
    tmp_path: Path,
) -> None:
    latest = tmp_path / "heads.pt"
    loss_log = tmp_path / "train.jsonl"
    archive = epoch_checkpoint_path(latest, 7)
    archive.write_bytes(b"old-epoch")
    other = tmp_path / "other.epoch_0007.pt"
    other.write_bytes(b"other-run")
    with pytest.raises(ValueError, match="training output already exists"):
        _validate_output_state(
            checkpoint_path=latest,
            loss_log_path=loss_log,
            resume=False,
            overwrite=False,
        )
    assert archive.read_bytes() == b"old-epoch"
    _validate_output_state(
        checkpoint_path=latest, loss_log_path=loss_log, resume=False, overwrite=True
    )
    assert not archive.exists()
    assert other.read_bytes() == b"other-run"


@pytest.mark.parametrize("destination", ["data", "outside"])
def test_epoch_preflight_rejects_unsafe_symlinks(
    tmp_path: Path, destination: str
) -> None:
    data_root, work_root = tmp_path / "data", tmp_path / "work"
    data_root.mkdir()
    work_root.mkdir()
    latest = work_root / "heads.pt"
    target = tmp_path / destination / "untouched.pt"
    target.parent.mkdir(exist_ok=True)
    target.write_bytes(b"untouched")
    epoch_checkpoint_path(latest, 1).symlink_to(target)
    with pytest.raises(
        ValueError,
        match=f"{'inside data_root' if destination == 'data' else 'inside work_root'}",
    ):
        _validate_epoch_checkpoint_paths(
            latest, start_epoch=0, epochs=1, data_root=data_root, work_root=work_root
        )
    assert target.read_bytes() == b"untouched"


def test_resume_preflight_retains_completed_epochs_and_rejects_future_collisions(
    tmp_path: Path,
) -> None:
    data_root, work_root = tmp_path / "data", tmp_path / "work"
    data_root.mkdir()
    work_root.mkdir()
    latest = work_root / "heads.pt"
    epoch_checkpoint_path(latest, 1).write_bytes(b"completed")
    _validate_epoch_checkpoint_paths(
        latest, start_epoch=1, epochs=3, data_root=data_root, work_root=work_root
    )
    epoch_checkpoint_path(latest, 3).write_bytes(b"future")
    with pytest.raises(FileExistsError, match="epoch checkpoint already exists"):
        _validate_epoch_checkpoint_paths(
            latest, start_epoch=1, epochs=3, data_root=data_root, work_root=work_root
        )
    assert epoch_checkpoint_path(latest, 1).read_bytes() == b"completed"
    assert epoch_checkpoint_path(latest, 3).read_bytes() == b"future"
    with pytest.raises(ValueError, match="greater than zero"):
        epoch_checkpoint_path(latest, 0)


@pytest.mark.parametrize("mode", ["mean", "sum", "duration_only", "mean_duration"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_training_command_saves_each_epoch_and_resume_preserves_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Run three tiny synthetic epochs on one explicit GPU, with real restore."""
    assert torch.cuda.device_count() == 1, "select one GPU with CUDA_VISIBLE_DEVICES"
    data_root, work_root = tmp_path / "data", tmp_path / "work"
    data_root.mkdir()
    model_dir = work_root / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text("[standardisation]\nmean=0\nstdev=1\n")
    pod5 = data_root / "synthetic.pod5"
    pod5.touch()
    manifest = work_root / "manifest.tsv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(
            _row(
                f"synthetic-{split}",
                pod5,
                split=split,
                pod5_file_id="synthetic",
                poly_type="polyA",
                length_nt=20,
            )
            for split in ("train", "validation")
        )

    def signal_reader(**kwargs: float):
        """Replace POD5 access with deterministic synthetic signal."""
        return lambda path, read_id: torch.linspace(-1, 1, 240)

    monkeypatch.setattr(command, "Pod5SignalReader", signal_reader)
    monkeypatch.setattr(
        command, "load_bonito_v5_model", lambda *args, **kwargs: tiny_v5_basecaller()
    )
    latest, log = work_root / "heads.pt", work_root / "train.jsonl"
    arguments = [
        "--data-root",
        str(data_root),
        "--work-root",
        str(work_root),
        "--manifest",
        str(manifest),
        "--model-dir",
        str(model_dir),
        "--checkpoint",
        str(latest),
        "--loss-log",
        str(log),
        "--epochs",
        "2",
        "--batch-size",
        "1",
        "--chunk-size-samples",
        "120",
        "--hidden-features",
        "16",
        "--length-head-mode",
        mode,
        "--data-order-seed",
        "17",
        "--dropout",
        "0",
        "--learning-rate",
        "0.001",
        "--weight-decay",
        "0",
        "--gradient-clip-norm",
        "2",
        "--random-seed",
        "17",
        "--num-workers",
        "0",
        "--device",
        "cuda:0",
        "--lambda-location",
        "1",
        "--lambda-length",
        "1",
        "--lambda-status",
        "1",
    ]
    assert command.main(arguments) == 0
    old_bytes = {
        epoch: epoch_checkpoint_path(latest, epoch).read_bytes() for epoch in (1, 2)
    }
    arguments[arguments.index("--epochs") + 1] = "3"
    assert command.main([*arguments, "--resume"]) == 0
    for epoch in (1, 2):
        assert epoch_checkpoint_path(latest, epoch).read_bytes() == old_bytes[epoch]
    assert latest.read_bytes() == epoch_checkpoint_path(latest, 3).read_bytes()
    for epoch in (1, 2, 3):
        payload = torch.load(
            epoch_checkpoint_path(latest, epoch), map_location="cpu", weights_only=False
        )
        assert payload["epoch"] == payload["global_step"] == epoch
        assert payload["optimizer_state_dict"]["state"]
        assert payload["head_config"]["length_head_mode"] == mode
        assert payload["data_order_seed"] == 17
    entries = [json.loads(line) for line in log.read_text().splitlines()]
    assert [(row["epoch"], row["split"]) for row in entries] == [
        (epoch, split) for epoch in (1, 2, 3) for split in ("train", "validation")
    ]


@pytest.mark.parametrize("mode", ["mean", "sum", "duration_only", "mean_duration"])
def test_head_checkpoint_metadata_restore_and_mismatch_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Unit-test serialization/restore on CPU; mock CUDA metadata, no training."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "synthetic-device")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _: None)
    monkeypatch.setattr(engine, "require_single_cuda_device", lambda _: None)
    model = HolypolyModel(
        tiny_v5_basecaller(), hidden_features=6, length_head_mode=mode
    )
    optimizer = create_optimizer(model, learning_rate=0.001, weight_decay=0)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=2,
        global_step=7,
        context=CheckpointContext(
            random_seed=17,
            bonito_commit="synthetic",
            model_config={},
            manifest_schema_version="synthetic",
            loss_weights=PolyLossWeights(),
            git_commit="synthetic",
            data_order_seed=17,
        ),
    )
    before = path.read_bytes()
    restored = HolypolyModel(
        tiny_v5_basecaller(), hidden_features=6, length_head_mode=mode
    )
    restored_optimizer = create_optimizer(restored, learning_rate=0.001, weight_decay=0)
    state = load_checkpoint(
        path,
        model=restored,
        optimizer=restored_optimizer,
        device=torch.device("cpu"),
        data_order_seed=17,
    )
    assert (state.epoch, state.global_step) == (2, 7)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, restored.state_dict()[key], rtol=0, atol=0)
    wrong_mode = "sum" if mode == "mean" else "mean"
    mismatched = HolypolyModel(
        tiny_v5_basecaller(), hidden_features=6, length_head_mode=wrong_mode
    )
    with pytest.raises(ValueError, match="head configuration differs"):
        load_checkpoint(
            path,
            model=mismatched,
            optimizer=restored_optimizer,
            device=torch.device("cpu"),
            data_order_seed=17,
        )
    with pytest.raises(ValueError, match="data order seed differs"):
        load_checkpoint(
            path,
            model=restored,
            optimizer=restored_optimizer,
            device=torch.device("cpu"),
            data_order_seed=18,
        )
    assert path.read_bytes() == before


@pytest.mark.parametrize("length_loss", ["log_smooth_l1", "mae", "huber"])
def test_loss_metadata_roundtrip_log_and_resume_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, length_loss: str
) -> None:
    """Restore synthetic CPU state; no CPU training or real GPU is used."""
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "synthetic-device")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _: None)
    monkeypatch.setattr(engine, "require_single_cuda_device", lambda _: None)
    model = HolypolyModel(tiny_v5_basecaller(), hidden_features=6)
    optimizer = create_optimizer(model, learning_rate=0.001, weight_decay=0)
    weights = PolyLossWeights(length_loss=length_loss, huber_delta_nt=7.5)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        epoch=2,
        global_step=7,
        context=CheckpointContext(
            random_seed=17,
            bonito_commit="synthetic",
            model_config={},
            manifest_schema_version="synthetic",
            loss_weights=weights,
            git_commit="synthetic",
        ),
    )
    before = path.read_bytes()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload["loss_config"] == asdict(weights)
    state = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        loss_weights=weights,
    )
    assert state.epoch == 2
    wrong_mode = "mae" if length_loss != "mae" else "huber"
    for wrong in (
        replace(weights, length_loss=wrong_mode),
        replace(weights, lambda_length=2),
    ):
        with pytest.raises(ValueError, match="loss configuration differs"):
            load_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                device=torch.device("cpu"),
                loss_weights=wrong,
            )
    if length_loss == "huber":
        with pytest.raises(ValueError, match="loss configuration differs"):
            load_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                device=torch.device("cpu"),
                loss_weights=replace(weights, huber_delta_nt=10),
            )
    assert path.read_bytes() == before

    result = EpochResult(
        examples=2,
        batches=1,
        global_step=7,
        losses={"length_loss": 1.0},
        location_accuracy=0,
        location_macro_f1=0,
        poly_iou=0,
        length_mae_nt=2,
        status_accuracy=float("nan"),
    )
    log = tmp_path / "train.jsonl"
    append_loss_log(log, result, epoch=2, split="validation", loss_weights=weights)
    assert json.loads(log.read_text())["loss_config"] == asdict(weights)

    # Pre-option checkpoints contain only the three weights and mean log loss.
    for key in ("length_loss", "huber_delta_nt"):
        payload["loss_config"].pop(key)
    legacy = tmp_path / "legacy.pt"
    torch.save(payload, legacy)
    state = load_checkpoint(
        legacy,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )
    assert state.epoch == 2
    with pytest.raises(ValueError, match="loss configuration differs"):
        load_checkpoint(
            legacy,
            model=model,
            optimizer=optimizer,
            device=torch.device("cpu"),
            loss_weights=PolyLossWeights(length_loss="huber"),
        )
