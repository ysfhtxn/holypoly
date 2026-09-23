from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from holypoly.config import AppConfig, DoctorConfig, PathsConfig, TrainingConfig
from holypoly.train_launcher import (
    TrainingLaunchError,
    launch_training,
    resolve_training_paths,
)


def _config(data_root: Path, work_root: Path) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(
            data_root=data_root,
            work_root=work_root,
            pod5_dir=Path("pod5"),
            basecalls_dir=Path("basecalls"),
            reference_path=Path("reference.fa"),
            processed_dir=Path("processed"),
            metadata_dir=Path("metadata"),
            results_dir=Path("results"),
            models_dir=Path("models"),
            logs_dir=Path("logs"),
            cache_dir=Path("cache"),
        ),
        doctor=DoctorConfig(sample_size=1),
        training=TrainingConfig(
            manifest_path=Path("metadata/train.tsv"),
            bonito_model=Path("models/bonito-v5"),
            checkpoint_path=Path("models/heads.pt"),
        ),
    )


def _prepare_paths(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "data"
    work_root = tmp_path / "work"
    data_root.mkdir()
    (work_root / "metadata").mkdir(parents=True)
    (work_root / "models" / "bonito-v5").mkdir(parents=True)
    (work_root / "metadata" / "train.tsv").write_text("header\n", encoding="utf-8")
    (work_root / "models" / "bonito-v5" / "config.toml").write_text(
        "[standardisation]\nmean = 0\nstdev = 1\n",
        encoding="utf-8",
    )
    return data_root, work_root


def test_training_paths_are_resolved_under_work_root(tmp_path: Path) -> None:
    data_root, work_root = _prepare_paths(tmp_path)

    resolved = resolve_training_paths(_config(data_root, work_root))

    assert resolved.manifest_path == work_root / "metadata" / "train.tsv"
    assert resolved.bonito_model == work_root / "models" / "bonito-v5"
    assert resolved.checkpoint_path == work_root / "models" / "heads.pt"
    assert resolved.loss_log_path == work_root / "logs" / "train.jsonl"


def test_training_output_outside_work_root_is_rejected(tmp_path: Path) -> None:
    data_root, work_root = _prepare_paths(tmp_path)
    config = _config(data_root, work_root)
    config = AppConfig(
        paths=config.paths,
        doctor=config.doctor,
        auxiliary_loss=config.auxiliary_loss,
        training=TrainingConfig(
            manifest_path=config.training.manifest_path,
            bonito_model=config.training.bonito_model,
            checkpoint_path=tmp_path / "outside.pt",
        ),
    )

    with pytest.raises(TrainingLaunchError, match="inside work_root"):
        resolve_training_paths(config)


def test_launcher_uses_bonito_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, work_root = _prepare_paths(tmp_path)
    project_root = tmp_path / "project"
    python_path = project_root / "src" / "bonito" / ".venv" / "bin" / "python"
    python_path.parent.mkdir(parents=True)
    python_path.write_text("", encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("holypoly.train_launcher.subprocess.run", fake_run)

    return_code = launch_training(
        _config(data_root, work_root),
        project_root=project_root,
        dry_run=True,
        resume=False,
        overwrite=False,
    )

    command = captured["command"]
    assert isinstance(command, list)
    assert command[:3] == [str(python_path), "-m", "holypoly.train_command"]
    assert "--dry-run" in command
    assert captured["cwd"] == project_root
    environment = captured["env"]
    assert isinstance(environment, dict)
    assert str(project_root / "src") in environment["PYTHONPATH"]
    assert return_code == 0
