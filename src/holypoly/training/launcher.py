"""Parent-environment launcher for isolated Bonito training."""

from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from holypoly.core.config import AppConfig, TrainingConfig


class TrainingLaunchError(ValueError):
    """Raised when training paths or the isolated environment are invalid."""


@dataclass(frozen=True)
class ResolvedTrainingPaths:
    """Validated absolute paths passed to the Bonito environment."""

    data_root: Path
    work_root: Path
    manifest_path: Path
    bonito_model: Path
    checkpoint_path: Path
    loss_log_path: Path


def launch_training(
    config: AppConfig,
    *,
    project_root: Path,
    dry_run: bool,
    resume: bool,
    overwrite: bool,
) -> int:
    """Validate paths and run training inside ``src/bonito/.venv``."""
    paths = resolve_training_paths(config)
    python_path = project_root / "src" / "bonito" / ".venv" / "bin" / "python"
    if not python_path.is_file():
        raise TrainingLaunchError(
            "Bonito environment is missing: expected "
            f"{python_path}. Install training dependencies there, not in the "
            "parent project environment."
        )

    training = config.training
    command = [
        str(python_path),
        "-m",
        "holypoly.training.command",
        "--data-root",
        str(paths.data_root),
        "--work-root",
        str(paths.work_root),
        "--manifest",
        str(paths.manifest_path),
        "--model-dir",
        str(paths.bonito_model),
        "--checkpoint",
        str(paths.checkpoint_path),
        "--loss-log",
        str(paths.loss_log_path),
        "--epochs",
        str(training.epochs),
        "--batch-size",
        str(training.batch_size),
        "--chunk-size-samples",
        str(training.chunk_size_samples),
        "--hidden-features",
        str(training.hidden_features),
        "--dropout",
        str(training.dropout),
        "--learning-rate",
        str(training.learning_rate),
        "--weight-decay",
        str(training.weight_decay),
        "--gradient-clip-norm",
        str(training.gradient_clip_norm),
        "--random-seed",
        str(training.random_seed),
        "--num-workers",
        str(training.num_workers),
        "--device",
        training.device,
        "--lambda-location",
        str(config.auxiliary_loss.lambda_location),
        "--lambda-length",
        str(config.auxiliary_loss.lambda_length),
        "--lambda-status",
        str(config.auxiliary_loss.lambda_status),
    ]
    if dry_run:
        command.append("--dry-run")
    if resume:
        command.append("--resume")
    if overwrite:
        command.append("--overwrite")

    environment = dict(os.environ)
    source_path = str(project_root / "src")
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_path
        if not existing_pythonpath
        else os.pathsep.join((source_path, existing_pythonpath))
    )
    completed = subprocess.run(
        command,
        cwd=project_root,
        env=environment,
        check=False,
    )
    return completed.returncode


def resolve_training_paths(config: AppConfig) -> ResolvedTrainingPaths:
    """Resolve configured paths and enforce read/write separation."""
    if config.paths.data_root is None:
        raise TrainingLaunchError("data_root is required for training")
    if config.paths.work_root is None:
        raise TrainingLaunchError("work_root is required for training")

    data_root = config.paths.data_root.expanduser().resolve()
    work_root = config.paths.work_root.expanduser().resolve()
    if not data_root.is_dir():
        raise TrainingLaunchError(f"data_root is not a directory: {data_root}")
    if not work_root.is_dir():
        raise TrainingLaunchError(f"work_root is not a directory: {work_root}")
    if data_root == work_root or work_root.is_relative_to(data_root):
        raise TrainingLaunchError("work_root must be outside data_root")
    if not os.access(work_root, os.W_OK):
        raise TrainingLaunchError(f"work_root is not writable: {work_root}")

    training = config.training
    manifest_path = _required_training_input(
        training.manifest_path,
        "training manifest",
        work_root=work_root,
        expected="file",
    )
    bonito_model = _required_training_input(
        training.bonito_model,
        "Bonito model",
        work_root=work_root,
        expected="directory",
    )
    if not (bonito_model / "config.toml").is_file():
        raise TrainingLaunchError(
            f"Bonito model config does not exist: {bonito_model / 'config.toml'}"
        )

    checkpoint_value = training.checkpoint_path
    if checkpoint_value is None:
        checkpoint_value = config.paths.models_dir / "holypoly-heads.pt"
    checkpoint_path = _resolve_work_path(checkpoint_value, work_root)
    loss_log_path = _resolve_work_path(training.loss_log_path, work_root)
    _validate_output_path(checkpoint_path, data_root, work_root, "checkpoint")
    _validate_output_path(loss_log_path, data_root, work_root, "loss log")
    if checkpoint_path == manifest_path or loss_log_path == manifest_path:
        raise TrainingLaunchError("training outputs must not overwrite the manifest")
    if checkpoint_path == loss_log_path:
        raise TrainingLaunchError("checkpoint and loss log paths must differ")

    return ResolvedTrainingPaths(
        data_root=data_root,
        work_root=work_root,
        manifest_path=manifest_path,
        bonito_model=bonito_model,
        checkpoint_path=checkpoint_path,
        loss_log_path=loss_log_path,
    )


def apply_training_overrides(
    training: TrainingConfig,
    **overrides: object,
) -> TrainingConfig:
    """Return training settings with non-``None`` CLI values applied."""
    values = {
        field_name: value
        for field_name, value in overrides.items()
        if value is not None
    }
    if not values:
        return training
    from dataclasses import replace

    replace_untyped = cast(Any, replace)
    return cast(TrainingConfig, replace_untyped(training, **values))


def _required_training_input(
    value: Path | None,
    label: str,
    *,
    work_root: Path,
    expected: str,
) -> Path:
    if value is None:
        raise TrainingLaunchError(f"{label} path is required")
    path = _resolve_work_path(value, work_root)
    exists = path.is_file() if expected == "file" else path.is_dir()
    if not exists:
        raise TrainingLaunchError(f"{label} {expected} does not exist: {path}")
    return path


def _resolve_work_path(value: Path, work_root: Path) -> Path:
    expanded = value.expanduser()
    if not expanded.is_absolute():
        expanded = work_root / expanded
    return expanded.resolve()


def _validate_output_path(
    path: Path,
    data_root: Path,
    work_root: Path,
    label: str,
) -> None:
    if path == data_root or path.is_relative_to(data_root):
        raise TrainingLaunchError(f"{label} must not be inside data_root: {path}")
    if path != work_root and not path.is_relative_to(work_root):
        raise TrainingLaunchError(f"{label} must be inside work_root: {path}")
    if path.exists() and path.is_dir():
        raise TrainingLaunchError(f"{label} path is a directory: {path}")


def render_launch_error(error: Exception) -> None:
    """Print a consistent launcher error to stderr."""
    print(f"Training configuration error: {error}", file=sys.stderr)
