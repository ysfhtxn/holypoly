"""Configuration loading with explicit precedence for external paths."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
DATA_ROOT_ENV = "HOLYPOLY_DATA_ROOT"
WORK_ROOT_ENV = "HOLYPOLY_WORK_ROOT"
M3_MANIFEST_ENV = "HOLYPOLY_M3_MANIFEST"
BONITO_MODEL_ENV = "HOLYPOLY_BONITO_MODEL"
M3_CHECKPOINT_ENV = "HOLYPOLY_M3_CHECKPOINT"


class ConfigError(ValueError):
    """Raised when a configuration file or value is invalid."""


@dataclass(frozen=True)
class PathsConfig:
    """Configured external roots and paths relative to those roots."""

    data_root: Path | None
    work_root: Path | None
    pod5_dir: Path
    basecalls_dir: Path
    reference_path: Path
    processed_dir: Path
    metadata_dir: Path
    results_dir: Path
    models_dir: Path
    logs_dir: Path
    cache_dir: Path

    def data_path(self, relative_path: Path) -> Path | None:
        """Resolve a data-relative path without accessing the filesystem."""
        if self.data_root is None:
            return None
        return self.data_root / relative_path

    def work_path(self, relative_path: Path) -> Path | None:
        """Resolve a work-relative path without accessing the filesystem."""
        if self.work_root is None:
            return None
        return self.work_root / relative_path


@dataclass(frozen=True)
class DoctorConfig:
    """Settings controlling lightweight file readability sampling."""

    sample_size: int


@dataclass(frozen=True)
class AuxiliaryLossConfig:
    """Weights for the auxiliary location, length, and shared status losses."""

    lambda_location: float = 1.0
    lambda_length: float = 1.0
    lambda_status: float = 1.0


@dataclass(frozen=True)
class TrainingConfig:
    """Single-GPU frozen-encoder training settings."""

    manifest_path: Path | None = None
    bonito_model: Path | None = None
    checkpoint_path: Path | None = None
    loss_log_path: Path = Path("logs/train.jsonl")
    epochs: int = 1
    batch_size: int = 1
    chunk_size_samples: int = 12000
    hidden_features: int = 128
    dropout: float = 0.0
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    gradient_clip_norm: float = 2.0
    random_seed: int = 20260923
    num_workers: int = 0
    device: str = "cuda:0"


@dataclass(frozen=True)
class AppConfig:
    """Complete application configuration."""

    paths: PathsConfig
    doctor: DoctorConfig
    auxiliary_loss: AuxiliaryLossConfig = AuxiliaryLossConfig()
    training: TrainingConfig = TrainingConfig()


def load_config(
    config_path: Path | None = None,
    *,
    default_path: Path = DEFAULT_CONFIG_PATH,
    cli_data_root: Path | None = None,
    cli_work_root: Path | None = None,
    environ: Mapping[str, str] | None = None,
) -> AppConfig:
    """Load default and local YAML, then apply environment and CLI overrides.

    Precedence is CLI arguments, environment variables, local configuration,
    and finally the default configuration.
    """
    merged = _read_yaml(default_path, label="default configuration")
    if config_path is not None:
        merged = _deep_merge(
            merged,
            _read_yaml(config_path, label="local configuration"),
        )

    paths_values = _mapping_value(merged, "paths")
    doctor_values = _mapping_value(merged, "doctor")
    model_values = _mapping_value(merged, "model")
    auxiliary_loss_values = _mapping_value(model_values, "auxiliary_loss")
    training_values = _mapping_value(merged, "training")
    active_environ = os.environ if environ is None else environ

    data_root = _optional_path(paths_values.get("data_root"), "paths.data_root")
    work_root = _optional_path(paths_values.get("work_root"), "paths.work_root")
    if DATA_ROOT_ENV in active_environ:
        data_root = _required_path(active_environ[DATA_ROOT_ENV], DATA_ROOT_ENV)
    if WORK_ROOT_ENV in active_environ:
        work_root = _required_path(active_environ[WORK_ROOT_ENV], WORK_ROOT_ENV)
    if cli_data_root is not None:
        data_root = cli_data_root.expanduser()
    if cli_work_root is not None:
        work_root = cli_work_root.expanduser()

    manifest_path = _optional_path(
        training_values.get("manifest_path"), "training.manifest_path"
    )
    bonito_model = _optional_path(
        training_values.get("bonito_model"), "training.bonito_model"
    )
    checkpoint_path = _optional_path(
        training_values.get("checkpoint_path"), "training.checkpoint_path"
    )
    if M3_MANIFEST_ENV in active_environ:
        manifest_path = _required_path(active_environ[M3_MANIFEST_ENV], M3_MANIFEST_ENV)
    if BONITO_MODEL_ENV in active_environ:
        bonito_model = _required_path(
            active_environ[BONITO_MODEL_ENV], BONITO_MODEL_ENV
        )
    if M3_CHECKPOINT_ENV in active_environ:
        checkpoint_path = _required_path(
            active_environ[M3_CHECKPOINT_ENV], M3_CHECKPOINT_ENV
        )

    sample_size = doctor_values.get("sample_size", 3)
    if not isinstance(sample_size, int) or isinstance(sample_size, bool):
        raise ConfigError("doctor.sample_size must be an integer")
    if sample_size < 1:
        raise ConfigError("doctor.sample_size must be at least 1")

    return AppConfig(
        paths=PathsConfig(
            data_root=data_root,
            work_root=work_root,
            pod5_dir=_required_path(paths_values.get("pod5_dir"), "paths.pod5_dir"),
            basecalls_dir=_required_path(
                paths_values.get("basecalls_dir"), "paths.basecalls_dir"
            ),
            reference_path=_required_path(
                paths_values.get("reference_path"), "paths.reference_path"
            ),
            processed_dir=_required_path(
                paths_values.get("processed_dir"), "paths.processed_dir"
            ),
            metadata_dir=_required_path(
                paths_values.get("metadata_dir"), "paths.metadata_dir"
            ),
            results_dir=_required_path(
                paths_values.get("results_dir"), "paths.results_dir"
            ),
            models_dir=_required_path(
                paths_values.get("models_dir"), "paths.models_dir"
            ),
            logs_dir=_required_path(paths_values.get("logs_dir"), "paths.logs_dir"),
            cache_dir=_required_path(paths_values.get("cache_dir"), "paths.cache_dir"),
        ),
        doctor=DoctorConfig(sample_size=sample_size),
        auxiliary_loss=AuxiliaryLossConfig(
            lambda_location=_non_negative_float(
                auxiliary_loss_values.get("lambda_location", 1.0),
                "model.auxiliary_loss.lambda_location",
            ),
            lambda_length=_non_negative_float(
                auxiliary_loss_values.get("lambda_length", 1.0),
                "model.auxiliary_loss.lambda_length",
            ),
            lambda_status=_non_negative_float(
                auxiliary_loss_values.get("lambda_status", 1.0),
                "model.auxiliary_loss.lambda_status",
            ),
        ),
        training=TrainingConfig(
            manifest_path=manifest_path,
            bonito_model=bonito_model,
            checkpoint_path=checkpoint_path,
            loss_log_path=_required_path(
                training_values.get("loss_log_path", "logs/train.jsonl"),
                "training.loss_log_path",
            ),
            epochs=_positive_int(training_values.get("epochs", 1), "training.epochs"),
            batch_size=_positive_int(
                training_values.get("batch_size", 1), "training.batch_size"
            ),
            chunk_size_samples=_positive_int(
                training_values.get("chunk_size_samples", 12000),
                "training.chunk_size_samples",
            ),
            hidden_features=_positive_int(
                training_values.get("hidden_features", 128),
                "training.hidden_features",
            ),
            dropout=_bounded_float(
                training_values.get("dropout", 0.0),
                "training.dropout",
                minimum=0.0,
                maximum=1.0,
                maximum_inclusive=False,
            ),
            learning_rate=_positive_float(
                training_values.get("learning_rate", 1e-3),
                "training.learning_rate",
            ),
            weight_decay=_non_negative_float(
                training_values.get("weight_decay", 0.0),
                "training.weight_decay",
            ),
            gradient_clip_norm=_positive_float(
                training_values.get("gradient_clip_norm", 2.0),
                "training.gradient_clip_norm",
            ),
            random_seed=_non_negative_int(
                training_values.get("random_seed", 20260923),
                "training.random_seed",
            ),
            num_workers=_non_negative_int(
                training_values.get("num_workers", 0),
                "training.num_workers",
            ),
            device=_required_device(training_values.get("device", "cuda:0")),
        ),
    )


def _read_yaml(path: Path, *, label: str) -> dict[str, Any]:
    try:
        with path.open(encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
    except OSError as error:
        raise ConfigError(f"Cannot read {label} {path}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"Invalid YAML in {label} {path}: {error}") from error

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(f"{label.capitalize()} {path} must contain a mapping")
    return loaded


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _mapping_value(config: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = config.get(key, {})
    if not isinstance(value, dict):
        raise ConfigError(f"{key} must be a mapping")
    return value


def _optional_path(value: object, field_name: str) -> Path | None:
    if value is None or value == "":
        return None
    return _required_path(value, field_name)


def _required_path(value: object, field_name: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty path string")
    return Path(value).expanduser()


def _non_negative_float(value: object, field_name: str) -> float:
    if not isinstance(value, int | float) or isinstance(value, bool):
        raise ConfigError(f"{field_name} must be a number")
    converted = float(value)
    if converted < 0:
        raise ConfigError(f"{field_name} must be non-negative")
    return converted


def _positive_float(value: object, field_name: str) -> float:
    converted = _non_negative_float(value, field_name)
    if converted == 0:
        raise ConfigError(f"{field_name} must be greater than zero")
    return converted


def _bounded_float(
    value: object,
    field_name: str,
    *,
    minimum: float,
    maximum: float,
    maximum_inclusive: bool,
) -> float:
    converted = _non_negative_float(value, field_name)
    above_maximum = converted > maximum if maximum_inclusive else converted >= maximum
    if converted < minimum or above_maximum:
        operator = "<=" if maximum_inclusive else "<"
        raise ConfigError(
            f"{field_name} must satisfy {minimum} <= value {operator} {maximum}"
        )
    return converted


def _positive_int(value: object, field_name: str) -> int:
    converted = _non_negative_int(value, field_name)
    if converted == 0:
        raise ConfigError(f"{field_name} must be greater than zero")
    return converted


def _non_negative_int(value: object, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ConfigError(f"{field_name} must be an integer")
    if value < 0:
        raise ConfigError(f"{field_name} must be non-negative")
    return value


def _required_device(value: object) -> str:
    if value != "cuda:0":
        raise ConfigError(
            "training.device must be cuda:0 after selecting the physical GPU "
            "with CUDA_VISIBLE_DEVICES"
        )
    return value
