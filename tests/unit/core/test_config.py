from pathlib import Path

import pytest

from holypoly.core.config import ConfigError, load_config

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def _write_default(path: Path, data_root: Path, work_root: Path) -> None:
    path.write_text(
        f"""
paths:
  data_root: {data_root}
  work_root: {work_root}
  pod5_dir: pod5
  basecalls_dir: basecalls
  reference_path: reference.fa
  processed_dir: processed
  metadata_dir: metadata
  results_dir: results
  models_dir: models
  logs_dir: logs
  cache_dir: cache
doctor:
  sample_size: 2
""".strip(),
        encoding="utf-8",
    )


def test_repository_defaults_do_not_embed_machine_paths() -> None:
    config = load_config(
        default_path=REPOSITORY_ROOT / "configs" / "default.yaml",
        environ={},
    )

    assert config.paths.data_root is None
    assert config.paths.work_root is None
    assert config.training.manifest_path is None
    assert config.training.bonito_model is None
    assert config.training.checkpoint_path is None


def test_path_precedence(tmp_path: Path) -> None:
    default_path = tmp_path / "default.yaml"
    local_path = tmp_path / "local.yaml"
    default_data = tmp_path / "default-data"
    default_work = tmp_path / "default-work"
    local_data = tmp_path / "local-data"
    local_work = tmp_path / "local-work"
    environment_data = tmp_path / "environment-data"
    environment_work = tmp_path / "environment-work"
    cli_data = tmp_path / "cli-data"
    _write_default(default_path, default_data, default_work)
    local_path.write_text(
        f"paths:\n  data_root: {local_data}\n  work_root: {local_work}\n",
        encoding="utf-8",
    )

    config = load_config(
        local_path,
        default_path=default_path,
        cli_data_root=cli_data,
        environ={
            "HOLYPOLY_DATA_ROOT": str(environment_data),
            "HOLYPOLY_WORK_ROOT": str(environment_work),
        },
    )

    assert config.paths.data_root == cli_data
    assert config.paths.work_root == environment_work
    assert config.paths.pod5_dir == Path("pod5")
    assert config.doctor.sample_size == 2


def test_local_values_override_defaults(tmp_path: Path) -> None:
    default_path = tmp_path / "default.yaml"
    local_path = tmp_path / "local.yaml"
    default_data = tmp_path / "default-data"
    default_work = tmp_path / "default-work"
    local_data = tmp_path / "local-data"
    _write_default(default_path, default_data, default_work)
    local_path.write_text(
        f"paths:\n  data_root: {local_data}\n  reference_path: custom.fa\n",
        encoding="utf-8",
    )

    config = load_config(local_path, default_path=default_path, environ={})

    assert config.paths.data_root == local_data
    assert config.paths.work_root == default_work
    assert config.paths.reference_path == Path("custom.fa")


def test_invalid_sample_size_is_rejected(tmp_path: Path) -> None:
    default_path = tmp_path / "default.yaml"
    _write_default(
        default_path,
        tmp_path / "default-data",
        tmp_path / "default-work",
    )
    local_path = tmp_path / "local.yaml"
    local_path.write_text("doctor:\n  sample_size: 0\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="at least 1"):
        load_config(local_path, default_path=default_path, environ={})


def test_auxiliary_loss_weights_are_loaded_and_validated(tmp_path: Path) -> None:
    default_path = tmp_path / "default.yaml"
    _write_default(
        default_path,
        tmp_path / "default-data",
        tmp_path / "default-work",
    )
    local_path = tmp_path / "local.yaml"
    local_path.write_text(
        """
model:
  auxiliary_loss:
    lambda_location: 2.0
    lambda_length: 0.5
    lambda_status: 3
""".strip(),
        encoding="utf-8",
    )

    config = load_config(local_path, default_path=default_path, environ={})

    assert config.auxiliary_loss.lambda_location == 2.0
    assert config.auxiliary_loss.lambda_length == 0.5
    assert config.auxiliary_loss.lambda_status == 3.0

    local_path.write_text(
        "model:\n  auxiliary_loss:\n    lambda_length: -1\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="must be non-negative"):
        load_config(local_path, default_path=default_path, environ={})


def test_training_paths_use_environment_precedence(tmp_path: Path) -> None:
    default_path = tmp_path / "default.yaml"
    _write_default(
        default_path,
        tmp_path / "data",
        tmp_path / "work",
    )
    local_path = tmp_path / "local.yaml"
    local_path.write_text(
        "training:\n"
        "  manifest_path: local.tsv\n"
        "  bonito_model: local-model\n"
        "  checkpoint_path: local.pt\n",
        encoding="utf-8",
    )

    config = load_config(
        local_path,
        default_path=default_path,
        environ={
            "HOLYPOLY_M3_MANIFEST": str(tmp_path / "environment.tsv"),
            "HOLYPOLY_BONITO_MODEL": str(tmp_path / "environment-model"),
            "HOLYPOLY_M3_CHECKPOINT": str(tmp_path / "environment.pt"),
        },
    )

    assert config.training.manifest_path == tmp_path / "environment.tsv"
    assert config.training.bonito_model == tmp_path / "environment-model"
    assert config.training.checkpoint_path == tmp_path / "environment.pt"


def test_invalid_training_device_is_rejected(tmp_path: Path) -> None:
    default_path = tmp_path / "default.yaml"
    _write_default(
        default_path,
        tmp_path / "data",
        tmp_path / "work",
    )
    local_path = tmp_path / "local.yaml"
    local_path.write_text("training:\n  device: cuda:1\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="must be cuda:0"):
        load_config(local_path, default_path=default_path, environ={})
