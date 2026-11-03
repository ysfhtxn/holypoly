"""Validate pooling comparison safety and selection with synthetic metadata."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT / "scripts" / "run_length_pooling_ablation.sh"


def _environment() -> dict[str, str]:
    """Keep tests independent of machine data paths and experiment overrides."""
    excluded = {
        "EXPERIMENT_ID",
        "EPOCHS",
        "BATCH_SIZE",
        "SEED",
        "LEARNING_RATE",
        "LENGTH_LOSS",
        "HUBER_DELTA_NT",
        "SUBSET_PER_GROUP",
        "GPU_ID",
    }
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("HOLYPOLY_") and key not in excluded
    }
    return {**env, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(PROJECT / "src")}


def _block(index: int, args: list[str]) -> subprocess.CompletedProcess[str]:
    """Execute an actual embedded script block without launching training."""
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", SCRIPT.read_text(), re.DOTALL)
    assert len(blocks) == 3
    return subprocess.run(
        [sys.executable, "-c", blocks[index], *args],
        env=_environment(),
        capture_output=True,
        text=True,
        check=False,
    )


def _table(path: Path, rows: list[dict[str, object]]) -> None:
    """Write synthetic TSV fixtures with no real sample information."""
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def test_dry_run_is_read_only_and_refuses_existing_experiments(tmp_path: Path) -> None:
    data, work = tmp_path / "data", tmp_path / "work"
    data.mkdir()
    work.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(f"paths:\n  data_root: {data}\n  work_root: {work}\n")
    manifest = work / "manifest.tsv"
    _table(
        manifest,
        [
            {
                "read_id": f"synthetic-{split}",
                "split": split,
                "nominal_length_nt": 20,
                "poly_type": "polyA",
                "manifest_version": "construct-nominal-head-poly-v4",
                "supervision_mode": "length_only",
                "raw_signal_length": 120,
            }
            for split in ("train", "validation", "test")
        ],
    )
    before = manifest.read_bytes()
    args = ["bash", str(SCRIPT), str(manifest), str(config), "--dry-run"]
    env = {**_environment(), "EXPERIMENT_ID": "synthetic-comparison"}
    result = subprocess.run(args, env=env, capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert "no outputs written" in result.stdout
    assert "batch=8" in result.stderr and "loss=huber" in result.stderr
    assert list(work.iterdir()) == [manifest]
    assert manifest.read_bytes() == before
    collision = work / "models" / "synthetic-comparison-sum"
    collision.mkdir(parents=True)
    result = subprocess.run(args, env=env, capture_output=True, text=True, check=False)
    assert result.returncode != 0
    assert "Choose a new EXPERIMENT_ID" in result.stderr
    assert not (work / "logs").exists()


def test_best_checkpoint_uses_validation_and_earliest_tie(tmp_path: Path) -> None:
    run = "synthetic-mean"
    log = tmp_path / "logs" / run / "train.jsonl"
    log.parent.mkdir(parents=True)
    rows = [
        {
            "epoch": epoch,
            "split": split,
            "length_mae_nt": value,
            "loss_config": {"length_loss": "huber"},
        }
        for epoch, value in ((1, 20), (2, 10), (3, 10))
        for split in ("train", "validation")
    ]
    log.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    checkpoint = tmp_path / "models" / run / "heads.epoch_0002.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    result = _block(1, [str(tmp_path), run, "3"])
    assert result.returncode == 0, result.stderr
    best = json.loads(
        (
            tmp_path / "results" / "training_evaluation" / run / "best_checkpoint.json"
        ).read_text()
    )
    assert best["epoch"] == 2 and best["checkpoint"] == str(checkpoint)
    log.write_text(json.dumps(rows[0]) + "\n")
    result = _block(1, [str(tmp_path), run, "3"])
    assert result.returncode != 0 and "incomplete" in result.stderr


@pytest.mark.parametrize("mismatch", [None, "target", "batch", "head", "delta"])
def test_comparison_tables_require_same_targets_and_settings(
    tmp_path: Path,
    mismatch: str | None,
) -> None:
    experiment = "synthetic-comparison"
    for mode in ("mean", "sum"):
        directory = (
            tmp_path / "results" / "training_evaluation" / f"{experiment}-{mode}"
        )
        result = directory / "test-best-validation"
        result.mkdir(parents=True)
        loss = {"length_loss": "huber", "huber_delta_nt": 10}
        summary = {
            "partial_evaluation": False,
            "selected_reads": 2,
            "split_total_reads": 2,
            "epoch": 2,
            "checkpoint": f"synthetic-{mode}.pt",
            "loss_config": loss,
            "head_config": {
                "length_head_mode": mode,
                "hidden_features": 128,
                "dropout": 0,
            },
            "manifest": "synthetic-manifest.tsv",
            "model_dir": "synthetic-model",
            "random_seed": 17,
            "data_order_seed": 17,
            "batch_size": 8,
            "model_config": {},
            "git_commit": "synthetic",
            "bonito_commit": "synthetic",
            "metrics": {"mae_nt": 2, "within_10nt_fraction": 1},
        }
        targets = [0, 150]
        if mode == "sum":
            if mismatch == "target":
                targets[1] = 100
            elif mismatch == "batch":
                summary["batch_size"] = 16
            elif mismatch == "head":
                summary["head_config"]["length_head_mode"] = "mean"
            elif mismatch == "delta":
                loss["huber_delta_nt"] = 5
        best = {
            "epoch": 2,
            "validation_mae_nt": 3,
            "checkpoint": summary["checkpoint"],
            "loss_config": loss,
        }
        (directory / "best_checkpoint.json").write_text(json.dumps(best))
        (result / "summary.json").write_text(json.dumps(summary))
        _table(
            result / "predictions.tsv",
            [
                {
                    "read_id": f"synthetic-{index}",
                    "training_length_nt": target,
                    "regression_length_nt": target + 2,
                }
                for index, target in enumerate(targets)
            ],
        )
        _table(
            result / "metrics_by_group.tsv",
            [
                {
                    "nominal_length_nt": target,
                    "evaluable_reads": 1,
                    "mae_nt": 2,
                }
                for target in targets
            ],
        )
    completed = _block(2, [str(tmp_path), experiment, "huber", "10"])
    output = tmp_path / "results" / "training_evaluation" / experiment
    if mismatch:
        assert completed.returncode != 0 and "ValueError" in completed.stderr
        assert not output.exists()
    else:
        assert completed.returncode == 0, completed.stderr
        with (output / "comparison.tsv").open() as stream:
            overall = list(csv.DictReader(stream, delimiter="\t"))
        assert [row["length_head_mode"] for row in overall] == ["mean", "sum"]
        assert all(float(row["positive_mae_nt"]) == 2 for row in overall)
        with (output / "comparison_by_length.tsv").open() as stream:
            assert len(list(csv.DictReader(stream, delimiter="\t"))) == 4
