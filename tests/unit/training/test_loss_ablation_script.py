"""Check loss-comparison preflight using synthetic metadata and no GPU."""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

import pytest

PROJECT = Path(__file__).resolve().parents[3]
SCRIPT = PROJECT / "scripts/run_length_loss_ablation.sh"


@pytest.mark.parametrize("gpu_id", ["0", "3"])
def test_loss_preflight_reports_selected_gpu_and_actual_batch(
    tmp_path: Path, gpu_id: str
) -> None:
    """Explicit device selection and default batch match the read-only report."""
    data, work = tmp_path / "data", tmp_path / "work"
    data.mkdir()
    work.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(f"paths:\n  data_root: {data}\n  work_root: {work}\n")
    manifest = work / "manifest.tsv"
    rows = [
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
    ]
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    excluded = {
        "EXPERIMENT_ID",
        "EPOCHS",
        "BATCH_SIZE",
        "SEED",
        "LEARNING_RATE",
        "LENGTH_HEAD_MODE",
        "HUBER_DELTA_NT",
        "SUBSET_PER_GROUP",
        "GPU_ID",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith("HOLYPOLY_") and key not in excluded
    }
    environment.update(GPU_ID=gpu_id, EXPERIMENT_ID="synthetic-loss-check")
    completed = subprocess.run(
        ["bash", str(SCRIPT), str(manifest), str(config), "--dry-run"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert f"GPU {gpu_id}" in completed.stderr
    assert "batch=8" in completed.stderr
    assert "no outputs written" in completed.stdout
    assert list(work.iterdir()) == [manifest]
