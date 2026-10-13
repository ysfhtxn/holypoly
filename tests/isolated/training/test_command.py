from __future__ import annotations

import csv
from pathlib import Path

import pytest

from holypoly.training.command import _validate_manifest_sources, main
from holypoly.training.data import (
    ManifestLabelStatus,
    ManifestPolyType,
    ManifestSplit,
    PolyTrainingRecord,
)

FIELDNAMES = [
    "manifest_version",
    "read_id",
    "pod5_path",
    "bam_path",
    "pod5_file_id",
    "split",
    "orientation",
    "raw_signal_length",
    "chunk_start_sample",
    "chunk_end_sample",
    "poly_start_sample",
    "poly_end_sample",
    "poly_type",
    "training_length_nt",
    "nominal_length_nt",
    "observed_sequence_length_nt",
    "label_status",
    "poly_is_complete",
    "failure_reason",
    "label_source",
    "flank_source",
    "flank_status",
    "selection_reason",
]


def _row(
    read_id: str,
    pod5_path: Path,
    *,
    split: str,
    pod5_file_id: str,
    poly_type: str,
    length_nt: int,
) -> dict[str, object]:
    return {
        "manifest_version": "train-command-test-v1",
        "read_id": read_id,
        "pod5_path": pod5_path,
        "bam_path": "",
        "pod5_file_id": pod5_file_id,
        "split": split,
        "orientation": "forward",
        "raw_signal_length": 240,
        "chunk_start_sample": 60,
        "chunk_end_sample": 180,
        "poly_start_sample": 84,
        "poly_end_sample": 144,
        "poly_type": poly_type,
        "training_length_nt": length_nt,
        "nominal_length_nt": length_nt,
        "observed_sequence_length_nt": length_nt,
        "label_status": "poly_present",
        "poly_is_complete": "true",
        "failure_reason": "",
        "label_source": "synthetic",
        "flank_source": "synthetic",
        "flank_status": "validated",
        "selection_reason": "unit_test",
    }


def test_train_command_dry_run_reads_only_manifest_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data_root = tmp_path / "data"
    work_root = tmp_path / "work"
    pod5_dir = data_root / "pod5"
    model_dir = work_root / "models" / "bonito-v5"
    pod5_dir.mkdir(parents=True)
    model_dir.mkdir(parents=True)
    shared_pod5 = pod5_dir / "shared.pod5"
    shared_pod5.write_bytes(b"")
    (model_dir / "config.toml").write_text(
        "[standardisation]\nmean = 0\nstdev = 1\n",
        encoding="utf-8",
    )
    manifest_path = work_root / "metadata" / "train.tsv"
    manifest_path.parent.mkdir(parents=True)
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(
            [
                _row(
                    "train-read",
                    shared_pod5,
                    split="train",
                    pod5_file_id="shared-file",
                    poly_type="polyA",
                    length_nt=10,
                ),
                _row(
                    "validation-read",
                    shared_pod5,
                    split="validation",
                    pod5_file_id="shared-file",
                    poly_type="polyT",
                    length_nt=20,
                ),
            ]
        )
    checkpoint_path = work_root / "models" / "heads.pt"
    loss_log_path = work_root / "logs" / "train.jsonl"

    return_code = main(
        [
            "--data-root",
            str(data_root),
            "--work-root",
            str(work_root),
            "--manifest",
            str(manifest_path),
            "--model-dir",
            str(model_dir),
            "--checkpoint",
            str(checkpoint_path),
            "--loss-log",
            str(loss_log_path),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--chunk-size-samples",
            "120",
            "--hidden-features",
            "16",
            "--dropout",
            "0",
            "--learning-rate",
            "0.001",
            "--weight-decay",
            "0",
            "--gradient-clip-norm",
            "2",
            "--random-seed",
            "20260923",
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
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert return_code == 0
    assert '"records": 2' in captured.out
    assert "Dry run complete" in captured.out
    assert not checkpoint_path.exists()
    assert not loss_log_path.exists()


def test_train_and_validation_reject_duplicate_read_id(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    pod5_path = data_root / "shared.pod5"
    pod5_path.write_bytes(b"")
    common = {
        "read_id": "duplicate-read",
        "pod5_path": pod5_path,
        "pod5_file_id": "shared-file",
        "orientation": "forward",
        "raw_signal_length": 360,
        "poly_type": ManifestPolyType.POLY_A,
        "true_length_nt": 20.0,
        "label_status": ManifestLabelStatus.POLY_PRESENT,
        "poly_is_complete": True,
        "failure_reason": "",
        "nominal_length_nt": 20.0,
        "observed_sequence_length_nt": 20.0,
    }
    records = (
        PolyTrainingRecord(
            **common,
            split=ManifestSplit.TRAIN,
            chunk_start_sample=0,
            chunk_end_sample=120,
            poly_start_sample=24,
            poly_end_sample=84,
        ),
        PolyTrainingRecord(
            **common,
            split=ManifestSplit.VALIDATION,
            chunk_start_sample=120,
            chunk_end_sample=240,
            poly_start_sample=144,
            poly_end_sample=204,
        ),
    )

    with pytest.raises(ValueError, match="share read_id values: duplicate-read"):
        _validate_manifest_sources(records, data_root=data_root)
