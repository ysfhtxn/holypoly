"""Shared path, manifest and provenance checks for joint and frozen training."""

from __future__ import annotations

import math
import re
import subprocess
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from holypoly.training.data import ManifestSplit, PolyTrainingRecord
from holypoly.training.engine import epoch_checkpoint_path


def validate_training_paths(
    *,
    data_root: Path,
    work_root: Path,
    manifest_path: Path,
    model_directory: Path,
    checkpoint_path: Path,
    loss_log_path: Path,
) -> None:
    """Validate input existence and output containment before training."""
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    if not work_root.is_dir():
        raise FileNotFoundError(f"work_root does not exist: {work_root}")
    if data_root == work_root or work_root.is_relative_to(data_root):
        raise ValueError("work_root must be outside data_root")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training manifest does not exist: {manifest_path}")
    if not model_directory.is_dir():
        raise FileNotFoundError(
            f"Bonito model directory does not exist: {model_directory}"
        )
    if not (model_directory / "config.toml").is_file():
        raise FileNotFoundError(
            f"Bonito model config does not exist: {model_directory / 'config.toml'}"
        )
    for output_path, label in (
        (checkpoint_path, "checkpoint"),
        (loss_log_path, "loss log"),
        *(
            (path.resolve(), "epoch checkpoint")
            for path in _epoch_checkpoints(checkpoint_path)
        ),
    ):
        if output_path == manifest_path:
            raise ValueError(f"{label} must not overwrite the manifest")
        if label == "epoch checkpoint" and output_path == loss_log_path:
            raise ValueError("epoch checkpoint must not overwrite the loss log")
        if output_path == data_root or output_path.is_relative_to(data_root):
            raise ValueError(f"{label} must not be inside data_root: {output_path}")
        if output_path != work_root and not output_path.is_relative_to(work_root):
            raise ValueError(f"{label} must be inside work_root: {output_path}")


def validate_manifest_sources(
    records: Sequence[PolyTrainingRecord], *, data_root: Path
) -> None:
    """Check POD5 sources and required train/validation read separation."""
    split_read_ids: dict[ManifestSplit, set[str]] = {
        split: set() for split in ManifestSplit
    }
    for record in records:
        pod5_path = record.pod5_path.expanduser().resolve()
        if pod5_path != data_root and not pod5_path.is_relative_to(data_root):
            raise ValueError(
                f"manifest POD5 path is outside data_root for read "
                f"{record.read_id!r}: {pod5_path}"
            )
        if not pod5_path.is_file():
            raise FileNotFoundError(
                f"manifest POD5 file does not exist for read {record.read_id!r}: "
                f"{pod5_path}"
            )
        split_read_ids[record.split].add(record.read_id)
    overlap = (
        split_read_ids[ManifestSplit.TRAIN] & split_read_ids[ManifestSplit.VALIDATION]
    )
    if overlap:
        raise ValueError(
            "train and validation splits share read_id values: "
            + ", ".join(sorted(overlap))
        )
    if not split_read_ids[ManifestSplit.TRAIN]:
        raise ValueError("manifest contains no training reads")
    if not split_read_ids[ManifestSplit.VALIDATION]:
        raise ValueError("manifest contains no validation reads")


def manifest_summary(
    records: Sequence[PolyTrainingRecord], *, epochs: int
) -> dict[str, object]:
    """Summarize label groups and logical full-read I/O without loading signal."""
    split_counts = Counter(record.split.value for record in records)
    status_counts = Counter(record.label_status.value for record in records)
    poly_counts = Counter(record.poly_type.value for record in records)
    active_records = [
        record
        for record in records
        if record.split in {ManifestSplit.TRAIN, ManifestSplit.VALIDATION}
    ]
    raw_samples_per_epoch = sum(record.raw_signal_length for record in active_records)
    positive_lengths = {
        record.training_length_nt
        for record in active_records
        if math.isfinite(record.training_length_nt) and record.training_length_nt > 0
    }
    return {
        "records": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "poly_type_counts": dict(sorted(poly_counts.items())),
        "pod5_files": len({record.pod5_path for record in active_records}),
        "raw_samples_per_epoch": raw_samples_per_epoch,
        "approx_float32_gib_per_epoch": raw_samples_per_epoch * 4 / 1024**3,
        "epochs": epochs,
        "positive_length_values": len(positive_lengths),
    }


def validate_output_state(
    *,
    checkpoint_path: Path,
    loss_log_path: Path,
    resume: bool,
    overwrite: bool,
) -> None:
    """Require explicit resume/overwrite for existing training artifacts."""
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"cannot resume because checkpoint does not exist: {checkpoint_path}"
            )
        return
    archives = _epoch_checkpoints(checkpoint_path)
    existing = [
        path for path in (checkpoint_path, loss_log_path, *archives) if path.exists()
    ]
    if existing and not overwrite:
        raise ValueError(
            "training output already exists; use --resume or --overwrite: "
            + ", ".join(str(path) for path in existing)
        )
    if overwrite:
        if any(not path.is_file() for path in archives):
            raise ValueError("epoch checkpoint output must be a regular file")
        for path in archives:
            path.unlink()


def _epoch_checkpoints(checkpoint_path: Path) -> list[Path]:
    """Find only this checkpoint's numbered snapshots, excluding other outputs."""
    pattern = re.compile(
        re.escape(checkpoint_path.stem)
        + r"\.epoch_[0-9]{4,}"
        + re.escape(checkpoint_path.suffix)
    )
    if not checkpoint_path.parent.is_dir():
        return []
    return sorted(
        path
        for path in checkpoint_path.parent.iterdir()
        if pattern.fullmatch(path.name)
    )


def validate_epoch_checkpoint_paths(
    checkpoint_path: Path,
    *,
    start_epoch: int,
    epochs: int,
    data_root: Path,
    work_root: Path,
    protected_paths: Sequence[Path] = (),
) -> None:
    """Reject snapshot collisions and unsafe paths before any epoch is trained."""
    for epoch in range(start_epoch + 1, epochs + 1):
        archive = epoch_checkpoint_path(checkpoint_path, epoch)
        for path in (archive, archive.with_suffix(archive.suffix + ".tmp")):
            resolved = path.resolve()
            if resolved.is_relative_to(data_root):
                raise ValueError(
                    f"epoch checkpoint must not be inside data_root: {path}"
                )
            if not resolved.is_relative_to(work_root):
                raise ValueError(f"epoch checkpoint must be inside work_root: {path}")
            if resolved in protected_paths:
                raise ValueError(
                    f"epoch checkpoint must not overwrite an input/log: {path}"
                )
        if archive.exists() or archive.is_symlink():
            raise FileExistsError(f"epoch checkpoint already exists: {archive}")


def single_manifest_version(records: Sequence[PolyTrainingRecord]) -> str:
    """Require one schema version for all records in an experiment."""
    versions = {record.manifest_version for record in records}
    if len(versions) != 1:
        raise ValueError(
            "training manifest must contain exactly one manifest_version, found: "
            + ", ".join(sorted(versions))
        )
    return next(iter(versions))


def git_commit(repository: Path) -> str:
    """Read the repository commit recorded in experiment provenance."""
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cannot determine Git commit for {repository}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()
