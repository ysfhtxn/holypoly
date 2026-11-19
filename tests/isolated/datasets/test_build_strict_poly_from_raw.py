"""Raw input -> fresh metadata -> strict tables -> existing TCN dry-run."""

from __future__ import annotations

import json
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import pod5
import pysam
import pytest

from holypoly.datasets import build_strict_poly_from_raw as module
from holypoly.datasets.prepare_sequence import sha256
from holypoly.training.data import load_training_manifest
from holypoly.training.dataset_bundle import resolve_dataset_bundle
from tests.isolated.datasets.test_build_strict_poly_dataset import (
    strict_fixture as strict_fixture,
)


@pytest.fixture
def raw_fixture(strict_fixture):
    """Keep only synthetic raw inputs; delete both existing manifest files."""
    source_args, data, work, _, _ = strict_fixture
    head = "".join(random.Random(73).choices("ACGT", k=300))
    reference = data / "heads.fa"
    reference.write_text(f">synthetic_0A\n{head}\n>synthetic_50A\n{head}\n")
    bam = Path(source_args.dorado_bam)
    with pysam.AlignmentFile(bam, "rb") as stream:
        header = stream.header
        records = list(stream)
    with pysam.AlignmentFile(bam, "wb", header=header) as stream:
        for record in records:
            qualities = record.query_qualities
            record.query_sequence = head + "A" * 200
            record.query_qualities = qualities
            stream.write(record)
    Path(source_args.source_manifest).unlink()
    Path(source_args.source_sequence_manifest).unlink()
    args = module.build_parser().parse_args(
        [
            "--data-root",
            str(data),
            "--work-root",
            str(work),
            "--bonito-bam",
            str(bam),
            "--dorado-bam",
            str(bam),
            "--pod5-dir",
            str(data),
            "--reference",
            str(reference),
            "--output-dir",
            "metadata/strict-fresh-raw",
            "--poly-types",
            "polyA",
            "polyT",
            "--per-group",
            "3",
            "--target-bases",
            "32",
            "--margin-bases",
            "12",
            "--no-progress",
        ]
    )
    return args, data, work


@pytest.mark.parametrize("extra_head_platform", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_fresh_raw_build_requires_no_existing_manifest_and_preserves_inputs(
    raw_fixture, monkeypatch, extra_head_platform, workers
):
    args, data, work = raw_fixture
    args.workers = workers
    assert list(work.iterdir()) == []
    before = {path: sha256(path) for path in data.iterdir()}
    original_bam_open = pysam.AlignmentFile
    opened_bams = []

    def track_bam_opens(*args, **kwargs):
        opened_bams.append(args[0])
        return original_bam_open(*args, **kwargs)

    def forbidden_rescan(*args, **kwargs):
        raise AssertionError("raw path must reuse this run's sequence scan moves")

    monkeypatch.setattr(pysam, "AlignmentFile", track_bam_opens)
    monkeypatch.setattr(module.strict, "_alignments", forbidden_rescan)
    args.target_per_group = 3
    if extra_head_platform:
        original_load = module.strict._load_signal

        def signal_with_head_platform(source, data, max_samples, **kwargs):
            values, fs, path = original_load(source, data, max_samples, **kwargs)
            values = values.copy()
            start, end = (
                (2100, 2400) if source["orientation"] == "reverse" else (2600, 2900)
            )
            values[start:end] = 140 + np.random.default_rng(17).normal(0, 0.8, 300)
            return values, fs, path

        monkeypatch.setattr(module.strict, "_load_signal", signal_with_head_platform)
    args.run = True
    result = module.run(args)
    assert result["status"] == "complete"
    assert result["workflow"] == module.WORKFLOW
    assert result["selection_scope"] == module.strict.SELECTION_SCOPE
    assert result["current_stage"] == "finished"
    assert result["selected_reads"] == 12
    assert len(opened_bams) == 3  # One Bonito + two Dorado passes, never a fourth.
    assert (
        result["stages"]["strict"]["move_source"] == "this_run_sequence_scan_in_memory"
    )
    assert result["stages"]["strict"]["resources"]["bam_passes"] == 0
    assert result["stages"]["strict"]["early_stop"]["target_reached"]
    assert result["target_split_quotas_per_length_direction"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }
    output = work / args.output_dir
    pool = output / "metadata_pool/length/manifest.tsv"
    final = output / "strict/length/manifest.tsv"
    source_records = load_training_manifest(pool)
    records = load_training_manifest(final)
    original_splits = {row.read_id: row.split for row in source_records}
    assert all(row.split == original_splits[row.read_id] for row in records)
    assert len({row.read_id for row in records}) == 12
    assert (output / "strict/read_audit.tsv").is_file()
    assert before == {path: sha256(path) for path in before}
    metadata = json.loads((output / "raw_strict_dataset.json").read_text())
    assert metadata["artifacts"]["length_manifest"]["sha256"] == sha256(final)
    assert metadata["stages"]["strict"]["source_table_sha256"]["length"] == sha256(pool)
    # Exercise the user's exact training script entry using config only, no weights.
    model = work / "synthetic-model"
    model.mkdir()
    (model / "config.toml").write_text(
        "[model.encoder.transformer_encoder]\ndepth=2\n[model.seqdist]\nstate_len=3\n"
    )
    config = work / "synthetic-config.yaml"
    config.write_text(
        f"paths:\n  data_root: {data}\n  work_root: {work}\n"
        f"training:\n  bonito_model: {model}\n"
    )
    environment = {
        **os.environ,
        "GPU_ID": "0",
        "EXPERIMENT_ID": "synthetic-raw-tcn",
        "RESUME": "0",
        "INITIALIZE_FROM": "",
        "RESUME_FROM": "",
        "HEAD_TYPE": "",
    }
    completed = subprocess.run(
        [
            "bash",
            "scripts/run_joint_training.sh",
            "--dataset-dir",
            str(args.output_dir),
            str(config),
            "--dry-run",
        ],
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert str(final) in completed.stdout
    assert '"head_type": "tcn"' in completed.stdout
    assert not (work / "models/synthetic-raw-tcn").exists()
    with pytest.raises(ValueError, match="metadata_pool"):
        resolve_dataset_bundle(output / "metadata_pool", work_root=work, data_root=data)
    assert resolve_dataset_bundle(
        output / "strict", work_root=work, data_root=data
    ) == (
        "strict-poly-candidate-v1",
        ".",
        12000,
    )
    with pytest.raises(ValueError, match="new external"):
        module.run(args)
    metadata["status"] = "failed"
    (output / "raw_strict_dataset.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="dataset is incomplete"):
        resolve_dataset_bundle(output / "strict", work_root=work, data_root=data)


def test_plan_does_not_scan_raw_data_or_write_any_manifest(raw_fixture, monkeypatch):
    args, _, work = raw_fixture

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run may not read BAM/POD5 records")

    monkeypatch.setattr(pysam, "AlignmentFile", forbidden)
    monkeypatch.setattr(pod5, "Reader", forbidden)
    result = module.run(args)
    assert result["resources"]["bonito_passes"] == 1
    assert result["resources"]["dorado_passes"] == 2
    assert result["nominal_lengths_nt"] == [0, 50]
    assert list(work.iterdir()) == []


def test_full_scan_is_explicit_and_rules_checked_before_scanning(raw_fixture):
    args, _, _ = raw_fixture
    args.max_records = 0
    with pytest.raises(ValueError, match="allow-full-bam-scan"):
        module.preflight(args)
    args.allow_full_bam_scan = True
    assert module.preflight(args)["status"] == "planned"
    args.min_mapping_quality = 0
    with pytest.raises(ValueError):
        module.preflight(args)


def test_intermediate_failure_never_marks_raw_bundle_complete(raw_fixture, monkeypatch):
    args, _, work = raw_fixture
    args.run = True

    def fail(*args, **kwargs):
        raise ValueError("synthetic metadata failure")

    monkeypatch.setattr(module.build, "run", fail)
    with pytest.raises(ValueError, match="synthetic metadata failure"):
        module.run(args)
    result = json.loads(
        (work / args.output_dir / "raw_strict_dataset.json").read_text()
    )
    assert result["status"] == "failed" and result["failed_stage"] == "metadata_pool"
    assert not (work / args.output_dir / "strict/length/manifest.tsv").exists()
