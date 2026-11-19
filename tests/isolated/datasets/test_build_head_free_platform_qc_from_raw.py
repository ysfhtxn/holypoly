"""Whole head-free inputs and metadata-only zeros using synthetic raw data."""

from __future__ import annotations

import csv
import os
import subprocess
from pathlib import Path

import numpy as np
import pysam
import pytest

from holypoly.datasets import build_head_free_platform_qc_from_raw as module
from holypoly.datasets import build_strict_poly_dataset as engine
from holypoly.datasets.prepare_sequence import sha256
from holypoly.training.data import load_training_manifest
from holypoly.training.sequence_data import load_sequence_manifest
from tests.isolated.datasets.test_build_strict_poly_dataset import (
    strict_fixture as strict_fixture,
)
from tests.isolated.datasets.test_build_strict_poly_from_raw import (
    raw_fixture as raw_fixture,
)


def _read_table(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream, delimiter="\t"))


@pytest.mark.parametrize("missing_zero_sequences", [False, True])
@pytest.mark.parametrize("workers", [1, 2])
def test_raw_build_keeps_whole_intervals_and_metadata_zeros(
    raw_fixture, monkeypatch, missing_zero_sequences, workers
):
    old_args, data, work = raw_fixture
    # Start with raw fixtures only; no existing manifest is available.
    assert list(work.iterdir()) == []
    args = module.build_parser().parse_args(
        [
            "--data-root",
            str(data),
            "--work-root",
            str(work),
            "--bonito-bam",
            str(old_args.bonito_bam),
            "--dorado-bam",
            str(old_args.dorado_bam),
            "--pod5-dir",
            str(data),
            "--reference",
            str(old_args.reference),
            "--output-dir",
            "metadata/synthetic-head-free-qc",
            "--poly-types",
            "polyA",
            "polyT",
            "--per-group",
            "3",
            "--target-per-group",
            "3",
            "--max-signal-reads",
            "6",
            "--target-bases",
            "32",
            "--margin-bases",
            "12",
            "--workers",
            str(workers),
            "--no-progress",
            "--run",
        ]
    )
    if missing_zero_sequences:
        bam = Path(args.dorado_bam)
        with pysam.AlignmentFile(bam, "rb") as stream:
            header, records = stream.header, list(stream)
        with pysam.AlignmentFile(bam, "wb", header=header) as stream:
            for record in records:
                if record.reference_id == 0:
                    record.query_qualities = [0] * record.query_length
                stream.write(record)
    before = {path: sha256(path) for path in data.iterdir()}
    original_load = engine._load_signal
    loaded = []

    def load_positive_only(source, data, max_samples, **kwargs):
        assert float(source["nominal_length_nt"]) > 0, (
            "zero signals must not be decoded"
        )
        loaded.append(source["read_id"])
        return original_load(source, data, max_samples, **kwargs)

    monkeypatch.setattr(engine, "_load_signal", load_positive_only)
    result = module.run(args)
    assert result["status"] == "complete"
    assert result["workflow"] == module.POLICY.raw_workflow
    assert result["selected_reads"] == 12
    assert result["sequence_reads"] == (6 if missing_zero_sequences else 12)
    assert result["entry_code_sha256"] == sha256(Path(module.__file__))
    stage = result["stages"]["head_free_qc"]
    assert stage["signal_reads_attempted"] == len(loaded) == 6
    assert stage["audit_status_counts"]["weak_zero_metadata_only"] == 6
    assert stage["audit_status_counts"]["preferred_suspected_poly"] == 6
    assert stage["method_policy"]["full_head_free_input"] is True
    output = work / args.output_dir
    assert not (output / "raw_strict_dataset.json").exists()
    assert not (output / "strict").exists()
    assert (output / "head_free_qc_raw_dataset.json").is_file()
    final = output / "head_free_qc/length/manifest.tsv"
    sequences = output / "head_free_qc/sequence/sequence_manifest.tsv"
    pool = {
        r["read_id"]: r
        for r in _read_table(output / "metadata_pool/length/manifest.tsv")
    }
    rows = _read_table(final)
    for row in rows:
        parent = pool[row["read_id"]]
        assert row["dataset_method"] == "head-free-platform-qc-v1"
        assert row["chunk_start_sample"] == parent["chunk_start_sample"]
        assert row["chunk_end_sample"] == parent["chunk_end_sample"]
        assert row["candidate_start_sample"] == parent["candidate_start_sample"]
        assert row["split"] == parent["split"]
        assert row["training_length_nt"] == parent["nominal_length_nt"]
        assert row["label_source"] == "construct_nominal_weak_supervision"
        assert row["training_interval_policy"] == "full_head_free_raw_edge"
        assert not row["poly_start_sample"] and not row["poly_end_sample"]
        assert not row["poly_is_complete"]
        assert not any(key.startswith("strict_") for key in row)
        if float(row["nominal_length_nt"]) == 0:
            assert row["label_status"] == "zero_nt"
            assert row["platform_qc_signal_evaluated"] == "false"
            assert row["platform_qc_candidate_start_sample"] == ""
            assert row["reference_head_sequence_available"] == (
                "false" if missing_zero_sequences else "true"
            )
        else:
            assert row["platform_qc_signal_evaluated"] == "true"
            assert int(row["chunk_end_sample"]) - int(row["chunk_start_sample"]) == 2000
            assert (
                0
                < int(row["platform_qc_candidate_end_sample"])
                - int(row["platform_qc_candidate_start_sample"])
                < 600
            )
    records = load_training_manifest(final)
    sequence_records = load_sequence_manifest(
        sequences, records, source_manifest_sha256=sha256(final), data_root=data
    )
    positive_ids = {r.read_id for r in records if r.training_length_nt > 0}
    assert positive_ids <= {r.read_id for r in sequence_records}
    assert len(sequence_records) == result["sequence_reads"]
    assert before == {path: sha256(path) for path in before}

    # A real launch-plan check needs config only, no model weights or GPU.
    model = work / "synthetic-model"
    model.mkdir()
    (model / "config.toml").write_text(
        "[model.encoder.transformer_encoder]\ndepth=2\n[model.seqdist]\nstate_len=3\n"
    )
    config = work / "synthetic.yaml"
    config.write_text(
        f"paths:\n  data_root: {data}\n  work_root: {work}\n"
        f"training:\n  bonito_model: {model}\n"
    )
    environment = {
        **os.environ,
        "GPU_ID": "0",
        "EXPERIMENT_ID": "synthetic-full-head-free",
        "RESUME": "0",
        "INITIALIZE_FROM": "",
        "RESUME_FROM": "",
        "HEAD_TYPE": "",
    }
    command = [
        "bash",
        "scripts/run_joint_training.sh",
        "--dataset-dir",
        str(args.output_dir),
        str(config),
        "--dry-run",
    ]
    completed = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert str(final) in completed.stdout
    assert '"head_type": "tcn"' in completed.stdout
    assert not (work / "models/synthetic-full-head-free").exists()
    saved = final.read_bytes()
    final.write_bytes(saved + b"\n")
    rejected = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert rejected.returncode != 0 and "hash mismatch" in rejected.stderr


def test_positive_qc_failure_is_not_replaced_by_metadata_acceptance(
    raw_fixture, monkeypatch
):
    args, _, work = raw_fixture
    args.output_dir = Path("metadata/synthetic-head-free-positive-failure")
    args.run = True
    original = engine._load_signal

    def nonfinite_positive(source, data, max_samples, **kwargs):
        assert float(source["nominal_length_nt"]) > 0
        values, fs, path = original(source, data, max_samples, **kwargs)
        values = values.copy()
        values[0] = np.nan
        return values, fs, path

    monkeypatch.setattr(engine, "_load_signal", nonfinite_positive)
    result = module.run(args)
    assert result["status"] == "incomplete_no_training_manifest"
    stage = result["stages"]["head_free_qc"]
    assert stage["eligible_reads"] == 6
    assert stage["audit_status_counts"]["weak_zero_metadata_only"] == 6
    assert stage["audit_status_counts"]["failure_or_deferred"] == 6
    assert not (work / args.output_dir / "head_free_qc/length/manifest.tsv").exists()


def test_full_head_free_chunk_is_never_shortened_to_fit_model(strict_fixture):
    args, _, _, rows, _ = strict_fixture
    source = next(
        r
        for r in rows
        if r["nominal_length_nt"] == 50 and r["orientation"] == "forward"
    )
    source = {**source, "chunk_start_sample": 0, "chunk_end_sample": 13000}
    decision = {
        "status": "preferred_suspected_poly",
        "selected": {
            "start": 3600,
            "end": 4100,
            "boundary_status": "one_sided_or_edge_unknown",
        },
    }
    with pytest.raises(ValueError, match="exceeds_model_input"):
        engine._training_row(source, decision, 5000, args, policy=module.POLICY)
    assert (
        engine._training_row(source, decision, 5000, args)["chunk_end_sample"] < 13000
    )


def test_new_entry_plan_is_distinct_and_requires_no_old_manifest(
    raw_fixture, monkeypatch
):
    args, _, work = raw_fixture
    args.output_dir = Path("metadata/synthetic-head-free-plan")
    result = module.run(args)
    assert result["workflow"] == "head-free-platform-qc-v1-from-raw"
    assert result["method_policy"]["metadata_only_zero"] is True
    assert result["training_dataset_dir"].endswith("/head_free_qc")
    assert list(work.iterdir()) == []
    assert not any(
        a.dest in {"source_manifest", "crop_context_ms"}
        for a in module.build_parser()._actions
    )
