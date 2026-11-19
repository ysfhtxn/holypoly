"""Synthetic end-to-end bundle preparation and read-only/path-safety checks."""

from __future__ import annotations

import csv
import json
import subprocess
from array import array
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pod5
import pysam
import pytest

from holypoly.datasets import build, prepare
from holypoly.datasets.prepare_sequence import sha256
from holypoly.training.data import load_training_manifest
from holypoly.training.sequence_data import load_sequence_manifest
from holypoly.training.timing_reference import load_timing_references
from tests.isolated.datasets.test_prepare_sequence import HEAD, sequence_alignment

BUNDLE_HEAD = HEAD + "CGTA" * 30


def bundle_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Create four strata of synthetic metadata, with fractional trailing blocks."""
    data, work = tmp_path / "data", tmp_path / "work"
    data.mkdir()
    work.mkdir()
    (data / "synthetic.pod5").touch()
    (data / "heads.fa").write_text(f">1_0A\n{BUNDLE_HEAD}\n>2_20A\n{BUNDLE_HEAD}\n")
    header = pysam.AlignmentHeader.from_references(
        ["1_0A", "2_20A"], [len(BUNDLE_HEAD)] * 2
    )
    records = []
    for length in (0, 20):
        for reverse in (False, True):
            for _ in range(3):
                record = sequence_alignment(
                    str(UUID(int=len(records) + 1)), reverse=reverse
                )
                record.query_sequence = BUNDLE_HEAD + (
                    "A" * 20 if length else "CGTA" * 5
                )
                record.query_qualities = array("B", [40] * record.query_length)
                record.cigar = [(0, len(BUNDLE_HEAD)), (4, 20)]
                record.set_tag("mv", array("b", [6] + [1, 0] * record.query_length))
                record.set_tag("ns", 10 + record.query_length * 12)
                fields = record.to_string().split("\t")
                fields[2] = "1_0A" if length == 0 else "2_20A"
                record = pysam.AlignedSegment.fromstring("\t".join(fields), header)
                record.set_tag("fn", "synthetic.pod5")
                records.append(record)
    for name in ("bonito.bam", "dorado.bam"):
        with pysam.AlignmentFile(data / name, "wb", header=header) as stream:
            for record in records:
                stream.write(record)
    counts = {record.query_name: int(record.get_tag("ns")) + 2 for record in records}

    class MetadataReader:
        """Expose counts only; attempting signal access fails."""

        def __init__(self, path):
            assert path == data / "synthetic.pod5"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def reads(self, *, selection, missing_ok):
            return [SimpleNamespace(num_samples=counts[selection[0]])]

    monkeypatch.setattr(pod5, "Reader", MetadataReader)
    args = build.build_parser().parse_args(
        [
            "--data-root",
            str(data),
            "--work-root",
            str(work),
            "--bonito-bam",
            "bonito.bam",
            "--dorado-bam",
            "dorado.bam",
            "--pod5-dir",
            ".",
            "--reference",
            "heads.fa",
            "--output-dir",
            "metadata/dataset",
            "--per-group",
            "3",
            "--poly-types",
            "polyA",
            "polyT",
            "--target-bases",
            "32",
            "--margin-bases",
            "12",
            "--with-timing-reference",
            "--no-progress",
        ]
    )
    return args, data, work


def test_bundle_pairs_hashes_splits_and_timing_without_changing_inputs(
    tmp_path, monkeypatch
):
    args, data, work = bundle_setup(tmp_path, monkeypatch)
    before = {path: path.read_bytes() for path in data.iterdir()}
    args.run = True
    result = build.run(args)
    output = work / args.output_dir
    assert result["status"] == "complete"
    assert result["balancing"] == "strict_length_type_zero_orientation"
    manifest = output / "length/manifest.tsv"
    lengths = load_training_manifest(manifest)
    sequences = load_sequence_manifest(
        output / "sequence/sequence_manifest.tsv",
        lengths,
        source_manifest_sha256=sha256(manifest),
        data_root=data,
    )
    assert len(lengths) == len(sequences) == 12
    splits = {row.read_id: row.split for row in lengths}
    assert all(row.split == splits[row.read_id] for row in sequences)
    with (output / "sequence/sequence_manifest.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert all(row["source_manifest_sha256"] == sha256(manifest) for row in rows)
    timing = load_timing_references(
        output / "timing/timing_reference.tsv", lengths, source_manifest=manifest
    )
    assert len(timing) == 12
    assert all(value[2] == 1 for value in timing.values())
    assert result["stages"]["length"]["balancing"]["effective_per_group"] == 3
    assert before == {path: path.read_bytes() for path in data.iterdir()}
    assert json.loads((output / "dataset.json").read_text())["status"] == "complete"
    with pytest.raises(ValueError, match="new external"):
        build.run(args)


def test_joint_launcher_validates_completed_v4_bundle(tmp_path, monkeypatch):
    """Use the recorded size and validate the manifest pairing before training."""
    args, data, work = bundle_setup(tmp_path, monkeypatch)
    args.run = True
    args.chunk_size_samples = 2400
    build.run(args)
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
    monkeypatch.setenv("GPU_ID", "0")
    for key in ("HEAD_TYPE", "RESUME", "RESUME_FROM", "INITIALIZE_FROM"):
        monkeypatch.delenv(key, raising=False)
    command = [
        "bash",
        "scripts/run_joint_training.sh",
        "--dataset-dir",
        str(args.output_dir),
        str(config),
        "--dry-run",
    ]
    ready = subprocess.run(command, capture_output=True, text=True, check=False)
    assert ready.returncode == 0, ready.stderr
    plan = json.loads(ready.stdout)
    assert plan["arguments"]["head_type"] == "legacy"
    assert plan["joint_config"]["chunk_size_samples"] == 2400
    assert not (work / "models").exists() and not (work / "logs").exists()

    output = work / args.output_dir
    summary_path = output / "dataset.json"
    summary = json.loads(summary_path.read_text())
    summary["settings"]["chunk_size_samples"] = 12000
    summary_path.write_text(json.dumps(summary))
    rejected = subprocess.run(command, capture_output=True, text=True, check=False)
    assert rejected.returncode == 2 and "model_input_samples" in rejected.stderr

    summary["settings"]["chunk_size_samples"] = 2400
    sequence = output / "sequence/sequence_manifest.tsv"
    source_hash = summary["artifacts"]["length_manifest"]["sha256"]
    sequence.write_text(sequence.read_text().replace(source_hash, "0" * 64))
    summary["artifacts"]["sequence_manifest"]["sha256"] = sha256(sequence)
    summary_path.write_text(json.dumps(summary))
    rejected = subprocess.run(command, capture_output=True, text=True, check=False)
    assert rejected.returncode == 2 and "sequence label provenance" in rejected.stderr


def test_default_preflight_opens_no_bam_or_pod5_and_writes_nothing(
    tmp_path, monkeypatch
):
    args, _, work = bundle_setup(tmp_path, monkeypatch)

    def forbidden(*args, **kwargs):
        raise AssertionError("preflight must not scan BAM/POD5")

    monkeypatch.setattr(pysam, "AlignmentFile", forbidden)
    monkeypatch.setattr(pod5, "Reader", forbidden)
    result = build.run(args)
    assert result["status"] == "planned"
    assert result["resources"]["dorado_passes"] == 3
    assert list(work.iterdir()) == []


@pytest.mark.parametrize("target", ["data", "repository", "work", "symlink"])
def test_unsafe_outputs_fail_before_scanning(tmp_path, monkeypatch, target):
    args, data, work = bundle_setup(tmp_path, monkeypatch)
    if target == "data":
        args.output_dir = data / "out"
    elif target == "repository":
        args.output_dir = Path.cwd() / "dataset-out"
    elif target == "work":
        args.output_dir = work
    else:
        (work / "link").symlink_to(data, target_is_directory=True)
        args.output_dir = work / "link/out"
    with pytest.raises(ValueError, match="new external"):
        build.run(args)
    assert not (data / "out").exists()


def test_empty_sequence_stage_marks_bundle_failed_and_keeps_length_audit(
    tmp_path, monkeypatch
):
    args, _, work = bundle_setup(tmp_path, monkeypatch)
    args.run = True
    args.target_bases = len(BUNDLE_HEAD) + 1
    with pytest.raises(ValueError, match="accepted train and validation"):
        build.run(args)
    output = work / args.output_dir
    result = json.loads((output / "dataset.json").read_text())
    assert result["status"] == "failed"
    assert result["failed_stage"] == "sequence"
    assert (output / "length/manifest.tsv").is_file()
    assert (output / "sequence/audit.tsv").is_file()
    assert "artifacts" not in result


def test_unified_entry_passes_single_type_without_strict_balance(tmp_path, monkeypatch):
    args, _, _ = bundle_setup(tmp_path, monkeypatch)
    args.poly_types = ["polyA"]
    args.with_timing_reference = False
    args.run = True
    result = build.run(args)
    assert result["balancing"] == "per_length"
    assert (
        len(
            load_training_manifest(Path(result["artifacts"]["length_manifest"]["path"]))
        )
        == 6
    )
    assert "timing_reference_manifest" not in result["artifacts"]
    assert result["stages"]["length"]["manifest_version"] == prepare.HEAD_POLY_VERSION
