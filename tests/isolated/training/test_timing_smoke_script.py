"""Exercise the small experiment with synthetic BAM and a training-command spy."""

from __future__ import annotations

import csv
import json
import os
import shutil
import subprocess
from array import array
from pathlib import Path

import pysam
import pytest

from holypoly.datasets import prepare_sequence
from holypoly.datasets.prepare_sequence import sha256
from holypoly.training.data import load_training_manifest
from holypoly.training.sequence_data import load_sequence_manifest
from tests.isolated.datasets.test_prepare_sequence import sequence_setup


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    """Save synthetic metadata preserving every field."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def small_pipeline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Construct ten reads with one valid timing source and no real model."""
    args, records, data, work, manifest = sequence_setup(tmp_path)
    prepare_sequence.prepare(args)
    sequence = work / args.output_dir / "sequence_manifest.tsv"
    with manifest.open() as handle:
        length_rows = list(csv.DictReader(handle, delimiter="\t"))
    with sequence.open() as handle:
        sequence_rows = list(csv.DictReader(handle, delimiter="\t"))
    lengths, sequences = [], []
    for row in length_rows:
        paired = next(
            item for item in sequence_rows if item["read_id"] == row["read_id"]
        )
        for index in range(8 if row["split"] == "train" else 1):
            read_id = f"{row['read_id']}-{index}"
            lengths.append({**row, "read_id": read_id})
            sequences.append({**paired, "read_id": read_id})
    write_rows(manifest, lengths)
    for row in sequences:
        row["source_manifest_sha256"] = sha256(manifest)
    write_rows(sequence, sequences)
    bam = data / "timing.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=records[0].header) as stream:
        for row in lengths:
            record = pysam.AlignedSegment(stream.header)
            record.query_name = row["read_id"]
            record.query_sequence = "ACGT" * 50
            record.query_qualities = array("B", [40] * 200)
            record.flag = 4
            record.set_tag("mv", array("b", [6] + [1, 0] * 200))
            record.set_tag("ts", 10)
            record.set_tag("ns", int(row["raw_signal_length"]))
            stream.write(record)
    model = work / "synthetic-model"
    model.mkdir()
    (model / "config.toml").write_text("# synthetic metadata only\n")
    args.config.write_text(
        args.config.read_text() + f"\ntraining:\n  bonito_model: {model}\n"
    )
    real_project = Path(__file__).resolve().parents[3]
    project = tmp_path / "script-project"
    (project / "scripts").mkdir(parents=True)
    (project / "configs").mkdir()
    shutil.copyfile(
        real_project / "configs/default.yaml", project / "configs/default.yaml"
    )
    (project / "src").symlink_to(real_project / "src", target_is_directory=True)
    (project / ".venv").symlink_to(real_project / ".venv", target_is_directory=True)
    script = project / "scripts/run_timing_reference_smoke.sh"
    shutil.copyfile(real_project / "scripts/run_timing_reference_smoke.sh", script)
    spy = project / "scripts/run_reference_joint.sh"
    spy.write_text(
        "#!/usr/bin/env bash\nset -eu\n"
        f"'{real_project / '.venv/bin/python'}' - \"$@\" <<'PY'\n"
        "import os, sys, json\n"
        "with open(os.environ['SMOKE_CALLS'], 'a') as handle:\n"
        "    handle.write(json.dumps({'args': sys.argv[1:], "
        "'settings': {key: os.environ[key] for key in "
        "['EXPERIMENT_ID', 'USE_TIMING_REFERENCE', 'TIMING_REFERENCE_MANIFEST', "
        "'CUDA_VISIBLE_DEVICES', 'EPOCHS', 'BATCH_SIZE', 'STATUS_SUPERVISION', "
        "'LAMBDA_BASECALL', 'RESUME', 'INITIALIZE_FROM']}}) + '\\n')\n"
        "PY\n"
    )
    for key in tuple(os.environ):
        if key.startswith("HOLYPOLY_") or key in {
            "EXPERIMENT_ID",
            "INITIALIZE_FROM",
            "RESUME",
            "PER_GROUP",
            "EPOCHS",
            "GPU_ID",
            "BATCH_SIZE",
            "RUN_BASELINE",
            "SEED",
        }:
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("SMOKE_CALLS", str(work / "calls.jsonl"))
    monkeypatch.setenv("EXPERIMENT_ID", "synthetic-timing")
    monkeypatch.setenv("PER_GROUP", "10")
    monkeypatch.setenv("GPU_ID", "3")
    command = [
        "bash",
        str(script),
        str(manifest),
        str(sequence),
        str(bam),
        str(args.config),
    ]
    return command, work, manifest, sequence, bam, data


def invoke(command: list[str], *flags: str) -> subprocess.CompletedProcess[str]:
    """Run only synthetic metadata and CPU BAM extraction; never GPU training."""
    return subprocess.run(
        [*command, *flags], capture_output=True, text=True, check=False, timeout=30
    )


def test_dry_run_only_reads_metadata(small_pipeline) -> None:
    command, work, manifest, sequence, bam, _ = small_pipeline
    before = {path: path.read_bytes() for path in work.rglob("*") if path.is_file()}
    result = invoke(command, "--dry-run")
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"selected_reads": 10' in result.stderr
    assert '"training_runs": 1' in result.stderr
    assert "no BAM scan" in result.stdout
    after = {path: path.read_bytes() for path in work.rglob("*") if path.is_file()}
    assert before == after
    assert manifest.exists() and sequence.exists() and bam.exists()


@pytest.mark.parametrize("baseline", [False, True])
def test_prepares_bound_subset_and_forwards_timing_without_training(
    small_pipeline, monkeypatch: pytest.MonkeyPatch, baseline: bool
) -> None:
    command, work, manifest, sequence, bam, data = small_pipeline
    before = {path: path.read_bytes() for path in (manifest, sequence, bam)}
    monkeypatch.setenv("RUN_BASELINE", str(int(baseline)))
    result = invoke(command)
    assert result.returncode == 0, result.stdout + result.stderr
    output = work / "metadata/synthetic-timing-inputs"
    selected = output / "length/manifest.tsv"
    paired = output / "sequence/sequence_manifest.tsv"
    source_records = load_training_manifest(manifest)
    derived_records = load_training_manifest(selected)
    assert {row.read_id for row in derived_records} == {
        row.read_id for row in source_records
    }
    assert (
        len(
            load_sequence_manifest(
                paired,
                derived_records,
                source_manifest_sha256=sha256(selected),
                data_root=data,
            )
        )
        == 10
    )
    summary = json.loads((output / "timing/summary.json").read_text())
    assert summary["coverage_by_split"]["train"] == {"reads": 8, "available": 8}
    calls = [
        json.loads(line) for line in (work / "calls.jsonl").read_text().splitlines()
    ]
    assert len(calls) == (2 if baseline else 1)
    assert calls[-1]["settings"]["USE_TIMING_REFERENCE"] == "1"
    assert calls[-1]["settings"]["TIMING_REFERENCE_MANIFEST"].endswith(
        "inputs/timing/timing_reference.tsv"
    )
    assert calls[-1]["settings"]["EXPERIMENT_ID"].endswith("-timing")
    for call in calls:
        assert call["args"][:2] == [
            str(selected.relative_to(work)),
            str(paired.relative_to(work)),
        ]
        assert call["settings"]["CUDA_VISIBLE_DEVICES"] == "3"
        assert call["settings"]["EPOCHS"] == "3"
        assert call["settings"]["BATCH_SIZE"] == "8"
        assert call["settings"]["STATUS_SUPERVISION"] == "1"
        assert call["settings"]["LAMBDA_BASECALL"] == "0.1"
        assert (
            call["settings"]["RESUME"] == "0"
            and call["settings"]["INITIALIZE_FROM"] == ""
        )
    if baseline:
        assert calls[0]["settings"]["USE_TIMING_REFERENCE"] == "0"
        assert calls[0]["settings"]["TIMING_REFERENCE_MANIFEST"] == ""
    assert all(path.read_bytes() == content for path, content in before.items())
    assert (
        invoke(command).returncode != 0
    )  # Existing outputs must never be overwritten.


def test_no_timing_coverage_stops_before_training(small_pipeline) -> None:
    command, work, _, _, bam, _ = small_pipeline
    with pysam.AlignmentFile(str(bam), "rb", check_sq=False) as stream:
        header = stream.header
        records = list(stream)
    with pysam.AlignmentFile(str(bam), "wb", header=header) as stream:
        for record in records:
            record.query_qualities = array("B", [0] * len(record.query_sequence))
            stream.write(record)
    result = invoke(command)
    assert result.returncode != 0
    assert "No usable timing" in result.stdout + result.stderr
    assert not (work / "calls.jsonl").exists()


@pytest.mark.parametrize("setting", ["bad-group", "resume", "old-weights"])
def test_invalid_settings_stop_before_output(
    small_pipeline, monkeypatch: pytest.MonkeyPatch, setting: str
) -> None:
    command, work, *_ = small_pipeline
    if setting == "bad-group":
        monkeypatch.setenv("PER_GROUP", "11")
    elif setting == "resume":
        monkeypatch.setenv("RESUME", "1")
    else:
        monkeypatch.setenv("INITIALIZE_FROM", "old.pt")
    result = invoke(command)
    assert result.returncode != 0
    assert not (work / "metadata/synthetic-timing-inputs").exists()
    assert not (work / "calls.jsonl").exists()
