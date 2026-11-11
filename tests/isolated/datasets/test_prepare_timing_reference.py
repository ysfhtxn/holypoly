"""Small synthetic BAM extraction, provenance and opt-in training preflight."""

import csv
import json
from array import array
from pathlib import Path

import pysam
import pytest
import torch

from holypoly.datasets import prepare_sequence, prepare_timing_reference
from holypoly.datasets.prepare_sequence import sha256
from holypoly.training import joint_command
from holypoly.training.data import load_training_manifest
from holypoly.training.timing_reference import load_timing_references
from tests.isolated.datasets.test_prepare_sequence import sequence_setup


def setup_table(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Write ordinary called windows, one missing source read, and no real signal."""
    args, records, data, work, manifest = sequence_setup(tmp_path)
    prepare_sequence.prepare(args)
    # This second BAM supplies only ordinary sequences, without any Poly candidate.
    bam = data / "timing.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=records[0].header) as stream:
        for record in records[:2]:
            record.query_sequence = "ACGT" * 50
            record.query_qualities = array("B", [40] * 200)
            stream.write(record)
    parser = prepare_timing_reference.build_parser()
    timing_args = parser.parse_args(
        [
            "--config",
            str(args.config),
            "--manifest",
            str(manifest),
            "--bam",
            str(bam),
            "--output-dir",
            "metadata/timing",
        ]
    )
    before = manifest.read_bytes(), bam.read_bytes()
    preview = prepare_timing_reference.run(timing_args)
    assert preview["scan_bam"] is False
    assert not (work / "metadata/timing").exists()
    timing_args.scan_bam = True
    summary = prepare_timing_reference.run(timing_args)
    assert before == (manifest.read_bytes(), bam.read_bytes())
    return args, work, manifest, summary


def test_prepare_retains_missing_reads_and_loads_matching_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, work, manifest, summary = setup_table(tmp_path, monkeypatch)
    path = work / "metadata/timing/timing_reference.tsv"
    mapping = load_timing_references(
        path, load_training_manifest(manifest), source_manifest=manifest
    )
    assert len(mapping) == 3
    assert mapping["synthetic-train"][2] == 1
    assert mapping["synthetic-validation"][2] == 1
    assert mapping["synthetic-test"] == (0, 0, 0)
    assert summary["coverage_by_split"]["test"] == {"reads": 1, "available": 0}
    with pytest.raises(ValueError, match="new external"):
        prepare_timing_reference.run(
            prepare_timing_reference.build_parser().parse_args(
                [
                    "--config",
                    str(tmp_path / "config.yaml"),
                    "--manifest",
                    str(manifest),
                    "--bam",
                    str(tmp_path / "data/timing.bam"),
                    "--output-dir",
                    "metadata/timing",
                ]
            )
        )


@pytest.mark.parametrize("change", ["checksum", "split", "missing_row"])
def test_corrupted_or_incomplete_timing_tables_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, change: str
) -> None:
    _, work, manifest, _ = setup_table(tmp_path, monkeypatch)
    path = work / "metadata/timing/timing_reference.tsv"
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows, fields = list(reader), reader.fieldnames
    if change == "split":
        rows[0]["split"] = "test"
    elif change == "missing_row":
        rows.pop()
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
        if change == "checksum":
            handle.write("\n")
    if change != "checksum":
        metadata = path.parent / "summary.json"
        values = json.loads(metadata.read_text())
        values["timing_table_sha256"] = sha256(path)
        metadata.write_text(json.dumps(values))
    with pytest.raises(ValueError):
        load_timing_references(
            path, load_training_manifest(manifest), source_manifest=manifest
        )


def test_opt_in_joint_preflight_uses_no_signal_weights_or_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, work, manifest, _ = setup_table(tmp_path, monkeypatch)
    model_dir = work / "model"
    model_dir.mkdir()
    (model_dir / "config.toml").write_text(
        "[model.seqdist]\nstate_len=1\n[model.encoder.transformer_encoder]\ndepth=2\n[standardisation]\nmean=0\nstdev=1\n"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("no real model, GPU or POD5 access allowed")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    monkeypatch.setattr(joint_command, "load_bonito_v5_model", forbidden)
    arguments = [
        "--config",
        str(args.config),
        "--manifest",
        str(manifest),
        "--sequence-manifest",
        str(work / args.output_dir / "sequence_manifest.tsv"),
        "--model-dir",
        str(model_dir),
        "--run-id",
        "synthetic-timing-joint",
        "--dry-run",
    ]
    default = joint_command.run(joint_command.build_parser().parse_args(arguments))
    assert "use_timing_reference" not in default["joint_config"]
    enabled = joint_command.run(
        joint_command.build_parser().parse_args(
            [
                *arguments,
                "--use-timing-reference",
                "--timing-reference-manifest",
                str(work / "metadata/timing/timing_reference.tsv"),
            ]
        )
    )
    assert enabled["joint_config"]["use_timing_reference"] is True
    assert enabled["timing_reference_coverage"]["validation"]["available"] == 1
    with pytest.raises(ValueError, match="together"):
        joint_command.run(
            joint_command.build_parser().parse_args(
                [*arguments, "--use-timing-reference"]
            )
        )
