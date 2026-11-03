"""Synthetic non-Poly sequence extraction, coordinates, provenance and audits."""

from __future__ import annotations

import csv
import random
from array import array
from pathlib import Path

import pysam
import pytest

from holypoly.datasets import prepare_sequence as module
from holypoly.datasets.prepare import FIELDS, move_bounds, write_table
from holypoly.training.data import load_training_manifest
from holypoly.training.sequence_data import load_sequence_manifest

HEAD = "".join(random.Random(17).choices("ACGT", k=180))


def sequence_alignment(read_id: str, *, reverse: bool = False) -> pysam.AlignedSegment:
    """Make one known head and soft-clipped Poly with acquisition-order moves."""
    header = pysam.AlignmentHeader.from_references(["1_20A"], [len(HEAD)])
    record = pysam.AlignedSegment(header)
    record.query_name = read_id
    record.flag = 16 if reverse else 0
    record.reference_id = 0
    record.reference_start = 0
    record.mapping_quality = 60
    record.query_sequence = HEAD + "A" * 20
    record.query_qualities = array("B", [40] * record.query_length)
    record.cigar = [(0, len(HEAD)), (4, 20)]
    record.set_tag("mv", array("b", [6] + [1, 0] * record.query_length))
    record.set_tag("ts", 10)
    record.set_tag("ns", 10 + record.query_length * 12)
    return record


def sequence_setup(tmp_path: Path, *, reverse: bool = False):
    """Create real tiny synthetic BAM metadata and an unchanged v4 source manifest."""
    data, work = tmp_path / "data", tmp_path / "work"
    data.mkdir()
    work.mkdir()
    pod5 = data / "synthetic.pod5"
    pod5.touch()
    reference = data / "heads.fa"
    reference.write_text(">1_20A\n" + HEAD + "\n")
    records = [
        sequence_alignment(f"synthetic-{split}", reverse=reverse)
        for split in ("train", "validation", "test")
    ]
    bam = data / "synthetic.bam"
    with pysam.AlignmentFile(str(bam), "wb", header=records[0].header) as stream:
        for record in records:
            stream.write(record)
    rows = []
    for record, split in zip(records, ("train", "validation", "test"), strict=True):
        boundary = move_bounds(record, len(HEAD), len(HEAD))[0]
        raw = record.get_tag("ns")
        row = dict.fromkeys(FIELDS, "")
        row.update(
            manifest_version="construct-nominal-head-poly-v4",
            read_id=record.query_name,
            pod5_path=str(pod5),
            pod5_file_id="synthetic",
            bam_path=str(bam),
            split=split,
            orientation="reverse" if reverse else "forward",
            raw_signal_length=raw,
            chunk_start_sample=0 if reverse else boundary,
            chunk_end_sample=boundary if reverse else raw,
            poly_type="polyT" if reverse else "polyA",
            training_length_nt=20,
            true_length_nt=20,
            nominal_length_nt=20,
            label_status="poly_present",
            label_source="construct_nominal_weak_supervision",
            flank_source="not_applicable_no_flank",
            flank_status="not_applicable_no_flank",
            selection_reason="synthetic",
            reference_name="1_20A",
            supervision_mode="length_only",
            model_input_samples=252,
            head_excluded="true",
            head_boundary_sample=boundary,
        )
        rows.append(row)
    manifest = work / "length.tsv"
    write_table(manifest, rows, FIELDS)
    config = tmp_path / "config.yaml"
    config.write_text(f"paths:\n  data_root: {data}\n  work_root: {work}\n")
    args = module.build_parser().parse_args(
        [
            "--config",
            str(config),
            "--manifest",
            str(manifest),
            "--dorado-bam",
            str(bam),
            "--reference",
            str(reference),
            "--output-dir",
            "metadata/sequence",
            "--target-bases",
            "32",
            "--margin-bases",
            "12",
            "--no-progress",
        ]
    )
    return args, records, data, work, manifest


@pytest.mark.parametrize("reverse", [False, True])
def test_exact_head_window_uses_reference_and_acquisition_coordinates(
    reverse: bool,
) -> None:
    record = sequence_alignment("synthetic", reverse=reverse)
    window = module.select_head_window(record, HEAD, target_bases=32, margin_bases=12)
    start, end = window["reference_start_base"], window["reference_end_base"]
    assert 12 <= start < end <= len(HEAD) - 12
    target = HEAD[start:end]
    if reverse:
        target = target.translate(str.maketrans("ACGT", "TGCA"))[::-1]
    assert window["sequence_target"] == target
    assert (window["chunk_start_sample"], window["chunk_end_sample"]) == move_bounds(
        record, start, end
    )
    assert window["chunk_end_sample"] - window["chunk_start_sample"] == 32 * 12


def test_rejects_low_quality_indels_and_shared_move_boundaries() -> None:
    record = sequence_alignment("synthetic")
    record.query_qualities = array("B", [0] * record.query_length)
    with pytest.raises(ValueError, match="high_quality"):
        module.select_head_window(record, HEAD, target_bases=32, margin_bases=12)

    record = sequence_alignment("shared-moves")
    record.set_tag("mv", array("b", [6, 127, 73]))
    record.set_tag("ns", 22)
    with pytest.raises(ValueError, match="high_quality"):
        module.select_head_window(record, HEAD, target_bases=32, margin_bases=12)
    record.query_qualities = array("B", [40] * record.query_length)
    record.mapping_quality = 0
    with pytest.raises(ValueError, match="mapping_quality"):
        module.select_head_window(record, HEAD)
    record = sequence_alignment("synthetic")
    # An insertion at every 16 aligned bases prevents any exact 32-base run.
    sequence = "".join(HEAD[i : i + 16] + "A" for i in range(0, len(HEAD), 16))
    record.query_sequence = sequence
    record.query_qualities = array("B", [40] * len(sequence))
    cigar = []
    for i in range(0, len(HEAD), 16):
        cigar.extend([(0, len(HEAD[i : i + 16])), (1, 1)])
    record.cigar = cigar
    with pytest.raises(ValueError, match="high_quality"):
        module.select_head_window(record, HEAD, target_bases=32, margin_bases=12)


def test_overlapping_reference_repeats_are_not_unique() -> None:
    """str.count ignores overlapping repeats; both first and last matches matter."""
    head = "ACGT" * 45
    record = sequence_alignment("periodic")
    record.query_sequence = head + "A" * 20
    record.query_qualities = array("B", [40] * record.query_length)
    with pytest.raises(ValueError, match="high_quality"):
        module.select_head_window(record, head, target_bases=96, margin_bases=12)


@pytest.mark.parametrize("reverse", [False, True])
def test_preparation_preserves_splits_and_records_every_requested_read(
    tmp_path: Path, reverse: bool
) -> None:
    args, _, data, work, manifest = sequence_setup(tmp_path, reverse=reverse)
    original = manifest.read_bytes()
    args.max_records = 2
    summary = module.prepare(args)
    assert summary["accepted_reads"] == 2
    assert summary["audit_counts"]["not_seen_in_scanned_prefix"] == 1
    assert manifest.read_bytes() == original
    sequences = load_sequence_manifest(
        work / args.output_dir / "sequence_manifest.tsv",
        load_training_manifest(manifest),
        source_manifest_sha256=module.sha256(manifest),
        data_root=data,
    )
    assert len(sequences) == 2
    assert {r.split.value for r in sequences} == {"train", "validation"}
    with (work / args.output_dir / "audit.tsv").open() as stream:
        audit = list(csv.DictReader(stream, delimiter="\t"))
    assert len(audit) == 3 and sum(row["status"] == "rejected" for row in audit) == 1
    with pytest.raises(FileExistsError):
        module.prepare(args)


def test_dry_run_reads_no_bam_or_pod5_and_duplicate_primary_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, records, _, work, _ = sequence_setup(tmp_path)

    def forbidden(*args, **kwargs):
        raise AssertionError("dry-run must not scan BAM")

    monkeypatch.setattr(module, "bam_records", forbidden)
    args.dry_run = True
    assert module.prepare(args)["dry_run"]
    assert not (work / args.output_dir).exists()
    args.dry_run = False
    monkeypatch.setattr(
        module,
        "bam_records",
        lambda *args, **kwargs: (r for r in [records[0], records[0], *records[1:]]),
    )
    summary = module.prepare(args)
    assert summary["accepted_reads"] == 2
    assert summary["audit_counts"]["duplicate_primary_in_scanned_prefix"] == 1
