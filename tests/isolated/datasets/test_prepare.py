"""Synthetic dataset preparation checks; no external reads or models."""

from __future__ import annotations

import csv
import random
from array import array
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from uuid import UUID

import pod5
import pysam
import pytest
import torch

from holypoly.datasets.prepare import (
    balance_and_split,
    bam_records,
    build_parser,
    construct_assignments,
    flank_boundary,
    head_boundary,
    head_free_bounds,
    infer_flank,
    move_bounds,
    poly_like,
    prepare,
    read_uuid,
)
from holypoly.training.data import (
    PolyTrainingDataset,
    collate_poly_training_examples,
    load_training_manifest,
)

HEAD = "GCTCGTAGCCTGACTCGATCGTCC"
FLANK = "CGGTCGATGCTAGCGTACGCTGCC"
NAMES = ["1_0A", "2_20A", "3_40A"]


def alignment(read_id: str, length: int, reverse: bool = False) -> pysam.AlignedSegment:
    """Make reference-oriented BAM SEQ with a soft-clipped Poly and flank."""
    record = pysam.AlignedSegment()
    record.query_name = read_id
    record.query_sequence = HEAD + "A" * length + FLANK
    record.flag = 16 if reverse else 0
    record.reference_id = [0, 20, 40].index(length)
    record.reference_start = 0
    record.mapping_quality = 60
    record.cigar = [(0, len(HEAD)), (4, length + len(FLANK))]
    record.set_tag("mv", array("b", [6] + [1, 0] * len(record.query_sequence)))
    record.set_tag("ts", 10)
    record.set_tag("ns", 10 + len(record.query_sequence) * 12)
    record.set_tag("fn", "synthetic.pod5")
    return record


@pytest.mark.parametrize("limit", [0, 2, 10])
@pytest.mark.parametrize("record_count", [0, 4])
def test_bam_progress_preserves_records_and_can_be_disabled(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], limit: int, record_count: int
) -> None:
    path = tmp_path / "progress.bam"
    header = {"SQ": [{"SN": name, "LN": len(HEAD)} for name in NAMES]}
    names = [str(UUID(int=index + 1)) for index in range(record_count)]
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        for name in names:
            bam.write(alignment(name, 20))
    visible = list(bam_records(path, limit, desc="Synthetic scan"))
    output = capsys.readouterr()
    assert output.out == ""
    assert "Synthetic scan" in output.err
    if not limit:
        assert "100%" in output.err
        assert f"records={record_count}" in output.err
    silent = list(bam_records(path, limit, progress=False))
    assert capsys.readouterr().err == ""
    expected = names[:limit] if limit else names
    assert [record.query_name for record in visible] == expected
    assert [record.query_name for record in silent] == expected


def test_chunk_uuid_and_inconsistent_constructs(tmp_path: Path) -> None:
    read_id = str(UUID(int=1))
    assert read_uuid(read_id + ":0:2") == read_id
    with pytest.raises(ValueError):
        read_uuid(read_id + ":invalid")
    path = tmp_path / "chunks.bam"
    header = {"SQ": [{"SN": name, "LN": len(HEAD)} for name in NAMES]}
    with pysam.AlignmentFile(path, "wb", header=header) as bam:
        bam.write(alignment(read_id + ":0:2", 20))
        bam.write(alignment(read_id + ":1:2", 40))
    assigned, conflicts, _ = construct_assignments(path, dict.fromkeys(NAMES, HEAD), 0)
    assert assigned[read_id] == "3_40A"
    assert conflicts == {read_id}


@pytest.mark.parametrize("reverse", [False, True])
def test_head_flank_and_moves_preserve_raw_orientation(reverse: bool) -> None:
    record = alignment("synthetic", 20, reverse)
    end = head_boundary(record, HEAD, 24, 0.1)
    start = flank_boundary(record.query_sequence, end, FLANK, 20, 0.1, 0.8)
    assert (end, start) == (24, 44)
    assert move_bounds(record, end, start) == (298, 538)
    # Asymmetric positions ensure reversing query bounds actually matters.
    assert move_bounds(record, 25, 44) == ((298, 526) if reverse else (310, 538))
    record.set_tag("mv", array("b", [6, 1]))
    with pytest.raises(ValueError, match="length_mismatch"):
        move_bounds(record, end, start)


def test_missing_flank_truncation_zero_and_interruption_rules() -> None:
    with pytest.raises(ValueError, match="missing_flank"):
        flank_boundary(HEAD + "A" * 20, 24, FLANK, 20, 0.1, 0.8)
    assert flank_boundary(HEAD + FLANK, 24, FLANK, 0, 0.1, 0.8) == 24
    with pytest.raises(ValueError, match="invalid_poly"):
        flank_boundary(HEAD + "AAAA" + FLANK, 24, FLANK, 0, 0.1, 0.8)
    assert poly_like("AAAAACCAAAAA", 0.8)
    assert not poly_like("AAAAACCCAAAAA", 0.8)
    with pytest.raises(ValueError, match="exceeds_chunk"):
        head_free_bounds(alignment("synthetic", 20), 24, 120)


@pytest.mark.parametrize("reverse", [False, True])
def test_head_free_bounds_exclude_shared_move_blocks(reverse: bool) -> None:
    record = alignment("synthetic", 20, reverse)
    start, end, boundary = head_free_bounds(record, 24, 1200)
    assert (start, end) == ((10, 538) if reverse else (298, 826))
    assert boundary == (end if reverse else start)
    blocks = [1, 0] * record.query_length
    index = record.query_length - 24 - 1 if reverse else 23
    blocks[index * 2] = 2
    blocks[(index + 1) * 2] = 0
    record.set_tag("mv", array("b", [6] + blocks))
    with pytest.raises(ValueError, match="head_downstream_share_move_block"):
        head_free_bounds(record, 24, 1200)


def test_flank_inference_requires_support_and_rejects_ties() -> None:
    assert infer_flank([FLANK] * 4 + ["A" + FLANK[1:]], 3, 0.1) == FLANK
    with pytest.raises(ValueError, match="support"):
        infer_flank([FLANK], 3, 0.1)
    with pytest.raises(ValueError, match="ambiguous"):
        infer_flank([FLANK] * 3 + ["G" * 24] * 3, 3, 0.1)


@pytest.mark.parametrize(
    ("tag", "value", "reason"),
    [
        ("ts", -1, "invalid_move_table"),
        ("ns", 15, "outside_raw_signal"),
        ("pi", "parent", "split_or_duplex"),
    ],
)
def test_moves_reject_invalid_or_parent_relative_coordinates(
    tag: str, value: object, reason: str
) -> None:
    record = alignment("synthetic", 20)
    record.set_tag(tag, value)
    with pytest.raises(ValueError, match=reason):
        move_bounds(record, 24, 44)


def test_balancing_is_deterministic_and_split_ids_do_not_overlap() -> None:
    rows = [
        {
            "read_id": f"{length}-{kind}-{index}",
            "nominal_length_nt": length,
            "poly_type": kind,
        }
        for length in [20, 40]
        for kind in ["polyA", "polyT"]
        for index in range(5 if length == 20 else 8)
    ]
    selected = balance_and_split(rows, 100, random.Random(5))
    assert Counter(
        (row["nominal_length_nt"], row["poly_type"]) for row in selected
    ) == {
        (20, "polyA"): 5,
        (20, "polyT"): 5,
        (40, "polyA"): 5,
        (40, "polyT"): 5,
    }
    assert len({row["read_id"] for row in selected}) == len(selected)
    assert {row["split"] for row in selected} == {"train", "validation", "test"}
    assert selected == balance_and_split(rows, 100, random.Random(5))
    sparse = [{"read_id": "sparse", "nominal_length_nt": 60, "poly_type": "polyT"}]
    sparse_result = balance_and_split(rows + sparse, 3, random.Random(5))
    assert len(sparse_result) == 12
    assert all(row["nominal_length_nt"] != 60 for row in sparse_result)


@pytest.mark.parametrize("with_flank", [True, False])
def test_builder_roundtrip_through_training_dataset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    with_flank: bool,
) -> None:
    data_root, work_root = tmp_path / "inputs", tmp_path / "work"
    data_root.mkdir()
    work_root.mkdir()
    (data_root / "synthetic.pod5").touch()
    (data_root / "heads.fa").write_text("".join(f">{name}\n{HEAD}\n" for name in NAMES))
    records = [
        alignment(str(UUID(int=index + 1)), length, index % 2 == 1)
        for index, length in enumerate([0] * 6 + [20] * 8 + [40] * 10)
    ]
    truncated = alignment(str(UUID(int=99)), 20)
    truncated.query_sequence = HEAD + "A" * 20
    truncated.cigar = [(0, 24), (4, 20)]
    records.append(truncated)
    conflicting = alignment(str(UUID(int=100)), 20)
    records.append(conflicting)
    for record in records:
        if not with_flank:
            length = [0, 20, 40][record.reference_id]
            record.query_sequence = HEAD + "A" * length
            record.cigar = [(0, 24)] + ([(4, length)] if length else [])
        record.set_tag("mv", array("b", [6] + [1, 0] * record.query_length))
        record.set_tag("ns", 10 + record.query_length * 12)
    header = {"SQ": [{"SN": name, "LN": len(HEAD)} for name in NAMES]}
    for filename in ["bonito.bam", "dorado.bam"]:
        with pysam.AlignmentFile(data_root / filename, "wb", header=header) as bam:
            for record in records:
                if filename == "bonito.bam":
                    record.query_name += ":0:1"
                else:
                    record.query_name = record.query_name.split(":")[0]
                bam.write(record)
            if filename == "bonito.bam":
                bam.write(alignment(str(UUID(int=100)) + ":1:2", 40))
    # Real Dorado drops a fractional terminal move block; POD5 retains it.
    sample_counts = {record.query_name: record.get_tag("ns") + 2 for record in records}

    class MetadataReader:
        """POD5 metadata stand-in that cannot load any real signal."""

        def __init__(self, path: Path) -> None:
            assert path == data_root / "synthetic.pod5"

        def __enter__(self) -> MetadataReader:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def reads(
            self, *, selection: list[str], missing_ok: bool
        ) -> list[SimpleNamespace]:
            return [SimpleNamespace(num_samples=sample_counts[selection[0]])]

    monkeypatch.setattr(pod5, "Reader", MetadataReader)
    args = build_parser().parse_args(
        [
            "--data-root",
            str(data_root),
            "--work-root",
            str(work_root),
            "--bonito-bam",
            "bonito.bam",
            "--dorado-bam",
            "dorado.bam",
            "--reference",
            "heads.fa",
            "--pod5-dir",
            ".",
            "--chunk-size-samples",
            "960",
            "--min-flank-reads",
            "3",
        ]
    )
    summary = prepare(args)
    progress_output = capsys.readouterr()
    assert progress_output.out == ""
    for stage in (
        "1/3 Bonito constructs",
        "2/3 Dorado flank candidates",
        "3/3 Dorado boundaries / POD5 metadata",
    ):
        assert stage in progress_output.err
    output = work_root / "metadata/poly-dataset-v3"
    records_loaded = load_training_manifest(output / "manifest.tsv")
    expected_count = 20 if with_flank else 16
    assert len(records_loaded) == expected_count
    assert summary["flank_sequence"] == (FLANK if with_flank else "")
    assert summary["audit_counts"]["conflicting_bonito_constructs"] == 1
    dataset = PolyTrainingDataset(
        records_loaded,
        lambda path, read_id: torch.arange(sample_counts[read_id], dtype=torch.float32),
    )
    batch = collate_poly_training_examples(
        [dataset[index] for index in range(len(dataset))]
    )
    assert batch.signal.shape == (expected_count, 1, 960)
    assert batch.location_targets.shape == (expected_count, 160)
    assert torch.any(batch.raw_padding_mask)
    assert all(record.head_excluded for record in records_loaded)
    assert summary["head_excluded_reads"] == expected_count
    for index, record in enumerate(records_loaded):
        real_length = record.chunk_length_samples
        assert batch.raw_lengths_samples[index] == real_length
        assert torch.equal(
            batch.signal[index, 0, :real_length],
            torch.arange(record.chunk_start_sample, record.chunk_end_sample).float(),
        )
        assert torch.all(batch.signal[index, 0, real_length:] == 0)
        if record.orientation == "forward":
            assert record.chunk_start_sample == record.head_boundary_sample
        else:
            assert record.chunk_end_sample == record.head_boundary_sample
    if with_flank:
        assert {0, 1} <= set(batch.status_targets.tolist())
    else:
        assert torch.all(batch.location_targets == -100)
        assert torch.all(batch.status_targets == -100)
        assert all(record.poly_is_complete is None for record in records_loaded)
        assert summary["supervision_counts"] == {"length_only": 16}
    first_manifest = (output / "manifest.tsv").read_text()
    args.output_dir = Path("metadata/repeat")
    args.no_progress = True
    prepare(args)
    assert "1/3 Bonito constructs" not in capsys.readouterr().err
    assert first_manifest == (work_root / "metadata/repeat/manifest.tsv").read_text()
    with (output / "audit.tsv").open() as handle:
        assert len(list(csv.DictReader(handle, delimiter="\t"))) == 26
    args.output_dir = Path("metadata/insufficient-support")
    args.min_flank_reads = 1000
    unsupported = prepare(args)
    assert unsupported["flank_status"] == "unverified"
    assert unsupported["supervision_counts"] == {"length_only": 16}
    args.output_dir = Path("metadata/invalid-signal-length")
    for read_id in sample_counts:
        sample_counts[read_id] += 4  # Difference now equals one full stride.
    with pytest.raises(ValueError, match="no_positive_reads"):
        prepare(args)
    with (work_root / args.output_dir / "audit.tsv").open() as handle:
        audit_rows = list(csv.DictReader(handle, delimiter="\t"))
    assert any(
        row["status"] == "pod5_dorado_signal_length_mismatch" for row in audit_rows
    )
    args.output_dir = data_root / "forbidden"
    with pytest.raises(ValueError, match="outside_data_root|outside data_root"):
        prepare(args)
