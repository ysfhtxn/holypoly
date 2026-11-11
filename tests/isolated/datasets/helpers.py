"""Shared synthetic BAM and POD5 metadata fixtures for v3 and v4 preparation."""

from __future__ import annotations

import argparse
from array import array
from pathlib import Path
from types import SimpleNamespace

import pod5
import pysam
import pytest

from holypoly.datasets.prepare import build_parser

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


def quota_setup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    records: list[pysam.AlignedSegment],
    *,
    poly_types: tuple[str, ...] = ("polyA",),
) -> tuple[argparse.Namespace, list[str]]:
    """Write synthetic BAMs and track POD5 metadata calls without loading signal."""
    data_root, work_root = tmp_path / "inputs", tmp_path / "work"
    data_root.mkdir()
    work_root.mkdir()
    (data_root / "synthetic.pod5").touch()
    (data_root / "heads.fa").write_text("".join(f">{name}\n{HEAD}\n" for name in NAMES))
    header = {"SQ": [{"SN": name, "LN": len(HEAD)} for name in NAMES]}
    for filename in ("bonito.bam", "dorado.bam"):
        with pysam.AlignmentFile(data_root / filename, "wb", header=header) as bam:
            for record in records:
                bam.write(record)
    sample_counts = {record.query_name: record.get_tag("ns") + 2 for record in records}
    metadata_read_ids: list[str] = []

    class MetadataReader:
        """Count individual metadata requests; no real POD5 data is needed."""

        def __init__(self, path: Path) -> None:
            assert path == data_root / "synthetic.pod5"

        def __enter__(self) -> MetadataReader:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def reads(
            self, *, selection: list[str], missing_ok: bool
        ) -> list[SimpleNamespace]:
            metadata_read_ids.extend(selection)
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
            "--per-group",
            "3",
            "--max-records",
            "0",
            "--no-progress",
            "--stop-when-full",
            "--poly-types",
            *poly_types,
        ]
    )
    return args, metadata_read_ids
