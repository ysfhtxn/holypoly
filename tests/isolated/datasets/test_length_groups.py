from __future__ import annotations

import csv
import gzip
from array import array
from pathlib import Path

import pysam

from holypoly.datasets.length_groups import (
    build_length_groups,
    load_reference_catalog,
    parse_nominal_length_nt,
)


def _write_alignment(
    bam_file: pysam.AlignmentFile,
    *,
    read_id: str,
    reference_id: int | None,
    pod5_file_id: str,
    reverse: bool = False,
) -> None:
    alignment = pysam.AlignedSegment()
    alignment.query_name = read_id
    alignment.query_sequence = "ACGT" * 10
    alignment.query_qualities = pysam.qualitystring_to_array("I" * 40)
    if reference_id is None:
        alignment.flag = 4
    else:
        alignment.flag = 16 if reverse else 0
        alignment.reference_id = reference_id
        alignment.reference_start = 5
        alignment.mapping_quality = 60
        alignment.cigar = ((0, 40),)
    alignment.set_tag("fn", pod5_file_id)
    alignment.set_tag("ns", 240)
    alignment.set_tag("ts", 12)
    alignment.set_tag("mv", array("b", [6, 1, 0, 1]))
    bam_file.write(alignment)


def test_reference_name_nominal_length_parser() -> None:
    assert parse_nominal_length_nt("1_0A") == 0
    assert parse_nominal_length_nt("4_blpr_60A") == 60
    assert parse_nominal_length_nt("7_150A_fulllength") == 150
    assert parse_nominal_length_nt("reference") is None


def test_build_length_groups_creates_read_tables_without_copying_inputs(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    reference_path = data_root / "fixed_len_ref.fa"
    pod5_directory = data_root / "pod5" / "batch-a"
    bam_directory = data_root / "dorado_calls" / "batch-a"
    pod5_directory.mkdir(parents=True)
    bam_directory.mkdir(parents=True)
    reference_path.write_text(
        ">1_0A\nACGT\n>2_20A\nACGT\n>7_150A_fulllength\nACGT\n",
        encoding="utf-8",
    )
    pod5_path = pod5_directory / "reads.pod5"
    pod5_path.write_bytes(b"POD5")
    bam_path = bam_directory / "dorado_basecalls.bam"
    header = {
        "HD": {"VN": "1.6"},
        "SQ": [
            {"SN": "1_0A", "LN": 4},
            {"SN": "2_20A", "LN": 4},
            {"SN": "7_150A_fulllength", "LN": 4},
        ],
    }
    with pysam.AlignmentFile(bam_path, "wb", header=header) as bam_file:
        _write_alignment(
            bam_file,
            read_id="zero-read",
            reference_id=0,
            pod5_file_id=pod5_path.name,
        )
        _write_alignment(
            bam_file,
            read_id="twenty-read",
            reference_id=1,
            pod5_file_id=pod5_path.name,
            reverse=True,
        )
        _write_alignment(
            bam_file,
            read_id="unmapped-read",
            reference_id=None,
            pod5_file_id=pod5_path.name,
        )

    output_directory = data_root / "derived" / "nominal_length_groups_v1"
    summary = build_length_groups(
        data_root=data_root,
        reference_path=reference_path,
        bam_paths=(bam_path,),
        output_directory=output_directory,
        progress_every=100,
    )

    assert summary["unique_reads"] == 3
    assert summary["counts_by_group"] == {"0nt": 1, "20nt": 1, "unassigned": 1}
    assert summary["counts_by_status"] == {"assigned": 2, "unmapped": 1}
    assert pod5_path.read_bytes() == b"POD5"
    assert bam_path.is_file()
    with gzip.open(
        output_directory / "20nt" / "reads.tsv.gz", "rt", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows[0]["read_id"] == "twenty-read"
    assert rows[0]["nominal_length_nt"] == "20"
    assert rows[0]["alignment_orientation"] == "reverse"
    assert rows[0]["label_source"] == "reference_name_nominal"
    catalog = load_reference_catalog(reference_path)
    assert [record.nominal_length_nt for record in catalog] == [0, 20, 150]
