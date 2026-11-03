"""Synthetic BAM adapter checks without POD5, models or GPU."""

from array import array
from pathlib import Path

import pysam
import pytest

from holypoly.evaluation.fixed_len import read_dorado


def test_dorado_primary_tags_and_supplementary_filter(tmp_path: Path) -> None:
    path = tmp_path / "synthetic.bam"
    with pysam.AlignmentFile(path, "wb", header={"HD": {"VN": "1.6"}}) as bam:
        for index, pt in enumerate((15, 0, -1, None)):
            record = pysam.AlignedSegment()
            record.query_name = f"synthetic-{index}"
            record.flag = 4
            if pt is not None:
                record.set_tag("pt", pt)
                record.set_tag("pa", array("i", [10, 11, 40, -1, -1]))
            bam.write(record)
            if index == 0:
                record.flag |= 2048
                bam.write(record)
    rows, metadata = read_dorado(path)
    assert len(rows) == 4
    assert metadata["ignored_secondary_supplementary"] == 1
    assert rows["synthetic-0"]["positive_length_nt"] == 15
    assert rows["synthetic-1"]["native_status"] == "pt_zero_with_interval"
    assert rows["synthetic-2"]["reported_length_nt"] == -1
    assert rows["synthetic-3"]["native_status"] == "missing_pt"
    with pysam.AlignmentFile(
        tmp_path / "duplicate.bam", "wb", header={"HD": {"VN": "1.6"}}
    ) as bam:
        record = pysam.AlignedSegment()
        record.query_name = "synthetic-duplicate"
        record.flag = 4
        bam.write(record)
        bam.write(record)
    with pytest.raises(ValueError, match="duplicate"):
        read_dorado(tmp_path / "duplicate.bam")
