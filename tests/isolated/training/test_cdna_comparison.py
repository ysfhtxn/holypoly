"""CPU-only extraction from synthetic BAM/TSV; no models, signals or GPU."""

import argparse
import csv
import json

import pysam
import pytest

from holypoly.evaluation.cdna import extract


def test_completed_source_required_and_common_reads_exported(tmp_path):
    work, raw = tmp_path / "work", tmp_path / "raw"
    source = work / "prediction"
    source.mkdir(parents=True)
    raw.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(f"paths:\n  work_root: {work}\n  data_root: {raw}\n")
    holo = source / "predictions.tsv"
    holo.write_text(
        "read_id\tstatus\tpoly_type\tpredicted_length_nt\n"
        "synthetic-1\tcandidate_predicted\tpolyA\t20\n"
        "synthetic-2\tno_candidate\t\t\n"
    )
    bam = work / "synthetic.bam"
    with pysam.AlignmentFile(bam, "wb", header={"HD": {"VN": "1.6"}}) as stream:
        for number, pt in ((1, 22), (2, 0), (3, -1)):
            record = pysam.AlignedSegment()
            record.query_name = f"synthetic-{number}"
            record.flag = 4
            record.set_tag("pt", pt)
            stream.write(record)
    args = argparse.Namespace(
        config=config, holypoly=holo, dorado_bam=bam, output=work / "comparison"
    )
    with pytest.raises(FileNotFoundError, match="wait for prediction"):
        extract(args)
    assert not args.output.exists()
    (source / "summary.json").write_text(json.dumps({"selected_primary_bam_reads": 2}))
    extract(args)
    summary = json.loads((args.output / "summary.json").read_text())
    assert summary["common_reads"] == 2
    assert summary["common_positive_reads"] == 1
    assert summary["dorado_only_reads"] == 1
    with (args.output / "common_read_comparison.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert rows[1]["Dorado_positive_length_nt"] == ""
    assert rows[1]["Dorado_reported_length_nt"] == "0"
    assert rows[1]["Dorado_native_status"] == "pt_zero_without_interval"
    with pytest.raises(ValueError, match="new work-root"):
        extract(args)
