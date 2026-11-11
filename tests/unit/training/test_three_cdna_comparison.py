"""Synthetic three-tool cohorts and interrupted-table recovery checks."""

import csv
import json
from pathlib import Path
from runpy import run_path

import pytest

from holypoly.evaluation.fixed_len import normalize_dorado
from holypoly.evaluation.three_cdna import (
    FIELDS,
    TOOLS,
    cohort_summary,
    read_polyanalysis,
)


def test_partial_polyanalysis_keeps_complete_records_without_zero_imputation(tmp_path):
    source = tmp_path / "partial.tsv"
    source.write_text(
        "read_id\tstatus\tlength_proxy_nt\tdetector_region_type\tstart_sample\tend_sample\traw_signal_length\tinterval_samples\n"
        "synthetic-a\tpolyA_proxy_available\t17\tpolyA\t0\t512\t1024\t512\n"
        "synthetic-b\tno_candidate\t\t\t\t\t1024\t\n"
        "synthetic-c\tother_region_polyT\t\tpolyT\t0\t512\t1024\t512\n"
        "synthetic-truncated\tpolyA_pro"
    )
    before = source.read_bytes()
    rows, audit = read_polyanalysis(source)
    assert len(rows) == 3 and audit["discarded_terminal_fragments"] == 1
    assert rows["synthetic-a"]["positive_length_nt"] == 17
    assert rows["synthetic-b"]["positive_length_nt"] is None
    assert rows["synthetic-c"]["status_category"] == "other_or_ambiguous"
    assert source.read_bytes() == before
    with source.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="malformed complete"):
        read_polyanalysis(source)


@pytest.mark.parametrize("proxy", ["0", "18", "nan"])
def test_polyanalysis_rejects_inconsistent_proxy(tmp_path, proxy):
    source = tmp_path / "bad.tsv"
    source.write_text(
        "read_id\tstatus\tlength_proxy_nt\tdetector_region_type\tstart_sample\tend_sample\traw_signal_length\tinterval_samples\n"
        f"synthetic\tpolyA_proxy_available\t{proxy}\tpolyA\t0\t512\t1024\t512\n"
    )
    with pytest.raises(ValueError, match="interval/30"):
        read_polyanalysis(source)


def test_three_cohorts_keep_tool_only_and_native_zero_reads():
    outputs = {
        tool: {
            "shared": normalize_dorado(20, None),
            "shared-zero": normalize_dorado(0 if tool == "Dorado" else 20, None),
            tool + "-only": normalize_dorado(40, None),
        }
        for tool in TOOLS
    }
    summary = cohort_summary(outputs)
    assert summary["common_reads"] == 2
    assert summary["common_positive_reads"] == 1
    for tool in TOOLS:
        assert summary["tools"][tool]["all_reads"] == 3
        assert summary["tools"][tool]["outside_common_reads"] == 1
        assert sum(summary["tools"][tool]["common_status_counts"].values()) == 2
    assert summary["tools"]["Dorado"]["common_status_counts"]["zero_output"] == 1


@pytest.mark.parametrize("positive", [False, True])
def test_all_separate_and_joint_figures_keep_full_ranges(tmp_path, positive):
    outputs = {
        tool: {
            "shared": normalize_dorado(17 if positive else 0, None),
            tool + "-only": normalize_dorado(1000 if positive else None, None),
        }
        for tool in TOOLS
    }
    summary = cohort_summary(outputs)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    for tool, rows in outputs.items():
        with (tmp_path / f"{tool.lower()}_results.tsv").open("w") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["read_id", *FIELDS])
            for identifier, row in rows.items():
                writer.writerow([identifier, *(row[field] for field in FIELDS)])
    with (tmp_path / "common_read_comparison.tsv").open("w") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["read_id", *[f"{t}_{f}" for t in TOOLS for f in FIELDS]])
        writer.writerow(
            ["shared", *[outputs[t]["shared"][f] for t in TOOLS for f in FIELDS]]
        )
    render = run_path(
        str(Path(__file__).resolve().parents[3] / "scripts/compare_three_cdna_tools.py")
    )["render"]
    render(tmp_path, "synthetic", 5, 300)
    qa = json.loads((tmp_path / "plot_qa.json").read_text())
    assert not qa["zero_imputation"]
    assert qa["full_range_values_retained_in_ecdf"]
    assert len(qa["figure_stems"]) == 5
    for stem in qa["figure_stems"]:
        for suffix in ("png", "pdf", "svg"):
            assert (tmp_path / f"{stem}.{suffix}").stat().st_size
    with (tmp_path / "length_summary.tsv").open() as stream:
        stats = list(csv.DictReader(stream, delimiter="\t"))
    if positive:
        assert stats[0]["above_focus_reads"] == "1"
        assert stats[0]["max_nt"] == "1000.0"
