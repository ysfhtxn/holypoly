"""Source-separated rendering tests with synthetic local and published tables."""

import csv
from pathlib import Path
from runpy import run_path


def test_published_zeros_retained_in_counts_without_mapping_cohorts(tmp_path):
    local = tmp_path / "local.tsv"
    local.write_text(
        "tool\tnominal_length_nt\tpositive_length_nt\n"
        "Holypoly\t20\t18\nDorado\t20\t23\nPolyAnalysis\t20\t17\n"
    )
    published = tmp_path / "published.csv"
    published.write_text(
        "Group,Std_Type,tail_length\ncDNA_0,cDNA,0\ncDNA_0,cDNA,8\n"
        "cDNA_15,cDNA,14\ncDNA_15,cDNA,16\nR1,RNA,25\nR2,RNA,35\n"
    )
    plot = run_path(
        str(Path(__file__).resolve().parents[3] / "scripts/plot_nano3p_context.py")
    )["plot_context"]
    records = plot(local, published, tmp_path)
    source_zero = next(r for r in records if r["group"] == "cDNA_0")
    assert source_zero["reported_rows"] == 2
    assert source_zero["zero_rows"] == 1
    assert source_zero["positive_rows"] == 1
    assert source_zero["median_nt"] == 8
    assert not any(
        r["group"] == "15" and r["cohort"] == "local_fixed_len" for r in records
    )
    with (tmp_path / "group_distribution_summary.csv").open() as stream:
        assert len(list(csv.DictReader(stream))) == len(records)
    for suffix in ("png", "pdf", "svg"):
        assert (tmp_path / f"fixed_len_with_nano3p_context.{suffix}").stat().st_size
    svg = (tmp_path / "fixed_len_with_nano3p_context.svg").read_text()
    assert "different dataset" in svg
    assert "not a matched four-tool benchmark" in svg
