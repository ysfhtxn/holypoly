"""Synthetic comparison tests: sentinels, missing reads and paired denominators."""

import json
from pathlib import Path
from runpy import run_path

import pytest

from holypoly.evaluation.fixed_len import (
    TOOLS,
    join_outputs,
    normalize_dorado,
    normalize_table,
    read_unique,
    summarize,
)


def test_dorado_zero_and_failure_are_retained_but_not_positive_lengths() -> None:
    no_interval = normalize_dorado(0, [100, -1, -1, -1, -1])
    with_interval = normalize_dorado(0, [100, 101, 200, -1, -1])
    failure = normalize_dorado(-1, None)
    assert no_interval["native_status"] == "pt_zero_without_interval"
    assert with_interval["native_status"] == "pt_zero_with_interval"
    assert no_interval["reported_length_nt"] == with_interval["reported_length_nt"] == 0
    assert failure["reported_length_nt"] == -1
    assert all(
        r["positive_length_nt"] is None for r in (no_interval, with_interval, failure)
    )
    assert normalize_dorado(12, [100, 101, 200])["positive_length_nt"] == 12
    assert normalize_dorado(None, None)["native_status"] == "missing_pt"


def test_full_denominator_and_same_read_subset_with_zero_group() -> None:
    labels = {
        name: {
            "nominal_length_nt": str(length),
            "poly_type": "polyA" if length else "none",
            "label_source": "synthetic_construct",
        }
        for name, length in (
            ("zero", 0),
            ("paired", 20),
            ("unpaired", 20),
            ("missing", 20),
        )
    }
    outputs = {
        tool: {
            name: normalize_dorado(value, None)
            for name, value in (("zero", 15), ("paired", 22), ("unpaired", 30))
        }
        for tool in TOOLS
    }
    outputs["Dorado"]["unpaired"] = normalize_dorado(0, None)
    joined = join_outputs(labels, outputs)
    assert len(joined) == 12
    stats = summarize(joined)
    assert stats["total_reads"] == 4
    assert stats["shared_positive_nominal_gt0_reads"] == 1
    assert stats["shared_counts_by_nominal_length"] == {20: 1}
    assert stats["native_status_counts"]["Dorado"]["missing_read"] == 1
    assert stats["status_category_counts"]["Dorado"] == {
        "positive_length": 2,
        "zero_output": 1,
        "error_or_missing": 1,
    }
    paired = [
        r for r in stats["overall"] if r["scope"] == "shared_positive_nominal_gt0"
    ]
    assert all(
        r["n"] == 1 and r["mean_absolute_nominal_deviation_nt"] == 2 for r in paired
    )
    outputs["Holypoly"]["foreign"] = normalize_dorado(20, None)
    with pytest.raises(ValueError, match="outside"):
        join_outputs(labels, outputs)


@pytest.mark.parametrize(
    "tool,status,kind_key",
    [
        ("Holypoly", "candidate_predicted", "poly_type"),
        ("PolyAnalysis", "polyA_length_proxy_available", "detector_region_type"),
    ],
)
def test_candidate_types_nonfinite_values_and_no_imputation(
    tool: str, status: str, kind_key: str
) -> None:
    row = {"status": status, "predicted_length_nt": "17", kind_key: "polyA"}
    assert normalize_table(row, tool)["positive_length_nt"] == 17
    row["predicted_length_nt"] = "nan"
    with pytest.raises(ValueError, match="invalid candidate length"):
        normalize_table(row, tool)
    row["status"] = "no_candidate"
    row["predicted_length_nt"] = ""
    assert normalize_table(row, tool)["reported_length_nt"] is None
    row.update(status=status, predicted_length_nt="0")
    assert normalize_table(row, tool)["status_category"] == "zero_output"


def test_duplicate_read_is_not_silently_overwritten(tmp_path: Path) -> None:
    table = tmp_path / "duplicate.tsv"
    table.write_text(
        "read_id\tstatus\nsynthetic\tno_candidate\nsynthetic\tno_candidate\n"
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_unique(table)


def test_plot_exports_use_real_nominal_spacing_and_full_denominators(
    tmp_path: Path,
) -> None:
    labels = {
        name: {
            "nominal_length_nt": str(length),
            "poly_type": "polyA" if length else "none",
            "label_source": "synthetic",
        }
        for name, length in (("zero", 0), ("short", 20), ("long", 150))
    }
    outputs = {
        tool: {
            name: normalize_dorado(value, None)
            for name, value in (("zero", 0), ("short", 18), ("long", 140))
        }
        for tool in TOOLS
    }
    outputs["Holypoly"]["long"] = normalize_dorado(None, None)
    joined = join_outputs(labels, outputs)
    summary = json.loads(json.dumps(summarize(joined)))
    rows = [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in joined
    ]
    script = Path(__file__).resolve().parents[3] / "scripts/compare_fixed_len_tools.py"
    functions = run_path(str(script))
    functions["overview"](summary, rows, tmp_path)
    functions["distributions"](summary, rows, tmp_path)
    functions["box_scatter_comparison"](summary, rows, tmp_path)
    for name in (
        "fixed_len_tool_comparison",
        "fixed_len_group_distributions",
        "fixed_len_box_scatter_combined",
    ):
        for suffix in ("png", "pdf", "svg"):
            assert (tmp_path / f"{name}.{suffix}").stat().st_size > 0
    svg = (tmp_path / "fixed_len_group_distributions.svg").read_text()
    assert "polyT" not in svg
    assert "Plateau = output coverage." in svg
    assert summary["shared_positive_nominal_gt0_reads"] == 1
    box_meta = json.loads((tmp_path / "box_scatter_combined_summary.json").read_text())
    assert box_meta["nominal_x_values_nt"] == [0, 20, 150]
    assert box_meta["positive_outputs_by_nominal_length"]["Holypoly"] == {
        "0": 0,
        "20": 1,
        "150": 0,
    }
    assert box_meta["all_positive_output_points_shown"]
    assert box_meta["tool_offsets_nt"] == {
        "Holypoly": -4,
        "Dorado": 0,
        "PolyAnalysis": 4,
    }
    assert box_meta["agreement_x_offsets_nt"] == 0
    assert not box_meta["zero_or_missing_imputation"]
    assert [title[0] for title in box_meta["panel_titles"]] == ["a", "b", "c"]
    combined_svg = (tmp_path / "fixed_len_box_scatter_combined.svg").read_text()
    for title in box_meta["panel_titles"]:
        assert title in combined_svg


def test_stacked_coverage_keeps_failures_and_excludes_zero_from_overall(tmp_path):
    labels = {
        name: {
            "nominal_length_nt": str(length),
            "poly_type": "polyA" if length else "none",
            "label_source": "synthetic",
        }
        for name, length in (
            ("zero", 0),
            ("positive", 20),
            ("native-zero", 20),
            ("missing", 20),
            ("no-candidate", 20),
            ("ambiguous", 20),
        )
    }
    outputs = {
        tool: {
            "zero": normalize_dorado(9, None),
            "positive": normalize_dorado(18, None),
            "native-zero": normalize_dorado(0, None),
            "no-candidate": normalize_table({"status": "no_candidate"}, "Holypoly"),
            "ambiguous": normalize_table({"status": "ambiguous_candidate"}, "Holypoly"),
        }
        for tool in TOOLS
    }
    joined = join_outputs(labels, outputs)
    summary = json.loads(json.dumps(summarize(joined)))
    rows = [
        {key: "" if value is None else str(value) for key, value in row.items()}
        for row in joined
    ]
    script = Path(__file__).resolve().parents[3] / "scripts/compare_fixed_len_tools.py"
    plot = run_path(str(script))["stacked_output_coverage"]
    plot(summary, rows, tmp_path)
    meta = json.loads((tmp_path / "output_coverage_stacked_summary.json").read_text())
    assert meta["not_verified_biological_recall"]
    for row in meta["counts"]:
        assert sum(row[key] for key in meta["stack_categories"]) == row["denominator"]
        if row["scope"] == "nominal_gt0_overall":
            assert row["denominator"] == 5
            assert row["positive_output_percent"] == 20
            assert all(row[key] == 1 for key in meta["stack_categories"])
        elif row["nominal_length_nt"] == 0:
            assert row["positive_output_percent"] == 100
    for suffix in ("png", "pdf", "svg"):
        assert (tmp_path / f"fixed_len_output_coverage_stacked.{suffix}").stat().st_size
    assert (tmp_path / "output_coverage_stacked.tsv").exists()
    with pytest.raises(ValueError, match="full group denominator"):
        plot(summary, rows[:-1], tmp_path)
