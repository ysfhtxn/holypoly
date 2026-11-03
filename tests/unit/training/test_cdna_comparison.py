"""Synthetic unlabeled comparison tests: cohorts, missing states and plotting."""

import csv
import json
from pathlib import Path
from runpy import run_path

import pytest

from holypoly.evaluation.cdna import cohort_summary, read_holypoly
from holypoly.evaluation.fixed_len import normalize_dorado


def test_cohort_intersection_retains_native_zero_and_tool_only_reads():
    holo = {
        "shared-positive": normalize_dorado(20, None),
        "shared-zero": normalize_dorado(5, None),
        "holo-only": normalize_dorado(40, None),
    }
    dorado = {
        "shared-positive": normalize_dorado(22, None),
        "shared-zero": normalize_dorado(0, None),
        "dorado-only": normalize_dorado(30, None),
    }
    summary = cohort_summary(holo, dorado)
    assert summary["common_reads"] == 2
    assert summary["common_positive_reads"] == 1
    assert summary["holypoly_only_reads"] == summary["dorado_only_reads"] == 1
    assert summary["tools"]["Dorado"]["common_status_counts"] == {
        "positive_length": 1,
        "zero_output": 1,
    }
    assert summary["tools"]["Dorado"]["all_primary_reads"] == 3
    with pytest.raises(ValueError, match="no common"):
        cohort_summary({"a": holo["holo-only"]}, {"b": dorado["dorado-only"]})


def test_read_holypoly_rejects_duplicates_and_preserves_missing(tmp_path):
    source = tmp_path / "holo.tsv"
    source.write_text(
        "read_id\tstatus\tpoly_type\tpredicted_length_nt\n"
        "synthetic-a\tno_candidate\t\t\n"
        "synthetic-b\tcandidate_predicted\tpolyA\t20\n"
    )
    rows = read_holypoly(source)
    assert rows["synthetic-a"]["positive_length_nt"] is None
    assert rows["synthetic-b"]["positive_length_nt"] == 20
    with source.open("a") as stream:
        stream.write("synthetic-b\tno_candidate\t\t\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_holypoly(source)


@pytest.mark.parametrize("positive", [False, True])
def test_render_single_and_joint_distributions(tmp_path, positive):
    holo = {
        "one": normalize_dorado(20 if positive else 0, None),
        "two": normalize_dorado(None, None),
    }
    dorado = {
        "one": normalize_dorado(22 if positive else 0, None),
        "two": normalize_dorado(0, None),
    }
    summary = cohort_summary(holo, dorado)
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "holypoly_lengths.tsv").write_text(
        "read_id\tpositive_length_nt\n" + ("one\t20\n" if positive else "")
    )
    with (tmp_path / "common_read_comparison.tsv").open("w") as stream:
        fields = ["read_id", "Holypoly_positive_length_nt", "Dorado_positive_length_nt"]
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerow(
            {
                "read_id": "one",
                "Holypoly_positive_length_nt": 20 if positive else "",
                "Dorado_positive_length_nt": 22 if positive else "",
            }
        )
        writer.writerow({"read_id": "two"})
    render = run_path(
        str(Path(__file__).resolve().parents[3] / "scripts/compare_cdna_tools.py")
    )["render"]
    render(tmp_path, "synthetic", 5)
    qa = json.loads((tmp_path / "plot_qa.json").read_text())
    assert qa["common_reads"] == 2
    assert not qa["zero_imputation"]
    assert not qa["accuracy_or_recall_computed"]
    for stem in ("holypoly_polya_distribution", "holypoly_dorado_polya_comparison"):
        for suffix in ("png", "pdf", "svg"):
            assert (tmp_path / f"{stem}.{suffix}").stat().st_size
