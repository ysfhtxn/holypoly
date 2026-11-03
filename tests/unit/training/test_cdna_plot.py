"""Render actual candidate distributions, without nominal labels or real reads."""

import csv
import json
import sys
from pathlib import Path
from runpy import run_path

import pytest


@pytest.mark.parametrize("has_prediction", [False, True])
def test_distribution_keeps_denominator_and_only_present_types(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    has_prediction: bool,
) -> None:
    """Missing predictions are counted, not converted to a zero-length peak."""
    data, work = tmp_path / "raw", tmp_path / "work"
    data.mkdir()
    work.mkdir()
    config = tmp_path / "config.yaml"
    config.write_text(f"paths:\n  data_root: {data}\n  work_root: {work}\n")
    source = work / "predictions.tsv"
    rows = [
        {"status": "no_candidate", "poly_type": "", "predicted_length_nt": ""},
        {"status": "input_failure", "poly_type": "polyA", "predicted_length_nt": ""},
    ]
    if has_prediction:
        rows.append(
            {
                "status": "candidate_predicted",
                "poly_type": "polyA",
                "predicted_length_nt": "24.5",
            }
        )
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    output = work / "plots"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_cdna_predictions.py",
            "--config",
            str(config),
            "--predictions",
            str(source),
            "--output",
            str(output),
        ],
    )
    script = Path(__file__).resolve().parents[3] / "scripts/plot_cdna_predictions.py"
    run_path(str(script), run_name="__main__")
    summary = json.loads((output / "plot_summary.json").read_text())
    assert summary["selected_primary_bam_reads"] == len(rows)
    assert summary["predicted_candidates"] == int(has_prediction)
    assert summary["poly_types"] == (["polyA"] if has_prediction else [])
    assert "polyT" not in (output / "length_distribution.svg").read_text()
    for suffix in ("png", "pdf", "svg"):
        assert (output / f"length_distribution.{suffix}").stat().st_size > 0
