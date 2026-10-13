"""Verify plot selection and unknown labels without plotting dependencies."""

import csv
from pathlib import Path
from runpy import run_path

import pytest

_load_predictions = run_path(
    str(Path(__file__).resolve().parents[3] / "scripts" / "plot_training.py")
)["_load_predictions"]


def test_plot_inputs_keep_zero_and_failure_counts(tmp_path: Path) -> None:
    """Failed input stays counted; missing regression never becomes zero."""
    rows = [
        {
            "read_id": "zero",
            "split": "test",
            "nominal_length_nt": "0",
            "regression_length_nt": "2",
            "poly_type": "none",
            "supervision_mode": "full",
            "label_source": "synthetic",
            "label_status": "zero_nt",
            "prediction_status": "ok",
        },
        {
            "read_id": "weak",
            "split": "test",
            "nominal_length_nt": "20",
            "regression_length_nt": "15",
            "poly_type": "polyT",
            "supervision_mode": "length_only",
            "label_source": "synthetic",
            "label_status": "poly_present",
            "prediction_status": "ok",
        },
        {
            "read_id": "bad",
            "split": "test",
            "nominal_length_nt": "20",
            "regression_length_nt": "",
            "poly_type": "polyA",
            "supervision_mode": "full",
            "label_source": "synthetic",
            "label_status": "poly_present",
            "prediction_status": "input_failure",
        },
    ]
    path = tmp_path / "predictions.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    all_rows, valid = _load_predictions(path)
    assert len(all_rows) == 3
    assert [row["read_id"] for row in valid] == ["zero", "weak"]
    assert all_rows[-1]["regression_length_nt"] is None
    full, valid_full = _load_predictions(path, "full")
    assert len(full) == 2 and valid_full[0]["nominal_length_nt"] == 0
    _, weak = _load_predictions(path, "length_only")
    assert weak[0]["poly_type"] == "polyT"


def test_plot_inputs_reject_incompatible_schema(tmp_path: Path) -> None:
    """Do not silently plot a file with different meanings for length columns."""
    path = tmp_path / "other.tsv"
    path.write_text("read_id\tlength\nexample\t20\n")
    with pytest.raises(ValueError, match="required columns"):
        _load_predictions(path)
