"""Verify plot selection and unknown labels without plotting dependencies."""

import csv
import json
from pathlib import Path
from runpy import run_path

import pytest

_plot_module = run_path(
    str(Path(__file__).resolve().parents[3] / "scripts" / "plot_training.py")
)
_load_predictions = _plot_module["_load_predictions"]
_load_epoch_records = _plot_module["_load_epoch_records"]


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


def test_epoch_logs_preserve_values_and_sort_resumed_epochs(tmp_path: Path) -> None:
    """Reordered log lines must preserve each epoch, including zero-valued metrics."""
    rows = [
        {"epoch": epoch, "split": split, "status_accuracy": 0.0}
        for epoch in [3, 1, 2]
        for split in ["validation", "train"]
    ]
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    loaded = _load_epoch_records(path)
    assert [(row["epoch"], row["split"]) for row in loaded] == [
        (epoch, split) for epoch in [1, 2, 3] for split in ["train", "validation"]
    ]
    assert all(row["status_accuracy"] == 0.0 for row in loaded)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "contains no epoch"),
        ([{"epoch": 1, "split": "train"}], "both train and validation"),
        (
            [{"epoch": 1, "split": "train"}] * 2,
            "duplicate epoch/split",
        ),
    ],
)
def test_epoch_logs_reject_ambiguous_histories(
    tmp_path: Path, rows: list[dict], message: str
) -> None:
    path = tmp_path / "train.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match=message):
        _load_epoch_records(path)


@pytest.mark.parametrize(
    ("config", "label"),
    [
        ({}, "Log-length"),
        ({"length_loss": "huber"}, "Length Huber"),
        ({"length_loss": "mae"}, "Length MAE"),
    ],
)
def test_plot_labels_use_recorded_length_objective(config: dict, label: str) -> None:
    assert _plot_module["_length_loss_label"]([{"loss_config": config}]) == label


def test_joint_history_renders_and_preserves_reference_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic joint logs retain CRF values and missing online decode metrics."""
    work = tmp_path / "work"
    monkeypatch.setenv("MPLCONFIGDIR", str(work / "cache" / "matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        {
            "epoch": 1,
            "split": split,
            "examples": 4,
            "joint_config": {"lambda_basecall": 0.1},
            "loss_config": {"length_loss": "huber"},
            "length_mae_nt": 10.0,
            "sequence_examples": 2,
            "losses": {
                "total_loss": 5.2,
                "length_loss": 5.0,
                "location_loss": 0.0,
                "status_loss": 0.0,
                "basecalling_loss": 2.0,
            },
            **({"reference_edit_rate": 0.125} if split == "validation" else {}),
        }
        for split in ("train", "validation")
    ]
    output = work / "results"
    _plot_module["_plot_training_history"](
        rows, output, plt, source=work / "synthetic.jsonl"
    )
    with (output / "epoch_metrics.tsv").open() as stream:
        exported = list(csv.DictReader(stream, delimiter="\t"))
    assert exported[0]["reference_edit_rate"] == ""
    assert float(exported[1]["reference_edit_rate"]) == 0.125
    assert float(exported[1]["basecalling_loss"]) == 2.0
    assert (output / "training_curves.png").stat().st_size > 0
    assert (output / "loss_components.pdf").stat().st_size > 0


def test_status_history_renders_zero_and_twenty_nt_metrics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Classifier-only logs must render without inventing basecalling metrics."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [
        {
            "epoch": 1,
            "split": split,
            "examples": 4,
            "joint_config": {"schema": "frozen-status-calibration-v1"},
            "loss_config": {"length_loss": "huber"},
            "losses": {
                "total_loss": 0.5,
                "status_loss": 0.5,
                "location_loss": 0.0,
                "length_loss": 0.0,
            },
            "length_mae_nt": 40.0,
            "status_accuracy": 0.75,
            "zero_nt_recall": 0.5,
            "twenty_nt_present_recall": 1.0,
            "twenty_nt_false_zero_fraction": 0.0,
            "poly_present_recall": 1.0,
            "selection_score": 2 / 3,
        }
        for split in ("train", "validation")
    ]
    output = tmp_path / "figures"
    _plot_module["_plot_training_history"](
        rows, output, plt, source=tmp_path / "synthetic.jsonl"
    )
    with (output / "epoch_metrics.tsv").open() as stream:
        exported = list(csv.DictReader(stream, delimiter="\t"))
    assert float(exported[1]["zero_nt_recall"]) == 0.5
    assert float(exported[1]["twenty_nt_false_zero_fraction"]) == 0
    assert "basecalling_loss" not in exported[0]
    assert (output / "training_curves.png").stat().st_size > 0


@pytest.mark.parametrize("poly_types", [("polyA",), ("polyT",), ("polyA", "polyT")])
def test_prediction_plots_use_numeric_lengths_and_observed_types(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, poly_types: tuple[str, ...]
) -> None:
    """Inspect rendered axes: 50 nt spans 2.5 times 20 nt, without absent types."""
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure

    rows = [
        {
            "read_id": f"synthetic-{length}-{kind}-{index}",
            "split": "test",
            "nominal_length_nt": length,
            "regression_length_nt": length + index,
            "poly_type": kind,
            "supervision_mode": "length_only",
            "label_source": "synthetic",
            "label_status": "zero_nt" if length == 0 else "poly_present",
            "prediction_status": "ok",
        }
        for length in (0, 80, 100, 150)
        for kind in (("none",) if length == 0 else poly_types)
        for index in range(3)
    ]
    path = tmp_path / "predictions.tsv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    figures = []
    savefig = Figure.savefig

    def capture(figure, *args, **kwargs):
        """Keep the actual rendered figure for geometry and visible-label checks."""
        figures.append(figure)
        return savefig(figure, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture)
    output = tmp_path / "plots"
    _plot_module["_plot_predictions"](path, output, plt, mode="all", seed=17)
    figure = figures[-1]
    for axis in figure.axes:
        assert list(axis.get_xticks()) == [0, 80, 100, 150]
        at_80 = axis.transData.transform((80, 0))[0]
        at_100 = axis.transData.transform((100, 0))[0]
        at_150 = axis.transData.transform((150, 0))[0]
        assert (at_150 - at_100) / (at_100 - at_80) == pytest.approx(2.5)
    target = next(
        line for line in figure.axes[1].lines if line.get_label() == "Nominal target"
    )
    assert list(target.get_xdata()) == [0, 80, 100, 150]
    count_axis = figure.axes[0]
    handles, labels = count_axis.get_legend_handles_labels()
    assert labels == ["0 nt", *poly_types]
    assert len(count_axis.patches) == 1 + 3 * len(poly_types)
    if len(poly_types) == 1:
        bars = handles[1]
        assert [bar.get_x() + bar.get_width() / 2 for bar in bars] == pytest.approx(
            [80, 100, 150]
        )
        xlabel = f"Nominal {poly_types[0]} length (nt)"
        assert all(axis.get_xlabel() == xlabel for axis in figure.axes)
    visible_text = " ".join(
        [text.get_text() for text in figure.texts]
        + [text.get_text() for legend in figure.legends for text in legend.get_texts()]
        + [axis.get_xlabel() for axis in figure.axes]
    )
    for kind in ("polyA", "polyT"):
        assert (kind in visible_text) is (kind in poly_types)
    summary = json.loads((output / "plot_summary.json").read_text())
    assert summary["poly_types"] == list(poly_types)
    assert summary["length_axis"] == "numeric_nt"
    assert summary["plotted_pairs"] == len(rows)
    assert (output / "test_summary.svg").stat().st_size > 0
    assert (output / "test_summary.pdf").stat().st_size > 0
