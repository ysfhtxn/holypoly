"""Plot training summaries or exported predictions without loading signal/weights."""

import argparse
import csv
import json
import math
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from holypoly.core.config import load_config


def _load_predictions(path: Path, supervision_mode: str = "all") -> tuple[list, list]:
    """Read TSV; retain failed rows in counts and finite weak labels for plots."""
    required = {
        "read_id",
        "split",
        "nominal_length_nt",
        "regression_length_nt",
        "poly_type",
        "supervision_mode",
        "label_source",
        "label_status",
        "prediction_status",
    }
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("prediction table is missing required columns")
        rows = list(reader)
    if supervision_mode != "all":
        rows = [row for row in rows if row["supervision_mode"] == supervision_mode]
    if not rows:
        raise ValueError("no prediction rows match the selection")
    if len({row["read_id"] for row in rows}) != len(rows):
        raise ValueError("duplicate read_id in prediction table")
    if len({row["split"] for row in rows}) != 1:
        raise ValueError("plot one split at a time")
    valid = []
    for row in rows:
        if row["poly_type"] not in {"polyA", "polyT", "none"}:
            raise ValueError(f"unknown poly_type: {row['poly_type']}")
        nominal = float(row["nominal_length_nt"] or "nan")
        prediction = float(row["regression_length_nt"] or "nan")
        row["nominal_length_nt"] = nominal if math.isfinite(nominal) else None
        row["regression_length_nt"] = prediction if math.isfinite(prediction) else None
        if (
            row["prediction_status"] == "ok"
            and row["label_status"] != "failure"
            and math.isfinite(nominal)
            and math.isfinite(prediction)
        ):
            valid.append(row)
    if not valid:
        raise ValueError("no finite nominal-length/prediction pairs to plot")
    return rows, valid


def _plot_predictions(path: Path, output: Path, plt, *, mode: str, seed: int) -> None:
    """Plot all pairs, grouped distributions and signed errors against nominal nt.

    Boxes show median/IQR with 1.5-IQR whiskers. All observations, including
    outliers, are shown as dots. Seeded horizontal jitter affects display only.
    """
    rows, valid = _load_predictions(path, mode)
    colors = {"none": "#999999", "polyA": "#7397B5", "polyT": "#D0A275"}
    kinds = [kind for kind in colors if any(row["poly_type"] == kind for row in rows)]
    lengths = sorted(
        {
            row["nominal_length_nt"]
            for row in rows
            if row["nominal_length_nt"] is not None
        }
    )
    counts = Counter((row["nominal_length_nt"], row["poly_type"]) for row in rows)
    grouped = defaultdict(list)
    for row in valid:
        grouped[(row["nominal_length_nt"], row["poly_type"])].append(
            row["regression_length_nt"]
        )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2))
    count_ax, distribution_ax, scatter_ax, error_ax = axes.flat
    width = 0.72 / len(kinds)
    rng = random.Random(seed)
    for kind_index, kind in enumerate(kinds):
        offset = (kind_index - (len(kinds) - 1) / 2) * width
        positions = [index + offset for index in range(len(lengths))]
        count_ax.bar(
            positions,
            [counts[(length, kind)] for length in lengths],
            width=width * 0.88,
            color=colors[kind],
            label=kind,
        )
        selected = [row for row in valid if row["poly_type"] == kind]
        if selected:
            scatter_ax.scatter(
                [row["nominal_length_nt"] for row in selected],
                [row["regression_length_nt"] for row in selected],
                s=10,
                alpha=0.4,
                color=colors[kind],
                linewidths=0,
            )
        for index, length in enumerate(lengths):
            values = grouped[(length, kind)]
            if not values:
                continue
            position = positions[index]
            for axis, data in (
                (distribution_ax, values),
                (error_ax, [value - length for value in values]),
            ):
                axis.boxplot(
                    [data],
                    positions=[position],
                    widths=width * 0.76,
                    patch_artist=True,
                    showfliers=False,
                    manage_ticks=False,
                    boxprops={"facecolor": colors[kind], "alpha": 0.5},
                    medianprops={"color": "#333333"},
                    whiskerprops={"color": colors[kind]},
                    capprops={"color": colors[kind]},
                )
                axis.scatter(
                    [position + rng.uniform(-width * 0.3, width * 0.3) for _ in data],
                    data,
                    s=5,
                    alpha=0.3,
                    color=colors[kind],
                    linewidths=0,
                )
    distribution_ax.plot(
        range(len(lengths)),
        lengths,
        "k--",
        linewidth=1,
        marker="D",
        markersize=3,
        label="Nominal target",
    )
    limit = max(max(lengths), max(row["regression_length_nt"] for row in valid)) * 1.08
    scatter_ax.plot(
        [0, limit], [0, limit], "k--", linewidth=1, label="Prediction = nominal"
    )
    scatter_ax.set_xlim(-max(limit, 1) * 0.03, max(limit, 1))
    scatter_ax.set_ylim(0, max(limit, 1))
    error_ax.axhline(0, color="#333333", linestyle="--", linewidth=1)
    for axis in (count_ax, distribution_ax, error_ax):
        axis.set_xticks(range(len(lengths)), [f"{length:g}" for length in lengths])
        axis.set_xlim(-0.6, len(lengths) - 0.4)
    for axis, title, ylabel in (
        (count_ax, "a  Sample counts (all exported rows)", "Number of reads"),
        (
            distribution_ax,
            "b  Predicted length distributions",
            "Raw regression length (nt)",
        ),
        (scatter_ax, "c  Agreement with nominal length", "Raw regression length (nt)"),
        (error_ax, "d  Signed length deviations", "Prediction - nominal (nt)"),
    ):
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
        axis.set_xlabel("Nominal poly length (nt)")
        axis.set_ylabel(ylabel)
    fig.legend(
        *count_ax.get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=len(kinds),
        fontsize=9,
    )
    distribution_ax.legend(fontsize=8)
    scatter_ax.legend(fontsize=8)
    split = rows[0]["split"]
    fig.suptitle(
        f"{split.capitalize()} predictions | {len(valid)}/{len(rows)} reads plotted"
        f" | supervision: {mode}",
        fontsize=12,
    )
    fig.text(
        0.5,
        0.022,
        "Construct-level weak labels; boxes: median / IQR; "
        "dots: all finite labeled predictions.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.045, 1, 0.90), w_pad=2.5, h_pad=2.5)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"{split}_summary.{extension}", dpi=220)
    plt.close(fig)
    summary = {
        "source_predictions": str(path),
        "split": split,
        "selected_rows": len(rows),
        "plotted_pairs": len(valid),
        "excluded_pairs": len(rows) - len(valid),
        "supervision_filter": mode,
        "prediction_status_counts": dict(
            Counter(row["prediction_status"] for row in rows)
        ),
        "supervision_counts": dict(Counter(row["supervision_mode"] for row in rows)),
        "label_sources": dict(Counter(row["label_source"] for row in rows)),
        "rows_without_nominal_length": sum(
            row["nominal_length_nt"] is None for row in rows
        ),
        "jitter_seed": seed,
        "regression_field": "regression_length_nt",
        "notes": (
            "Nominal construction labels are not independently verified per-read "
            "truth. No confidence intervals or hypothesis tests; one model run. "
            "No prediction points were subsampled. Unknown or failed rows remain "
            "in source data and counts."
        ),
    }
    (output / "plot_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    source_copy = output / "source_predictions.tsv"
    if source_copy.resolve() != path.resolve():
        shutil.copyfile(path, source_copy)
    shutil.copyfile(__file__, output / "plot_training.py")
    print(json.dumps({"output": str(output), "plotted_pairs": len(valid)}))


def main() -> None:
    """Export logged values and descriptive figures beneath the work root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--data-root", type=Path)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--log", type=Path)
    inputs.add_argument("--predictions", type=Path)
    parser.add_argument(
        "--supervision-mode", choices=["all", "full", "length_only"], default="all"
    )
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(
        args.config, cli_data_root=args.data_root, cli_work_root=args.work_root
    )
    if config.paths.work_root is None or config.paths.data_root is None:
        parser.error("data_root and work_root must be configured")
    work_root = config.paths.work_root.expanduser().resolve()
    data_root = config.paths.data_root.expanduser().resolve()
    if not work_root.is_dir() or not data_root.is_dir():
        parser.error("data_root and work_root must exist")
    source = (work_root / (args.log or args.predictions)).expanduser().resolve()
    output = (work_root / args.output).resolve()
    if (
        not output.is_relative_to(work_root)
        or output.is_relative_to(data_root)
        or work_root.is_relative_to(data_root)
        or output.is_relative_to(Path(__file__).resolve().parents[1])
        or output == work_root
    ):
        raise ValueError("Output must be an external work-root child outside raw data")
    notes = (
        "Construct nominal lengths are weak labels, not independently verified "
        "per-read truth. Train metrics aggregate changing model weights; "
        "validation uses the epoch-end model. Location/status metrics apply "
        "only to supervised labels. Log-length loss is not measured in nt. "
        "One epoch does not establish convergence. No confidence intervals: "
        "one run, one seed."
    )
    os.environ["MPLCONFIGDIR"] = str(work_root / "cache" / "matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    if args.predictions is not None:
        _plot_predictions(
            source, output, plt, mode=args.supervision_mode, seed=args.seed
        )
        return
    rows = [
        json.loads(line) for line in source.read_text().splitlines() if line.strip()
    ]
    epoch = max(row["epoch"] for row in rows)
    selected = {row["split"]: row for row in rows if row["epoch"] == epoch}
    splits = ["train", "validation"]
    if any(split not in selected for split in splits):
        raise ValueError("Both train and validation summaries are required")
    colors = ["#7397B5", "#D0A275"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    metrics = ["location_loss", "length_loss", "status_loss", "total_loss"]
    labels = ["Location", "Log-length", "Status", "Total"]
    for index, split in enumerate(splits):
        values = [selected[split]["losses"][metric] for metric in metrics]
        positions = [x + (index - 0.5) * 0.36 for x in range(4)]
        bars = axes[0].bar(
            positions, values, width=0.36, label=split.capitalize(), color=colors[index]
        )
        axes[0].bar_label(
            bars, labels=[f"{v:.2f}" for v in values], fontsize=7, padding=3
        )
    axes[0].set_xticks(range(4), labels)
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Epoch-average loss")
    axes[0].set_ylim(0, max(selected[s]["losses"]["total_loss"] for s in splits) * 1.24)
    axes[0].set_title("a  Auxiliary losses", loc="left", fontweight="bold")
    axes[0].legend(fontsize=8)
    values = [selected[s]["length_mae_nt"] for s in splits]
    bars = axes[1].bar(range(2), values, width=0.55, color=colors)
    axes[1].bar_label(bars, labels=[f"{v:.2f}" for v in values], padding=4)
    axes[1].set_xticks(range(2), ["Train (online)", "Validation"])
    axes[1].set_ylim(0, max(values) * 1.24)
    axes[1].set_ylabel("Mean absolute deviation from\nnominal length (nt)")
    axes[1].set_title("b  Weak-label length agreement", loc="left", fontweight="bold")
    fig.suptitle(
        f"Epoch {epoch} | train n={selected['train']['examples']:,}; "
        f"validation n={selected['validation']['examples']:,}",
        fontsize=11,
    )
    fig.text(
        0.5,
        0.025,
        "Construct-level weak supervision; one run; no convergence claim.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.93), w_pad=2.5)
    output.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"epoch{epoch}_summary.{extension}", dpi=220)
    plt.close(fig)
    (output / "source_metrics.json").write_text(
        json.dumps(
            {
                "source_log": str(source),
                "records": list(selected.values()),
                "notes": notes,
            },
            indent=2,
        )
        + "\n"
    )
    shutil.copyfile(__file__, output / "plot_training.py")
    print(json.dumps({"output": str(output), "epoch": epoch}))


if __name__ == "__main__":
    main()
