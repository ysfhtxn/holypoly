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


def _load_epoch_records(path: Path) -> list[dict]:
    """Load paired epoch summaries without silently replacing duplicate entries."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("training log contains no epoch summaries")
    identities = [(row["epoch"], row["split"]) for row in rows]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate epoch/split summaries in training log")
    for epoch in {row["epoch"] for row in rows}:
        splits = {row["split"] for row in rows if row["epoch"] == epoch}
        if splits != {"train", "validation"}:
            raise ValueError(
                f"Epoch {epoch} requires both train and validation summaries"
            )
    return sorted(rows, key=lambda row: (row["epoch"], row["split"]))


def _length_loss_label(rows: list[dict]) -> str:
    """Name the recorded objective, retaining the legacy log-loss default."""
    mode = rows[0].get("loss_config", {}).get("length_loss", "log_smooth_l1")
    return {
        "log_smooth_l1": "Log-length",
        "mae": "Length MAE",
        "huber": "Length Huber",
    }.get(mode, "Length")


def _plot_training_history(
    rows: list[dict], output: Path, plt, *, source: Path
) -> None:
    """Plot unsmoothed epoch means; absent legacy metrics remain gaps, not zeros."""
    from matplotlib.ticker import MaxNLocator

    output.mkdir(parents=True, exist_ok=True)
    colors = {"train": "#7397B5", "validation": "#D0A275"}
    panels = [
        ("total_loss", "a  Total loss", "Epoch-average loss"),
        ("length_mae_nt", "b  Length agreement", "Mean absolute deviation (nt)"),
        ("location_accuracy", "c  Location accuracy", "Accuracy vs provisional labels"),
        ("status_accuracy", "d  Shared status accuracy", "Accuracy vs weak labels"),
        ("poly_iou", "e  Poly overlap", "Timestep-aggregated IoU"),
        ("location_macro_f1", "f  Location macro F1", "Macro F1 vs provisional labels"),
    ]
    joint = any(row.get("joint_config") for row in rows)
    calibration = any(
        row.get("joint_config", {}).get("schema") == "frozen-status-calibration-v1"
        for row in rows
    )
    if calibration:
        panels = [
            ("status_loss", "a  Balanced status loss", "Train-weighted cross entropy"),
            (
                "zero_nt_recall",
                "b  Zero-nt recall",
                "Fraction of zero labels called zero",
            ),
            (
                "twenty_nt_false_zero_fraction",
                "c  20-nt false zero",
                "Fraction called zero",
            ),
            (
                "twenty_nt_present_recall",
                "d  20-nt present recall",
                "Fraction called present",
            ),
            (
                "poly_present_recall",
                "e  All-positive recall",
                "Fraction called present",
            ),
            (
                "selection_score",
                "f  Validation selection score",
                "Harmonic mean of recalls",
            ),
        ]
    elif joint:
        panels = [
            ("total_loss", "a  Joint loss", "Epoch-average loss"),
            ("length_mae_nt", "b  Length agreement", "Weak-label deviation (nt)"),
            ("basecalling_loss", "c  Head CTC-CRF loss", "Native loss per sequence"),
            (
                "reference_edit_rate",
                "d  Head reference agreement",
                "Edits / reference bases",
            ),
        ]

    def draw(axis, metric: str, title: str, ylabel: str) -> None:
        """Keep actual epoch positions and distinguish train/validation throughout."""
        for split, color in colors.items():
            selected = [row for row in rows if row["split"] == split]
            values = [
                row["losses"].get(metric, math.nan)
                if metric.endswith("loss")
                else row.get(metric, math.nan)
                for row in selected
            ]
            axis.plot(
                [row["epoch"] for row in selected],
                values,
                color=color,
                marker="o",
                markersize=3,
                linewidth=1.5,
                label=("Train (epoch end)" if calibration else "Train (online)")
                if split == "train"
                else "Validation",
            )
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=6))
        axis.grid(axis="y", color="#dddddd", linewidth=0.5, alpha=0.6)
        if metric in {
            "location_accuracy",
            "status_accuracy",
            "poly_iou",
            "location_macro_f1",
            "zero_nt_recall",
            "twenty_nt_false_zero_fraction",
            "twenty_nt_present_recall",
            "poly_present_recall",
            "selection_score",
        }:
            axis.set_ylim(0, 1.02)
        else:
            axis.set_ylim(bottom=0)

    fig, axes = plt.subplots(len(panels) // 2, 2, figsize=(9.0, 3.0 * len(panels) / 2))
    for axis, (metric, title, ylabel) in zip(axes.flat, panels, strict=True):
        draw(axis, metric, title, ylabel)
    first, last = rows[0]["epoch"], rows[-1]["epoch"]
    fig.suptitle(f"Training history | epochs {first}\u2013{last}", fontsize=12)
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.965),
        ncol=2,
    )
    fig.text(
        0.5,
        0.018,
        (
            "Frozen encoder/regression; only existing status classifiers trained.\n"
            "Construct weak labels; no independent truth or failure-detection claim."
            if calibration
            else "One run; no smoothing or confidence intervals. "
            "Length uses nominal weak labels.\n"
            "Head edits use selected reference windows, not independent read truth."
            if joint
            else "One run; no smoothing or confidence intervals. "
            "Length uses nominal weak labels.\n"
            "Location/status cover supervised labels only; "
            "IoU precedes interval/status decoding."
        ),
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.935), h_pad=2.0, w_pad=2.5)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"training_curves.{extension}", dpi=300)
    plt.close(fig)

    components = ["location_loss", "length_loss", "status_loss"]
    titles = [
        "a  Location loss",
        f"b  {_length_loss_label(rows)} loss",
        "c  Shared status loss",
    ]
    if joint and not calibration:
        components.append("basecalling_loss")
        titles.append("d  Head CTC-CRF loss")
    fig, axes = plt.subplots(1, len(components), figsize=(3.4 * len(components), 3.5))
    for axis, metric, title in zip(
        axes,
        components,
        titles,
        strict=True,
    ):
        draw(axis, metric, title, "Epoch-average loss")
    fig.legend(
        *axes[0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
    )
    fig.text(
        0.5,
        0.025,
        "Status uses train-derived class weights; other objectives are frozen."
        if calibration
        else "Unweighted loss components; objective scales differ.",
        ha="center",
        fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.93), w_pad=2.5)
    for extension in ("png", "svg", "pdf"):
        fig.savefig(output / f"loss_components.{extension}", dpi=300)
    plt.close(fig)

    fields = [
        "epoch",
        "split",
        "examples",
        "total_loss",
        "location_loss",
        "length_loss",
        "status_loss",
        "length_mae_nt",
        "location_accuracy",
        "status_accuracy",
        "poly_iou",
        "location_macro_f1",
    ]
    if calibration:
        fields += [
            "zero_nt_recall",
            "twenty_nt_false_zero_fraction",
            "twenty_nt_present_recall",
            "poly_present_recall",
            "selection_score",
        ]
    elif joint:
        fields += ["basecalling_loss", "reference_edit_rate", "sequence_examples"]
    with (output / "epoch_metrics.tsv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: row["losses"].get(field, "")
                    if field.endswith("loss")
                    else row.get(field, "")
                    for field in fields
                }
            )
    (output / "source_training_history.json").write_text(
        json.dumps({"source_log": str(source), "records": rows}, indent=2) + "\n"
    )


def _plot_predictions(path: Path, output: Path, plt, *, mode: str, seed: int) -> None:
    """Plot all pairs, grouped distributions and signed errors against nominal nt.

    Boxes show median/IQR with 1.5-IQR whiskers. All observations, including
    outliers, are shown as dots. Seeded horizontal jitter affects display only.
    Length axes use numeric nt coordinates; only observed types are displayed.
    """
    rows, valid = _load_predictions(path, mode)
    colors = {"none": "#999999", "polyA": "#7397B5", "polyT": "#D0A275"}
    kinds = [kind for kind in colors if any(row["poly_type"] == kind for row in rows)]
    poly_types = [kind for kind in kinds if kind != "none"]
    poly_label = " / ".join(poly_types) if poly_types else "Poly"
    lengths = sorted(
        {
            row["nominal_length_nt"]
            for row in rows
            if row["nominal_length_nt"] is not None
        }
    )
    counts = Counter((row["nominal_length_nt"], row["poly_type"]) for row in rows)
    min_gap_nt = min(
        (right - left for left, right in zip(lengths, lengths[1:], strict=False)),
        default=20.0,
    )
    # Offsets and widths are in nt. Center groups containing only one type;
    # separate A/T only where both occur at the same nominal length.
    positions = {}
    widths = {}
    for length in lengths:
        present = [kind for kind in kinds if counts[(length, kind)]]
        width_nt = min_gap_nt * 0.72 / len(present)
        for index, kind in enumerate(present):
            positions[(length, kind)] = (
                length + (index - (len(present) - 1) / 2) * width_nt
            )
            widths[(length, kind)] = width_nt
    grouped = defaultdict(list)
    for row in valid:
        grouped[(row["nominal_length_nt"], row["poly_type"])].append(
            row["regression_length_nt"]
        )
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 7.2))
    count_ax, distribution_ax, scatter_ax, error_ax = axes.flat
    rng = random.Random(seed)
    for kind in kinds:
        kind_lengths = [length for length in lengths if counts[(length, kind)]]
        count_ax.bar(
            [positions[(length, kind)] for length in kind_lengths],
            [counts[(length, kind)] for length in kind_lengths],
            width=[widths[(length, kind)] * 0.88 for length in kind_lengths],
            color=colors[kind],
            label="0 nt" if kind == "none" else kind,
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
        for length in kind_lengths:
            values = grouped[(length, kind)]
            if not values:
                continue
            position = positions[(length, kind)]
            width = widths[(length, kind)]
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
        lengths,
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
        axis.set_xticks(lengths, [f"{length:g}" for length in lengths])
        axis.set_xlim(lengths[0] - min_gap_nt * 0.6, lengths[-1] + min_gap_nt * 0.6)
    scatter_ax.set_xticks(lengths, [f"{length:g}" for length in lengths])
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
        axis.set_xlabel(f"Nominal {poly_label} length (nt)")
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
        f"{split.capitalize()} {poly_label} predictions"
        f" | {len(valid)}/{len(rows)} reads plotted"
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
        "poly_types": poly_types,
        "length_axis": "numeric_nt",
        "nominal_lengths_nt": lengths,
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
    rows = _load_epoch_records(source)
    _plot_training_history(rows, output, plt, source=source)
    if rows[0].get("joint_config", {}).get("schema") == "frozen-status-calibration-v1":
        (output / "source_metrics.json").write_text(
            json.dumps({"source_log": str(source), "records": rows}, indent=2) + "\n"
        )
        shutil.copyfile(__file__, output / "plot_training.py")
        print(
            json.dumps({"output": str(output), "epoch": max(r["epoch"] for r in rows)})
        )
        return
    epoch = max(row["epoch"] for row in rows)
    selected = {row["split"]: row for row in rows if row["epoch"] == epoch}
    splits = ["train", "validation"]
    if any(split not in selected for split in splits):
        raise ValueError("Both train and validation summaries are required")
    colors = ["#7397B5", "#D0A275"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    metrics = ["location_loss", "length_loss", "status_loss", "total_loss"]
    labels = ["Location", _length_loss_label(rows), "Status", "Total"]
    if any(row.get("joint_config") for row in rows):
        metrics.insert(-1, "basecalling_loss")
        labels.insert(-1, "Head CRF")
    for index, split in enumerate(splits):
        values = [selected[split]["losses"][metric] for metric in metrics]
        positions = [x + (index - 0.5) * 0.36 for x in range(len(metrics))]
        bars = axes[0].bar(
            positions, values, width=0.36, label=split.capitalize(), color=colors[index]
        )
        axes[0].bar_label(
            bars, labels=[f"{v:.2f}" for v in values], fontsize=7, padding=3
        )
    axes[0].set_xticks(range(len(metrics)), labels)
    axes[0].tick_params(axis="x", labelsize=8)
    axes[0].set_ylabel("Epoch-average loss")
    axes[0].set_ylim(
        0,
        max(selected[s]["losses"][metric] for s in splits for metric in metrics) * 1.24,
    )
    axes[0].set_title("a  Loss components and total", loc="left", fontweight="bold")
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
