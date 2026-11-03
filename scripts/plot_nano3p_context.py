"""Compare length distributions across separate fixed-len and published cohorts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from holypoly.core.config import load_config  # noqa: E402
from holypoly.evaluation.fixed_len import TOOLS, fingerprint  # noqa: E402

COLORS = {"Holypoly": "#4477AA", "Dorado": "#CC9933", "PolyAnalysis": "#AA6677"}
PUBLISHED_COLOR = "#228877"


def plot_context(local_csv: Path, published_csv: Path, output: Path) -> list[dict]:
    """Draw conditional positive lengths without pooling or mapping source labels.

    Published zeros remain in the CSV and count table. A numeric cDNA group
    suffix determines display spacing only, not an asserted per-read true length.
    """
    local = defaultdict(list)
    lengths = set()
    with local_csv.open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            length = int(row["nominal_length_nt"])
            lengths.add(length)
            if row["positive_length_nt"]:
                value = float(row["positive_length_nt"])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError("invalid local positive length")
                local[row["tool"], length].append(value)
    published = defaultdict(list)
    with published_csv.open() as stream:
        reader = csv.DictReader(stream)
        if set(reader.fieldnames or []) != {"Group", "Std_Type", "tail_length"}:
            raise ValueError("expected Fig3c Group, Std_Type, tail_length columns")
        for row in reader:
            value = float(row["tail_length"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid published tail_length")
            kind, group = row["Std_Type"], row["Group"]
            if kind not in {"cDNA", "RNA"}:
                raise ValueError("unexpected standard type")
            if kind == "cDNA" and not re.fullmatch(r"cDNA_\d+", group):
                raise ValueError("expected numeric cDNA group suffix")
            published[kind, group].append(value)
    if not local or not published:
        raise ValueError("both cohorts must contain lengths")
    figure, axes = plt.subplots(
        1, 3, figsize=(15, 7), gridspec_kw={"width_ratios": [1.8, 1.25, 0.6]}
    )
    figure.subplots_adjust(left=0.055, right=0.985, top=0.79, bottom=0.34, wspace=0.32)
    figure.suptitle(
        "Poly(A) lengths: fixed-len tool comparison and Nano3P-seq source data",
        fontsize=14,
        y=0.975,
    )
    figure.legend(
        handles=[
            Patch(facecolor=COLORS[t], label=t + ("*" if t == "PolyAnalysis" else ""))
            for t in TOOLS
        ]
        + [Patch(facecolor=PUBLISHED_COLOR, label="Nano3P-seq (different dataset)")],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.93),
        ncol=4,
        fontsize=10,
    )
    records = []

    def box(axis, values, position, color, width):
        """Show median/IQR, 1.5-IQR whiskers and all outliers of positive values."""
        if not values:
            return
        axis.boxplot(
            [values],
            positions=[position],
            widths=width,
            manage_ticks=False,
            patch_artist=True,
            showfliers=True,
            boxprops={"facecolor": color, "alpha": 0.6, "edgecolor": color},
            medianprops={"color": "#202020", "linewidth": 1.1},
            whiskerprops={"color": color},
            capprops={"color": color},
            flierprops={
                "marker": ".",
                "markersize": 2,
                "markeredgewidth": 0,
                "markerfacecolor": color,
                "alpha": 0.2,
            },
        )

    def record(cohort, tool, group, values, total=None, zero=None):
        """Retain group sizes and descriptive positive-output summaries."""
        q = np.percentile(values, [25, 50, 75]) if values else [None] * 3
        records.append(
            {
                "cohort": cohort,
                "tool": tool,
                "group": group,
                "reported_rows": total,
                "zero_rows": zero,
                "positive_rows": len(values),
                "q25_nt": q[0],
                "median_nt": q[1],
                "q75_nt": q[2],
            }
        )

    ordered_lengths = sorted(lengths)
    for tool, offset in zip(TOOLS, (-4, 0, 4), strict=True):
        for length in ordered_lengths:
            values = local[tool, length]
            box(axes[0], values, length + offset, COLORS[tool], 3.2)
            record("local_fixed_len", tool, str(length), values)
    axes[0].plot(ordered_lengths, ordered_lengths, "--", color="#555555", lw=1)
    axes[0].set(
        xticks=ordered_lengths,
        xlim=(-10, max(lengths) + 10),
        xlabel="Nominal construct length (nt)",
        title="a  Fixed-len: three tools, same holdout",
    )
    cdna = sorted(
        (g for kind, g in published if kind == "cDNA"),
        key=lambda g: int(g.split("_")[-1]),
    )
    rna = sorted(g for kind, g in published if kind == "RNA")
    for axis, kind, groups in ((axes[1], "cDNA", cdna), (axes[2], "RNA", rna)):
        positions = (
            [int(g.split("_")[-1]) for g in groups]
            if kind == "cDNA"
            else list(range(len(groups)))
        )
        for group, position in zip(groups, positions, strict=True):
            reported = published[kind, group]
            values = [v for v in reported if v > 0]
            box(axis, values, position, PUBLISHED_COLOR, 5 if kind == "cDNA" else 0.5)
            record(
                "published_Fig3c",
                "Nano3P-seq",
                group,
                values,
                len(reported),
                sum(v == 0 for v in reported),
            )
            axis.text(
                position,
                0.98,
                f"n+={len(values):,}",
                rotation=90,
                va="top",
                ha="center",
                fontsize=8,
                transform=axis.get_xaxis_transform(),
            )
        axis.set(
            xticks=positions,
            xticklabels=[g.removeprefix("cDNA_") for g in groups],
            xlabel="Source group suffix (cDNA_<value>)"
            if kind == "cDNA"
            else "Source RNA group",
        )
        if kind == "cDNA":
            axis.set_xlim(min(positions) - 10, max(positions) + 10)
    axes[1].set_title("b  Nano3P-seq: cDNA standards")
    axes[2].set_title("c  Nano3P-seq: RNA")
    for axis in axes:
        axis.set_ylabel("Positive reported length (nt)")
        axis.set_ylim(bottom=0)
        axis.grid(axis="y", color="#dddddd", lw=0.5)
        axis.set_axisbelow(True)
        axis.title.set_fontsize(10)
        axis.title.set_fontweight("bold")
    notes = (
        "Separate datasets and protocols; distributions are descriptive context, "
        "not a matched four-tool benchmark or an accuracy/recall ranking.",
        "All panels use positive numeric outputs only. Boxes: median/IQR; "
        "whiskers: 1.5 IQR; dots: all outliers. Y ranges differ across panels.",
        "a: actual nominal nt spacing, tool offsets -4/0/+4 nt. "
        "b: numeric spacing from source group names, with no mapping to local labels.",
        "Nano3P-seq zeros are preserved in CSV and group counts; "
        "n+ counts positive rows within the published table, not all attempted reads.",
        "R1/R2 have no nominal length assigned here. "
        "*PolyAnalysis uses an uncalibrated interval/30 proxy; "
        "local labels are nominal weak labels.",
    )
    for index, note in enumerate(notes):
        figure.text(0.055, 0.235 - 0.032 * index, note, fontsize=8)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output / f"fixed_len_with_nano3p_context.{suffix}",
            dpi=300,
            facecolor="white",
        )
    plt.close(figure)
    with (output / "group_distribution_summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    return records


def main() -> None:
    """Render source-separated plots in a new safe external work directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    parser.add_argument("--local-comparison", type=Path, required=True)
    parser.add_argument("--fig3c-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.paths.work_root is None or config.paths.data_root is None:
        raise ValueError("configure external roots")
    work, data = config.paths.work_root.resolve(), config.paths.data_root.resolve()
    inputs = [
        (work / p).resolve(strict=True) for p in (args.local_comparison, args.fig3c_csv)
    ]
    output = (work / args.output).resolve()
    if (
        output.exists()
        or output == work
        or not output.is_relative_to(work)
        or output.is_relative_to(data)
        or output.is_relative_to(Path(__file__).resolve().parents[1])
        or any(p.is_relative_to(output) for p in inputs)
    ):
        raise ValueError("output must be a new external work-root child")
    before = [fingerprint(p) for p in inputs]
    output.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    records = plot_context(*inputs, output)
    if [fingerprint(p) for p in inputs] != before:
        raise ValueError("source input changed")
    (output / "figure_contract_and_qa.json").write_text(
        json.dumps(
            {
                "claim": "Show positive-length distributions "
                "in explicitly separate cohorts",
                "archetype": "quantitative grid",
                "backend": "python/matplotlib",
                "panels": [
                    "local fixed-len three-tool comparison",
                    "published cDNA standards",
                    "published RNA standards",
                ],
                "source_unchanged": True,
                "inputs": before,
                "source_sheet": "Fig3c",
                "group_counts": records,
                "cohorts_pooled": False,
                "recall_or_accuracy_inferred": False,
                "numeric_suffix_use": "Display coordinates only, "
                "not per-read ground truth",
                "statistics": "Descriptive medians/IQR, 1.5-IQR whiskers, all outliers",
                "dimensions_inches": [15, 7],
                "exports": ["png 300 dpi", "pdf", "svg"],
                "no_gpu_model_or_raw_signal_used": True,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Finished: {output}")


if __name__ == "__main__":
    main()
