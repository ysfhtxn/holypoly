"""Plot separate and joint descriptive poly(A) distributions from three tools."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from holypoly.core.config import load_config  # noqa: E402
from holypoly.evaluation.cdna import CATEGORIES  # noqa: E402
from holypoly.evaluation.three_cdna import TOOLS  # noqa: E402

COLORS = {"Holypoly": "#4477AA", "Dorado": "#CC9933", "PolyAnalysis": "#8866AA"}
STATE_COLORS = (None, "#E7C889", "#CBD0D4", "#B3A7BC", "#666A70")
STATE_NAMES = (
    "Positive output",
    "Native zero",
    "No candidate",
    "Other / ambiguous",
    "No estimate / error",
)


def render(output: Path, sample: str, bin_width: float, focus: float) -> None:
    """Export exact ECDFs, shared bins and full native-state denominators.

    Histograms show a labeled focus window; their denominator includes all
    positive lengths, including those outside that window. ECDFs retain every
    positive value using a labeled symlog x axis. No length is imputed as zero.
    """
    summary = json.loads((output / "summary.json").read_text())
    all_values: dict[str, list[float]] = {tool: [] for tool in TOOLS}
    for tool in TOOLS:
        with (output / f"{tool.lower()}_results.tsv").open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["positive_length_nt"]:
                    all_values[tool].append(float(row["positive_length_nt"]))
    shared: dict[str, list[float]] = {tool: [] for tool in TOOLS}
    paired: dict[str, list[float]] = {tool: [] for tool in TOOLS}
    common_count = 0
    with (output / "common_read_comparison.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            common_count += 1
            for tool in TOOLS:
                if row[f"{tool}_positive_length_nt"]:
                    shared[tool].append(float(row[f"{tool}_positive_length_nt"]))
            if all(row[f"{tool}_positive_length_nt"] for tool in TOOLS):
                for tool in TOOLS:
                    paired[tool].append(float(row[f"{tool}_positive_length_nt"]))
    if common_count != summary["common_reads"]:
        raise ValueError("shared table and summary disagree")
    for tool in TOOLS:
        if len(paired[tool]) != summary["common_positive_reads"]:
            raise ValueError("shared-positive count disagrees")
        if len(all_values[tool]) != summary["tools"][tool]["all_status_counts"].get(
            "positive_length", 0
        ):
            raise ValueError("all-positive count disagrees")
    for values in [*all_values.values(), *shared.values(), *paired.values()]:
        if any(not math.isfinite(value) or value <= 0 for value in values):
            raise ValueError("invalid positive length")
    maximum = max((max(v, default=0) for v in all_values.values()), default=0)
    number = max(1, math.ceil(max(maximum, focus) / bin_width))
    if number > 10000:
        raise ValueError("too many bins; increase bin-width-nt")
    bins = np.arange(number + 1) * bin_width
    contract = {
        "claim": "Describe native reported poly(A) outputs and their coverage, "
        "without ranking accuracy.",
        "archetype": "quantitative grid",
        "backend": "python/matplotlib",
        "evidence": [
            "three separate all-available populations",
            "overlaid all-available distributions",
            "shared-ID and shared-positive distributions "
            "with full native-state denominators",
        ],
        "histogram_bin_width_nt": bin_width,
        "histogram_focus_nt": [0, focus],
        "histogram_denominator": "all positive outputs, including outside focus",
        "ecdf": "all values, exact unique-value ECDF, "
        "symlog x (linear threshold 20 nt)",
        "central_tendency": {
            "population": "each plotted tool/cohort's full positive-output subset, "
            "including lengths outside the histogram focus",
            "labels": "mean and median in histogram and ECDF legends, 2 decimals",
            "histogram_markers": "tool-colored dashed mean and dotted median lines",
        },
        "formats": ["PNG300dpi", "editable PDF", "editable-text SVG"],
        "source_data": [
            "*_results.tsv",
            "common_read_comparison.tsv",
            "histogram_source_data.tsv",
        ],
        "limitations": summary["limitations"],
    }
    (output / "figure_contract.json").write_text(json.dumps(contract, indent=2) + "\n")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    hist_rows, stats = [], []
    for cohort, values_by_tool in (
        ("all_available", all_values),
        ("common_ids", shared),
        ("common_positive", paired),
    ):
        for tool, values in values_by_tool.items():
            values_array = np.asarray(values)
            hist, _ = np.histogram(values_array, bins=bins)
            if int(hist.sum()) != len(values):
                raise ValueError("histogram does not retain all positive values")
            for left, right, count in zip(bins[:-1], bins[1:], hist, strict=True):
                hist_rows.append(
                    {
                        "cohort": cohort,
                        "tool": tool,
                        "bin_left_nt": left,
                        "bin_right_nt": right,
                        "count": int(count),
                        "percent_of_all_positive": 100 * count / len(values)
                        if values
                        else 0,
                    }
                )
            row = {
                "cohort": cohort,
                "tool": tool,
                "positive_reads": len(values),
                "above_focus_reads": int((values_array > focus).sum()),
            }
            for label, function in (
                ("mean_nt", np.mean),
                ("median_nt", np.median),
                ("min_nt", np.min),
                ("max_nt", np.max),
            ):
                row[label] = float(function(values_array)) if values else None
            for label, fraction in (
                ("q25_nt", 0.25),
                ("q75_nt", 0.75),
                ("p95_nt", 0.95),
            ):
                row[label] = (
                    float(np.quantile(values_array, fraction)) if values else None
                )
            stats.append(row)

    def save_table(name, rows):
        """Write exact plotting source data and descriptive summaries."""
        with (output / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    save_table("histogram_source_data.tsv", hist_rows)
    save_table("length_summary.tsv", stats)

    def center_label(values):
        """Describe the complete positive subset shown by a plotted series."""
        return f"Mean={np.mean(values):.2f}; Median={np.median(values):.2f}"

    def histogram(axis, series):
        """Overlay normalized native outputs using identical bin edges."""
        for tool, values in series.items():
            if values:
                axis.hist(
                    values,
                    bins=bins,
                    weights=np.full(len(values), 100 / len(values)),
                    histtype="step",
                    linewidth=1.5,
                    color=COLORS[tool],
                    label=f"{tool}: n={len(values):,}; "
                    f">{focus:g}: {sum(x > focus for x in values):,}\n"
                    f"{center_label(values)}",
                )
                for center, linestyle in (
                    (np.mean(values), "--"),
                    (np.median(values), ":"),
                ):
                    axis.axvline(
                        center,
                        color=COLORS[tool],
                        linestyle=linestyle,
                        linewidth=1,
                        alpha=0.85,
                    )
        axis.set(
            xlim=(0, focus),
            xlabel="Native length / proxy (nt)",
            ylabel="All positive outputs per bin (%)",
        )
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            handles.extend(
                Line2D([], [], color="#555555", linestyle=style, linewidth=1)
                for style in ("--", ":")
            )
            axis.legend(
                handles, [*labels, "Mean (dashed)", "Median (dotted)"], fontsize=7
            )

    def ecdf(axis, series):
        """Plot the exact full-range ECDF without clipping or downsampling."""
        for tool, values in series.items():
            if values:
                xs, counts = np.unique(values, return_counts=True)
                axis.step(
                    np.r_[0, xs],
                    np.r_[0, counts.cumsum() / len(values)],
                    where="post",
                    color=COLORS[tool],
                    linewidth=1.4,
                    label=f"{tool} (n={len(values):,})\n{center_label(values)}",
                )
        axis.set_xscale("symlog", linthresh=20)
        axis.set(
            xlim=(0, None),
            ylim=(0, 1.02),
            xlabel="Native length / proxy (nt; symlog scale)",
            ylabel="Cumulative fraction of positive outputs",
        )
        if axis.lines:
            axis.legend(fontsize=7, loc="lower right" if len(series) > 1 else "best")
        else:
            axis.text(
                0.5, 0.5, "No positive outputs", ha="center", transform=axis.transAxes
            )

    def state_bars(axis, tools, common):
        """Keep all zero, absent, other and failed outputs in their denominators."""
        for index, tool in enumerate(tools):
            counts = summary["tools"][tool][
                "common_status_counts" if common else "all_status_counts"
            ]
            total = common_count if common else summary["tools"][tool]["all_reads"]
            if sum(counts.values()) != total:
                raise ValueError("state denominator mismatch")
            left = 0.0
            for category, color in zip(CATEGORIES, STATE_COLORS, strict=True):
                width = 100 * counts.get(category, 0) / total if total else 0
                axis.barh(
                    index, width, left=left, color=color or COLORS[tool], height=0.55
                )
                if width >= 12:
                    axis.text(
                        left + width / 2,
                        index,
                        f"{width:.1f}%",
                        ha="center",
                        va="center",
                        fontsize=7,
                    )
                left += width
        axis.set(
            yticks=range(len(tools)),
            yticklabels=tools,
            xlim=(0, 100),
            xlabel="Input reads (%)",
        )
        axis.invert_yaxis()
        axis.legend(
            handles=[
                Patch(facecolor=color or "#777777", label=name)
                for color, name in zip(STATE_COLORS, STATE_NAMES, strict=True)
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, -0.22),
            ncol=2,
            fontsize=6.5,
        )

    stems = []

    def export(figure, stem):
        """Export white-background PNG/PDF/SVG with editable vector text."""
        for suffix in ("png", "pdf", "svg"):
            figure.savefig(output / f"{stem}.{suffix}", dpi=300, facecolor="white")
        plt.close(figure)
        stems.append(stem)

    for tool in TOOLS:
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.7))
        fig.subplots_adjust(left=0.06, right=0.985, top=0.80, bottom=0.30, wspace=0.45)
        histogram(axes[0], {tool: all_values[tool]})
        ecdf(axes[1], {tool: all_values[tool]})
        state_bars(axes[2], [tool], False)
        for axis, title in zip(
            axes,
            (
                "a  Length distribution (focus window)",
                "b  Full-range empirical CDF",
                "c  Native output states",
            ),
            strict=True,
        ):
            axis.set_title(title)
        fig.suptitle(
            f"{sample}: {tool} poly(A) outputs — "
            f"all {summary['tools'][tool]['all_reads']:,} available reads",
            fontsize=12,
        )
        note = (
            "PolyAnalysis: partial sequential cohort; "
            "length is fixed samples/30, uncalibrated."
            if tool == "PolyAnalysis"
            else "Native reported outputs; completeness and biological tail "
            "identity are unverified."
        )
        fig.text(0.06, 0.12, note, fontsize=8)
        fig.text(
            0.06,
            0.065,
            "Positive-output distributions only. Native zero / no candidate / "
            "other / error are retained in c. No truth-based accuracy.",
            fontsize=8,
        )
        export(fig, f"{tool.lower()}_polya_distribution")

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    fig.subplots_adjust(left=0.06, right=0.985, top=0.81, bottom=0.31, wspace=0.45)
    histogram(axes[0], all_values)
    ecdf(axes[1], all_values)
    state_bars(axes[2], list(TOOLS), False)
    for axis, title in zip(
        axes,
        (
            "a  Normalized lengths (focus window)",
            "b  Full-range empirical CDF",
            "c  Output coverage",
        ),
        strict=True,
    ):
        axis.set_title(title)
    fig.suptitle(
        f"{sample}: three-tool comparison — all available outputs", fontsize=12
    )
    fig.text(
        0.06,
        0.12,
        "Tools use different read populations; normalized distributions "
        "do not establish relative accuracy or biological recall.",
        fontsize=8,
    )
    fig.text(
        0.06,
        0.065,
        "PolyAnalysis: partial sequential cohort and uncalibrated samples/30 proxy. "
        "Missing/native-zero results are never imputed as tails.",
        fontsize=8,
    )
    export(fig, "three_tools_all_available")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8.4))
    fig.subplots_adjust(
        left=0.08, right=0.98, bottom=0.24, top=0.87, hspace=0.52, wspace=0.30
    )
    histogram(axes[0, 0], shared)
    ecdf(axes[0, 1], shared)
    ecdf(axes[1, 0], paired)
    state_bars(axes[1, 1], list(TOOLS), True)
    for axis, title in zip(
        axes.flat,
        (
            "a  Shared IDs: tool-specific positive outputs",
            "b  Shared IDs: full-range CDF",
            f"c  All three positive (n={summary['common_positive_reads']:,})",
            "d  Every shared read: native states",
        ),
        strict=True,
    ):
        axis.set_title(title)
    fig.suptitle(
        f"{sample}: three tools on {common_count:,} shared read IDs", fontsize=12
    )
    notes = [
        "a/b: identical input read IDs, but tool-specific positive subsets. "
        "c: identical IDs positive in all three tools.",
        "Histograms: same 5-nt bins; focus window only. "
        "All values are retained in full-range symlog CDFs and source tables.",
        "PolyAnalysis: uncalibrated samples/30 proxy from a partial sequential "
        "cohort. Native zero is not verified biological zero.",
        "Unlabeled cDNA: descriptive distributions and output coverage only; "
        "no accuracy, biological recall or tool ranking.",
    ]
    notes[1] = notes[1].replace("5-nt", f"{bin_width:g}-nt")
    for index, note in enumerate(notes):
        fig.text(0.08, 0.12 - index * 0.025, note, fontsize=8)
    export(fig, "three_tools_common_reads")
    qa = {
        **contract,
        "common_reads": common_count,
        "common_positive_reads": summary["common_positive_reads"],
        "histogram_percent_sums_100_per_nonempty_tool": True,
        "full_range_values_retained_in_ecdf": True,
        "zero_imputation": False,
        "accuracy_computed": False,
        "figure_stems": stems,
        "visual_review": "pending",
    }
    (output / "plot_qa.json").write_text(json.dumps(qa, indent=2) + "\n")


def main() -> None:
    """Extract on the existing isolated environment, then render with Python."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    for name in ("holypoly", "dorado-bam", "polyanalysis", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--sample-name", default="cDNA")
    parser.add_argument("--bin-width-nt", type=float, default=5)
    parser.add_argument("--focus-max-nt", type=float, default=300)
    args = parser.parse_args()
    if any(
        not math.isfinite(x) or x <= 0 for x in (args.bin_width_nt, args.focus_max_nt)
    ):
        raise ValueError("bin width and focus maximum must be finite and positive")
    project = Path(__file__).resolve().parents[1]
    python = project / "src/bonito/.venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError("existing Bonito environment required for pysam")
    config = load_config(args.config)
    if config.paths.work_root is None:
        raise ValueError("configure work_root")
    output = (config.paths.work_root / args.output).resolve()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(project / "src"))
    env["MPLCONFIGDIR"] = str(output / "matplotlib-cache")
    command = [str(python), "-m", "holypoly.evaluation.three_cdna"]
    for name in ("config", "holypoly", "dorado_bam", "polyanalysis", "output"):
        command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    subprocess.run(command, env=env, check=True)
    os.environ["MPLCONFIGDIR"] = str(output / "matplotlib-cache")
    render(output, args.sample_name, args.bin_width_nt, args.focus_max_nt)
    print(f"Finished: {output}")


if __name__ == "__main__":
    main()
