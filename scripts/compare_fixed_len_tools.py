"""Plot existing fixed-length outputs with auditable coverage and paired subsets.

Run in the parent environment. BAM tag extraction uses the existing Bonito
environment through a subprocess; no dependencies, models or services are added.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from holypoly.evaluation.fixed_len import (  # noqa: E402
    TOOLS,
    build_parser,
    resolve_paths,
)

COLORS = {"Holypoly": "#4477AA", "Dorado": "#CC9933", "PolyAnalysis": "#AA6677"}
MARKERS = {"Holypoly": "o", "Dorado": "s", "PolyAnalysis": "^"}
LABELS = {"Holypoly": "Holypoly", "Dorado": "Dorado", "PolyAnalysis": "PolyAnalysis*"}


def save_figure(figure: plt.Figure, output: Path, name: str) -> None:
    """Export editable vector files and a high-resolution PNG preview."""
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"{name}.{suffix}", dpi=300, facecolor="white")
    plt.close(figure)


def positive_rows(rows: list[dict[str, str]], tool: str) -> list[dict[str, str]]:
    """Select strictly positive native outputs, leaving zero/missing statuses intact."""
    return [row for row in rows if row["tool"] == tool and row["positive_length_nt"]]


def shared_ids(rows: list[dict[str, str]]) -> set[str]:
    """Return the same positive-estimate read intersection, excluding nominal zero."""
    return set.intersection(
        *[
            {
                row["read_id"]
                for row in positive_rows(rows, tool)
                if int(row["nominal_length_nt"]) > 0
            }
            for tool in TOOLS
        ]
    )


def overview(summary: dict, rows: list[dict[str, str]], output: Path) -> None:
    """Show conditional lengths, full-cohort coverage and same-read deviations."""
    figure, axes = plt.subplots(2, 2, figsize=(7.2, 6.8))
    figure.subplots_adjust(
        left=0.10, right=0.98, top=0.86, bottom=0.25, hspace=0.68, wspace=0.42
    )
    figure.suptitle(
        f"Fixed-length holdout: native tool outputs (n={summary['total_reads']:,})",
        fontsize=10,
        y=0.985,
    )
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                color=COLORS[t],
                marker=MARKERS[t],
                label=LABELS[t],
                linewidth=1.2,
            )
            for t in TOOLS
        ],
        loc="upper center",
        bbox_to_anchor=(0.53, 0.954),
        ncol=3,
        fontsize=8,
    )
    lengths = summary["nominal_lengths_nt"]
    ordinary = [
        row
        for row in summary["by_nominal_length"]
        if row["scope"] == "tool_positive_outputs"
    ]
    for tool in TOOLS:
        groups = [row for row in ordinary if row["tool"] == tool]
        nonempty = [row for row in groups if row["n"]]
        axes[0, 0].errorbar(
            [row["nominal_length_nt"] for row in nonempty],
            [row["median_nt"] for row in nonempty],
            yerr=[
                [row["median_nt"] - row["q25_nt"] for row in nonempty],
                [row["q75_nt"] - row["median_nt"] for row in nonempty],
            ],
            color=COLORS[tool],
            marker=MARKERS[tool],
            markersize=3,
            linewidth=1.1,
            capsize=2,
        )
        axes[0, 1].plot(
            [row["nominal_length_nt"] for row in groups],
            [100 * row["fraction_of_group"] for row in groups],
            color=COLORS[tool],
            marker=MARKERS[tool],
            markersize=3,
            linewidth=1.1,
        )
    axes[0, 0].plot([0, max(lengths)], [0, max(lengths)], "--", color=".5", lw=0.8)
    axes[0, 0].set(
        title="Positive outputs: median and IQR",
        ylabel="Reported length / native proxy (nt)",
    )
    axes[0, 1].set(
        title="Positive-length output coverage", ylabel="Group reads (%)", ylim=(0, 105)
    )
    for axis in axes[0]:
        axis.set(xlabel="Nominal construct length (nt)", xticks=lengths)
        axis.set_xticklabels(["0*" if n == 0 else str(n) for n in lengths])
        axis.grid(axis="y", alpha=0.16, linewidth=0.5)
    same = shared_ids(rows)
    max_error = 10.0
    for tool in TOOLS:
        errors = sorted(
            abs(float(row["positive_length_nt"]) - int(row["nominal_length_nt"]))
            for row in positive_rows(rows, tool)
            if row["read_id"] in same
        )
        if errors:
            max_error = max(max_error, max(errors))
            axes[1, 0].step(
                [0, *errors],
                [0, *[(i + 1) / len(errors) for i in range(len(errors))]],
                where="post",
                color=COLORS[tool],
                lw=1.2,
            )
    axes[1, 0].set_xscale("symlog", linthresh=10)
    ticks = [
        n
        for n in (0, 5, 10, 20, 50, 100, 200, 500, 1000, 5000)
        if n <= max_error * 1.05
    ]
    axes[1, 0].set(
        title=f"Same positive-output reads (n={len(same):,})",
        xlabel="Absolute nominal deviation (nt; symlog)",
        ylabel="Cumulative fraction",
        xlim=(0, max_error * 1.05),
        ylim=(0, 1.03),
        xticks=ticks,
    )
    axes[1, 0].set_xticklabels([str(n) for n in ticks])
    axes[1, 0].grid(alpha=0.16, linewidth=0.5)
    if not same:
        axes[1, 0].text(
            0.5,
            0.5,
            "No common positive estimates",
            ha="center",
            transform=axes[1, 0].transAxes,
        )
    categories = [
        ("positive_length", "Positive length", None),
        ("zero_output", "Native zero output", "#E4CD9C"),
        ("no_candidate", "No candidate", "#C7CCD1"),
        ("other_or_ambiguous", "Other / ambiguous", "#9C91AD"),
        ("error_or_missing", "Error / missing", "#565B61"),
    ]
    for index, tool in enumerate(TOOLS):
        left = 0.0
        for key, _, color in categories:
            n = summary["status_category_counts"][tool].get(key, 0)
            width = n / summary["total_reads"] * 100
            axes[1, 1].barh(
                index, width, left=left, color=color or COLORS[tool], height=0.60
            )
            if width >= 12:
                axes[1, 1].text(
                    left + width / 2,
                    index,
                    str(n),
                    ha="center",
                    va="center",
                    fontsize=6.7,
                    color="white" if key == "positive_length" else "#202020",
                )
            left += width
    axes[1, 1].set(
        title="All reads: native output states",
        yticks=range(3),
        yticklabels=[LABELS[t] for t in TOOLS],
        xlim=(0, 100),
        xlabel="All cohort reads (%)",
    )
    axes[1, 1].invert_yaxis()
    axes[1, 1].legend(
        handles=[
            Patch(facecolor=color or "#707070", label=label)
            for _, label, color in categories
        ],
        loc="upper center",
        bbox_to_anchor=(0.44, -0.23),
        ncol=2,
        fontsize=6,
        columnspacing=0.8,
        handlelength=1.1,
    )
    for letter, axis in zip("abcd", axes.flat, strict=True):
        axis.text(
            -0.19,
            1.13,
            letter,
            fontweight="bold",
            fontsize=10,
            transform=axis.transAxes,
        )
        axis.tick_params(labelsize=6.5)
    figure.text(
        0.10,
        0.080,
        "0* = nominal-zero negative group. Dots: median; whiskers: "
        "interquartile range of positive outputs.",
        fontsize=6.5,
    )
    figure.text(
        0.10,
        0.055,
        "*PolyAnalysis: native interval / 30 proxy, uncalibrated. "
        "Dorado pt=0 is retained as a separate output state.",
        fontsize=6.5,
    )
    figure.text(
        0.10,
        0.030,
        "Construct weak labels; no per-read truth. Paired subset: nominal >0, "
        "all tools positive; length mix is unbalanced.",
        fontsize=6.5,
    )
    save_figure(figure, output, "fixed_len_tool_comparison")


def distributions(summary: dict, rows: list[dict[str, str]], output: Path) -> None:
    """Plot every group using a full-group denominator and explicit absent mass."""
    lengths = summary["nominal_lengths_nt"]
    cols = 4
    nrows = math.ceil((len(lengths) + 1) / cols)
    figure, axes = plt.subplots(nrows, cols, figsize=(10.8, 3.1 * nrows), squeeze=False)
    figure.subplots_adjust(
        left=0.065, right=0.98, top=0.86, bottom=0.16, hspace=0.58, wspace=0.32
    )
    figure.suptitle(
        "Fixed-length poly(A): cumulative native output distributions",
        fontsize=12,
        y=0.98,
    )
    figure.legend(
        handles=[
            Line2D([], [], color=COLORS[t], label=LABELS[t], lw=1.5) for t in TOOLS
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.94),
        ncol=3,
    )
    max_length = max(
        (float(r["positive_length_nt"]) for r in rows if r["positive_length_nt"]),
        default=20,
    )
    for index, length in enumerate(lengths):
        axis = axes.flat[index]
        total = summary["counts_by_nominal_length"][str(length)]
        for tool in TOOLS:
            values = sorted(
                float(r["positive_length_nt"])
                for r in positive_rows(rows, tool)
                if int(r["nominal_length_nt"]) == length
            )
            if values:
                axis.step(
                    [0, *values, max_length * 1.05],
                    [
                        0,
                        *[(i + 1) / total * 100 for i in range(len(values))],
                        len(values) / total * 100,
                    ],
                    where="post",
                    lw=1.25,
                    color=COLORS[tool],
                )
        axis.axvline(length, ls="--", color=".45", lw=0.8)
        axis.set_xscale("symlog", linthresh=20)
        ticks = [
            n for n in (0, 20, 50, 100, 200, 500, 1000, 5000) if n <= max_length * 1.05
        ]
        axis.set(
            title=f"{length} nt"
            + (" (negative)" if length == 0 else "")
            + f"; n={total}",
            ylim=(0, 103),
            xlim=(0, max_length * 1.05),
            xticks=ticks,
            xlabel="Native length / proxy (nt; symlog)",
        )
        axis.set_xticklabels([str(n) for n in ticks], fontsize=7)
        axis.set_yticks([0, 50, 100])
        if index % cols == 0:
            axis.set_ylabel("Cumulative group reads (%)")
        axis.grid(alpha=0.15, linewidth=0.5)
    for axis in list(axes.flat)[len(lengths) :]:
        axis.set_axis_off()
    note_axis = axes.flat[len(lengths)]
    note_axis.text(
        0,
        0.88,
        "Reading the curves",
        fontsize=10,
        fontweight="bold",
        transform=note_axis.transAxes,
    )
    note_axis.text(
        0,
        0.68,
        "Denominator: all group reads.\n"
        "Curves use positive lengths only.\n"
        "Plateau = output coverage.\n"
        "Dashed line = nominal length.\n"
        "Missing / zero tags stay separate.\n\n"
        "*PolyAnalysis: uncalibrated\n"
        "interval / 30 proxy.",
        fontsize=7.5,
        linespacing=1.7,
        va="top",
        transform=note_axis.transAxes,
    )
    figure.text(
        0.065,
        0.065,
        "All values and outliers are shown. Symlog axes are linear below 20 nt "
        "and logarithmic above 20 nt.",
        fontsize=8,
    )
    figure.text(
        0.065,
        0.035,
        "Construct nominal weak labels; descriptive tool outputs, "
        "not independently verified biological tail lengths.",
        fontsize=8,
    )
    save_figure(figure, output, "fixed_len_group_distributions")


def box_scatter_comparison(
    summary: dict, rows: list[dict[str, str]], output: Path
) -> None:
    """Combine all tools in a/b/c panels with actual nominal nt spacing.

    Each tool retains its own positive-output cohort and all outliers. Display
    offsets separate the boxes without changing their nominal group or metrics.
    """
    figure, axes = plt.subplots(1, 3, figsize=(14, 7.2))
    figure.subplots_adjust(left=0.065, right=0.985, top=0.82, bottom=0.39, wspace=0.28)
    figure.suptitle(
        "Fixed-length poly(A): comparison of three tools", fontsize=14, y=0.975
    )
    lengths = summary["nominal_lengths_nt"]
    rng = random.Random(20261103)
    offsets = dict(zip(TOOLS, (-4, 0, 4), strict=True))
    counts = {}
    all_values, all_deviations = [], []
    for tool in TOOLS:
        selected = sorted(positive_rows(rows, tool), key=lambda row: row["read_id"])
        counts[tool] = {}
        color = COLORS[tool]
        for length in lengths:
            values = [
                float(r["positive_length_nt"])
                for r in selected
                if int(r["nominal_length_nt"]) == length
            ]
            counts[tool][length] = len(values)
            deviations = [value - length for value in values]
            position = length + offsets[tool]
            jittered = [position + rng.uniform(-1.1, 1.1) for _ in values]
            for axis, points in ((axes[0], values), (axes[2], deviations)):
                if points:
                    axis.scatter(
                        jittered,
                        points,
                        s=6,
                        color=color,
                        alpha=0.24,
                        linewidths=0,
                        zorder=1,
                    )
                    axis.boxplot(
                        [points],
                        positions=[position],
                        widths=3.2,
                        patch_artist=True,
                        showfliers=False,
                        manage_ticks=False,
                        boxprops={
                            "facecolor": color,
                            "alpha": 0.6,
                            "edgecolor": color,
                            "linewidth": 1,
                        },
                        whiskerprops={"color": color, "linewidth": 1},
                        capprops={"color": color, "linewidth": 1},
                        medianprops={"color": "#202020", "linewidth": 1},
                        zorder=3,
                    )
            axes[1].scatter(
                [length] * len(values),
                values,
                s=9,
                alpha=0.28,
                color=color,
                marker=MARKERS[tool],
                linewidths=0,
                zorder=2,
            )
            all_values.extend(values)
            all_deviations.extend(deviations)
    figure.legend(
        handles=[
            Line2D(
                [],
                [],
                marker=MARKERS[t],
                color=COLORS[t],
                linestyle="none",
                label=f"{LABELS[t]} "
                f"({sum(counts[t].values()):,}/{summary['total_reads']:,})",
            )
            for t in TOOLS
        ],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.935),
        ncol=3,
        fontsize=10,
    )
    titles = (
        "a  Length distributions",
        "b  Agreement with nominal length",
        "c  Signed length deviations",
    )
    for index, axis in enumerate(axes):
        axis.set(
            xticks=lengths,
            xlim=(-10, max(lengths) + 12),
            xlabel="Nominal poly(A) length (nt)",
            ylabel="Output - nominal (nt)"
            if index == 2
            else "Reported length / proxy (nt)",
        )
        axis.set_xticklabels([str(n) for n in lengths])
        axis.set_title(
            titles[index], fontsize=11, fontweight="bold", loc="left", pad=12
        )
        axis.tick_params(labelsize=9)
        axis.grid(axis="y", color="#e6e6e6", linewidth=0.5, zorder=0)
    max_value = max([float(max(lengths)), *all_values, 1])
    for axis in axes[:2]:
        axis.set_ylim(-max_value * 0.025, max_value * 1.06)
    if all_deviations:
        lower, upper = min(0.0, min(all_deviations)), max(0.0, max(all_deviations))
        padding = max(5.0, (upper - lower) * 0.06)
        axes[2].set_ylim(lower - padding, upper + padding)
    axes[0].plot(
        lengths,
        lengths,
        "--D",
        color="#333333",
        markersize=3,
        linewidth=1,
        label="Nominal length",
        zorder=4,
    )
    axes[1].plot(
        [0, max(lengths)],
        [0, max(lengths)],
        "--",
        color="#333333",
        linewidth=1,
        label="Output = nominal",
        zorder=4,
    )
    axes[2].axhline(0, color="#333333", ls="--", lw=1, zorder=4)
    for axis in axes[:2]:
        axis.legend(loc="upper left", fontsize=8)
    table_axis = figure.add_axes((0.065, 0.18, 0.92, 0.115))
    table_axis.axis("off")
    table_axis.set_title(
        "Positive outputs included in each group (n)", fontsize=9, loc="left", pad=5
    )
    table = table_axis.table(
        cellText=[[LABELS[t], *[str(counts[t][n]) for n in lengths]] for t in TOOLS],
        colLabels=[
            "Tool / nominal nt",
            *[f"{n}" + (" (negative)" if n == 0 else "") for n in lengths],
        ],
        colWidths=[0.19] + [0.81 / len(lengths)] * len(lengths),
        cellLoc="center",
        bbox=(0, 0, 1, 1),
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    for (row_index, column_index), cell in table.get_celld().items():
        cell.set_edgecolor("#dddddd")
        cell.set_linewidth(0.4)
        if row_index == 0:
            cell.set_facecolor("#f3f3f3")
        elif column_index == 0:
            cell.get_text().set_color(COLORS[TOOLS[row_index - 1]])
    notes = (
        "Boxes: median / IQR; whiskers: 1.5 IQR. Dots include all positive outputs "
        "and outliers; linear axes show the full range.",
        "a/c: tool offsets (-4, 0, +4 nt) and seeded jitter separate boxes; "
        "b: exact nominal x. Actual nt spacing is preserved.",
        "Each tool uses its own positive-output cohort. Missing estimates and "
        "native zero tags are not imputed as biological zero tails.",
        "*PolyAnalysis: uncalibrated interval / 30 proxy. Nominal construct "
        "lengths are weak labels, not verified per-read truth.",
    )
    for index, note in enumerate(notes):
        figure.text(0.065, 0.135 - index * 0.027, note, fontsize=8)
    save_figure(figure, output, "fixed_len_box_scatter_combined")
    (output / "box_scatter_combined_summary.json").write_text(
        json.dumps(
            {
                "backend": "python/matplotlib",
                "layout": "a/b/c panels; three tools together in every panel",
                "panel_titles": list(titles),
                "positive_outputs_by_nominal_length": counts,
                "nominal_x_values_nt": lengths,
                "tool_offsets_nt": offsets,
                "agreement_x_offsets_nt": 0,
                "jitter_seed": 20261103,
                "jitter_half_width_nt": 1.1,
                "box_width_nt": 3.2,
                "whiskers": "1.5 IQR",
                "all_positive_output_points_shown": True,
                "y_ranges": "full linear ranges; length panels share limits",
                "zero_or_missing_imputation": False,
                "source": "per_read_comparison.tsv",
                "source_summary": "summary.json",
            },
            indent=2,
        )
        + "\n"
    )


def stacked_output_coverage(
    summary: dict, rows: list[dict[str, str]], output: Path
) -> None:
    """Plot native output fractions on full denominators, separately from accuracy.

    Overall coverage excludes nominal-zero constructs. Per-group stacks retain
    zero, missing and ambiguous outputs rather than imputing negative calls.
    """
    categories = (
        ("positive_length", "Positive length", None),
        ("zero_output", "Native zero output", "#E4CD9C"),
        ("no_candidate", "No candidate", "#C7CCD1"),
        ("other_or_ambiguous", "Other / ambiguous", "#9C91AD"),
        ("error_or_missing", "Error / missing", "#565B61"),
    )
    allowed = {key for key, _, _ in categories}
    lengths = summary["nominal_lengths_nt"]
    denominators = {int(k): v for k, v in summary["counts_by_nominal_length"].items()}
    records = []
    for tool in TOOLS:
        for length in lengths:
            selected = [
                r
                for r in rows
                if r["tool"] == tool and int(r["nominal_length_nt"]) == length
            ]
            if len(selected) != denominators[length] or len(
                {r["read_id"] for r in selected}
            ) != len(selected):
                raise ValueError("coverage rows must match the full group denominator")
            counts = Counter(r["status_category"] for r in selected)
            if set(counts) - allowed:
                raise ValueError("unknown output status category")
            records.append(
                {
                    "scope": "by_nominal_length",
                    "tool": tool,
                    "nominal_length_nt": length,
                    "denominator": len(selected),
                    **{key: counts[key] for key, _, _ in categories},
                }
            )
        positive_groups = [
            r
            for r in records
            if r["tool"] == tool
            and r["scope"] == "by_nominal_length"
            and r["nominal_length_nt"] > 0
        ]
        records.append(
            {
                "scope": "nominal_gt0_overall",
                "tool": tool,
                "nominal_length_nt": "",
                "denominator": sum(r["denominator"] for r in positive_groups),
                **{
                    key: sum(r[key] for r in positive_groups)
                    for key, _, _ in categories
                },
            }
        )
    for record in records:
        n = record["denominator"]
        record["positive_output_percent"] = (
            100 * record["positive_length"] / n if n else None
        )
    figure, axes = plt.subplots(
        1, 2, figsize=(14, 6.8), gridspec_kw={"width_ratios": [1, 3.8]}
    )
    figure.subplots_adjust(left=0.065, right=0.985, bottom=0.29, top=0.76, wspace=0.28)
    figure.suptitle(
        "Fixed-length poly(A): positive-output coverage and native states",
        fontsize=14,
        y=0.975,
    )
    offsets = dict(zip(TOOLS, (-4, 0, 4), strict=True))
    for record in records:
        tool, n = record["tool"], record["denominator"]
        if not n:
            continue
        overall = record["scope"] == "nominal_gt0_overall"
        axis = axes[0] if overall else axes[1]
        position = (
            TOOLS.index(tool)
            if overall
            else (record["nominal_length_nt"] + offsets[tool])
        )
        bottom = 0.0
        for key, _, color in categories:
            height = 100 * record[key] / n
            axis.bar(
                position,
                height,
                bottom=bottom,
                width=0.65 if overall else 3.3,
                color=color or COLORS[tool],
                edgecolor="white",
                linewidth=0.35,
                zorder=2,
            )
            bottom += height
        fraction = record["positive_output_percent"]
        axis.text(
            position,
            fraction / 2 if fraction >= 12 else fraction + 3,
            f"{fraction:.1f}%",
            ha="center",
            va="center",
            rotation=90,
            fontsize=8 if overall else 7,
            color="white" if fraction >= 12 else COLORS[tool],
            zorder=3,
        )
        if not overall:
            axis.text(
                position,
                -0.045,
                {"Holypoly": "H", "Dorado": "D", "PolyAnalysis": "P"}[tool],
                ha="center",
                va="top",
                fontsize=8,
                color=COLORS[tool],
                transform=axis.get_xaxis_transform(),
            )
    total = sum(denominators[n] for n in lengths if n > 0)
    axes[0].set(
        xticks=range(3),
        xticklabels=["H", "D", "P"],
        title=f"a  Positive constructs (n={total:,}/tool)",
        ylabel="Fraction of all group reads (%)",
        xlabel="Tool",
    )
    axes[1].set(
        xticks=lengths,
        xlim=(-10, max(lengths) + 10),
        xticklabels=["0*" if n == 0 else str(n) for n in lengths],
        title="b  By nominal construct length",
        xlabel="Nominal poly(A) length (nt)",
    )
    axes[1].tick_params(axis="x", pad=25)
    if 0 in lengths:
        axes[1].axvspan(-8, 8, color="#eeeeee", zorder=0)
    for axis in axes:
        axis.set_ylim(0, 105)
        axis.set_yticks(range(0, 101, 20))
        axis.grid(axis="y", color="#dddddd", linewidth=0.5, zorder=0)
        axis.title.set_fontsize(10)
        axis.title.set_fontweight("bold")
    figure.legend(
        handles=[
            Patch(facecolor=COLORS[t], label=f"{t}: positive length") for t in TOOLS
        ]
        + [Patch(facecolor=color, label=label) for key, label, color in categories[1:]],
        loc="upper center",
        bbox_to_anchor=(0.52, 0.925),
        ncol=4,
        fontsize=9,
    )
    group_counts = "; ".join(f"{n} nt: {denominators[n]}" for n in lengths)
    notes = (
        "H = Holypoly; D = Dorado; P = PolyAnalysis. Labels on colored segments: "
        "positive-length output percentage.",
        f"Full denominators per tool: {group_counts}. Each stack sums to 100%.",
        "0* = negative construct group: positive outputs here are NOT recall. "
        "Panel a excludes nominal 0 nt reads.",
        "Positive-output coverage is a weak-label detection proxy, "
        "not verified biological recall or length accuracy.",
        "Native zero / no candidate / error states remain separate; "
        "none is assumed to confirm absence of a biological tail.",
    )
    for index, note in enumerate(notes):
        figure.text(0.065, 0.18 - index * 0.03, note, fontsize=8)
    save_figure(figure, output, "fixed_len_output_coverage_stacked")
    with (output / "output_coverage_stacked.tsv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(records)
    (output / "output_coverage_stacked_summary.json").write_text(
        json.dumps(
            {
                "claim": "Tools differ in native positive-output coverage "
                "across nominal groups",
                "archetype": "quantitative grid",
                "backend": "python/matplotlib",
                "panels": {
                    "a": "All nominal-positive constructs",
                    "b": "Each nominal group",
                },
                "metric": "positive native outputs / all reads "
                "in the selected nominal group",
                "not_verified_biological_recall": True,
                "zero_imputation": False,
                "negative_construct_excluded_from_overall": True,
                "stack_categories": [key for key, _, _ in categories],
                "source": "per_read_comparison.tsv",
                "counts": records,
                "dimensions_inches": [14, 6.8],
                "exports": ["png 300 dpi", "pdf", "svg"],
                "statistics": "Descriptive fractions; "
                "no significance tests or biological replicates",
                "review_risks": [
                    "Nominal construct labels are weak labels",
                    "Native output states differ between tools",
                    "Positive outputs do not imply correct length",
                ],
            },
            indent=2,
        )
        + "\n"
    )


def report(summary: dict, output: Path) -> None:
    """Explain denominators, sentinel values and weak-label limits alongside plots."""
    lines = [
        "# Fixed-len：现有三种工具输出比较",
        "",
        f"同一队列 {summary['total_reads']:,} 条 read；"
        f"标称长度分组：{summary['counts_by_nominal_length']}。",
        "",
        "## 比较口径",
        "",
        "图中展示已有设置下的原生输出，未重新运行模型、调参、换算标定或重读 POD5。",
        "长度曲线仅纳入原生正长度输出；缺失、未检出及 Dorado pt=0 "
        "仍保留在全体状态统计和逐 read 表中，不能解释为确认无尾。",
        "Holypoly 的状态概率没有用于清零回归结果。"
        "PolyAnalysis 使用原生固定 30 samples/nt 换算，标记为未标定 proxy。",
        "Dorado 此次使用自定义 poly-A 配置并提供参考比对，工具运行条件不完全相同。",
        "",
        "## 正长度输出覆盖",
        "",
        "| 工具 | 正长度 read | 全体占比 |",
        "|---|---:|---:|",
    ]
    for tool in TOOLS:
        n = summary["status_category_counts"][tool].get("positive_length", 0)
        lines.append(f"| {tool} | {n} | {n / summary['total_reads']:.2%} |")
    lines += [
        "",
        "## 共同 read 子集的标称长度偏差",
        "",
        "三者都输出正长度且标称 >0："
        f"{summary['shared_positive_nominal_gt0_reads']} 条；"
        f"各组数量 {summary['shared_counts_by_nominal_length']}。",
        "该交集不再保持长度均衡，也不包含未检出 read；不可单独用于总体优劣排名。",
        "",
        "| 工具 | 平均绝对标称偏差 (nt) | 中位绝对标称偏差 (nt) | 平均标称 Bias (nt) |",
        "|---|---:|---:|---:|",
    ]
    for item in summary["overall"]:
        if item["scope"] == "shared_positive_nominal_gt0":
            values = [
                item[k]
                for k in (
                    "mean_absolute_nominal_deviation_nt",
                    "median_absolute_nominal_deviation_nt",
                    "mean_nominal_bias_nt",
                )
            ]
            text = " | ".join("NA" if v is None else f"{v:.2f}" for v in values)
            lines.append(f"| {item['tool']} | {text} |")
    lines += [
        "",
        "## 原生状态",
        "",
        "```json",
        json.dumps(summary["native_status_counts"], indent=2, ensure_ascii=False),
        "```",
        "",
        "## 解释限制",
        "",
    ]
    lines.extend("- " + text for text in summary["limitations"])
    lines += [
        "",
        "## 文件",
        "",
        "- fixed_len_tool_comparison.png/.pdf/.svg：主对比图。",
        "- fixed_len_group_distributions.png/.pdf/.svg："
        "7 组分布，分母始终为该组全部 read。",
        "- fixed_len_box_scatter_combined.png/.pdf/.svg："
        "a/b/c 三联图，各图同时展示三工具的长度分布、标称一致性和带符号偏差。",
        "- fixed_len_output_coverage_stacked.png/.pdf/.svg："
        "正构建总体及各标称长度组的全分母原生状态堆叠柱状图。",
        "- per_read_comparison.tsv：逐 read、逐工具的原始数值、正长度值、"
        "原生状态和 Dorado 区间有效性。",
        "- group_summary.tsv / overall_summary.tsv：分组与总体描述统计。",
        "- summary.json / figure_contract.json / plot_qa.json："
        "输入哈希、配置、统计口径和绘图核验。",
        "",
        "read 不是独立生物学重复；没有进行显著性检验。"
        "IQR 是 read 输出分布，不是均值置信区间。",
        "构建标称长度不等于逐 read 真值，因此这里不宣称正式准确率或跨批次泛化性能。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    """Extract in Bonito's environment and plot in the existing parent runtime."""
    args = build_parser().parse_args()
    inputs, output = resolve_paths(args)
    project = Path(__file__).resolve().parents[1]
    python = project / "src/bonito/.venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError("Bonito environment is required for BAM tag extraction")
    print(
        "Read existing tables/BAM only: "
        f"{sum(p.stat().st_size for p in inputs) / 1024**2:.1f} MiB; "
        "CPU only; no POD5/model/GPU.",
        flush=True,
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project / "src")
    subprocess.run(
        [str(python), "-m", "holypoly.evaluation.fixed_len", *sys.argv[1:]],
        cwd=project,
        env=environment,
        check=True,
    )
    summary = json.loads((output / "summary.json").read_text())
    with (output / "per_read_comparison.tsv").open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    (output / "figure_contract.json").write_text(
        json.dumps(
            {
                "claim": "Describe how native output lengths track nominal "
                "construct groups, with coverage and missing/zero-output "
                "states shown explicitly.",
                "archetype": "quantitative grid",
                "backend": "python/matplotlib",
                "evidence_chain": {
                    "a": "Median/IQR by actual nominal nt spacing",
                    "b": "Positive-output coverage on all reads",
                    "c": "Same-read deviations from weak labels",
                    "d": "Native states on the full cohort",
                    "supplement": "Group CDFs using full-group denominators",
                    "box_scatter": "Three tools together in a/b/c: "
                    "length boxes, nominal agreement, signed deviations",
                },
                "exports": ["png 300 dpi", "editable-text pdf", "editable-text svg"],
                "dimensions_inches": {
                    "overview": [7.2, 6.8],
                    "distributions": [10.8, 6.2],
                    "box_scatter": [14, 7.2],
                },
                "statistics": "Descriptive read-level medians/IQR; no "
                "biological replicates, confidence intervals or significance tests.",
                "review_risks": summary["limitations"],
            },
            indent=2,
        )
        + "\n"
    )
    overview(summary, rows, output)
    distributions(summary, rows, output)
    box_scatter_comparison(summary, rows, output)
    stacked_output_coverage(summary, rows, output)
    report(summary, output)
    qa = {
        "backend": "python",
        "source_rows": len(rows),
        "tool_rows": {t: sum(r["tool"] == t for r in rows) for t in TOOLS},
        "shared_subset_matches_summary": len(shared_ids(rows))
        == summary["shared_positive_nominal_gt0_reads"],
        "actual_nominal_x_values": summary["nominal_lengths_nt"],
        "zero_and_missing_never_imputed": True,
        "exports": {
            p.name: p.stat().st_size
            for p in output.iterdir()
            if p.suffix in {".png", ".svg", ".pdf"}
        },
    }
    if not qa["shared_subset_matches_summary"]:
        raise ValueError("plot/source paired subsets disagree")
    (output / "plot_qa.json").write_text(json.dumps(qa, indent=2) + "\n")
    print(f"Finished: {output}", flush=True)


if __name__ == "__main__":
    main()
