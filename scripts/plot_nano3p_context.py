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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from holypoly.core.config import load_config  # noqa: E402
from holypoly.evaluation.fixed_len import TOOLS, fingerprint  # noqa: E402

COLORS = {"Holypoly": "#4477AA", "Dorado": "#CC9933", "PolyAnalysis": "#AA6677"}
PUBLISHED_COLOR = "#228877"


def plot_context(
    local_csv: Path,
    published_csv: Path,
    output: Path,
    *,
    combined: bool = False,
    focus_max_nt: float | None = None,
    tailfindr_groups: dict[str, dict] | None = None,
    local_tools: tuple[str, ...] = TOOLS,
) -> list[dict]:
    """Draw conditional positive lengths without pooling or mapping source labels.

    Published zeros remain in the CSV and count table. A numeric cDNA group
    suffix determines display spacing only, not an asserted per-read true length.
    Combined layout uses one axis with numeric cDNA spacing and labeled RNA
    categories; an optional additional focus view retains full-subset statistics.
    """
    if not local_tools or len(set(local_tools)) != len(local_tools):
        raise ValueError("select distinct local tools")
    if set(local_tools) - set(TOOLS):
        raise ValueError("unsupported local tool")
    local = defaultdict(list)
    lengths = set()
    with local_csv.open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["tool"] not in local_tools:
                continue
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
    has_dna = any(
        group.get("molecule") == "DNA" for group in (tailfindr_groups or {}).values()
    )
    nominal_tail = {
        name: group
        for name, group in (tailfindr_groups or {}).items()
        if group.get("nominal_length_nt") is not None
    }
    categorical_tail = {
        name: group
        for name, group in (tailfindr_groups or {}).items()
        if group.get("nominal_length_nt") is None
    }
    if combined:
        figure, axis = plt.subplots(figsize=(20 if has_dna else 16, 8))
        axes = [axis]
    else:
        figure, axes = plt.subplots(
            1, 3, figsize=(15, 7), gridspec_kw={"width_ratios": [1.8, 1.25, 0.6]}
        )
    figure.subplots_adjust(left=0.055, right=0.985, top=0.79, bottom=0.34, wspace=0.32)
    if tailfindr_groups and not combined:
        raise ValueError("tailfindr context requires the combined layout")
    figure.suptitle(
        "Poly(A/T) lengths: local cDNA, Nano3P-seq and tailfindr RNA/DNA"
        if has_dna
        else "Poly(A) lengths: fixed-len, Nano3P-seq and tailfindr source data"
        if tailfindr_groups
        else "Poly(A) lengths: fixed-len tool comparison and Nano3P-seq source data",
        fontsize=14,
        y=0.975,
    )
    figure.legend(
        handles=[
            Patch(facecolor=COLORS[t], label=t + ("*" if t == "PolyAnalysis" else ""))
            for t in local_tools
        ]
        + [Patch(facecolor=PUBLISHED_COLOR, label="Nano3P-seq (different dataset)")]
        + (
            [Patch(facecolor="#6655AA", label="tailfindr (different RNA dataset)")]
            if tailfindr_groups
            else []
        )
        + (
            [
                Patch(
                    facecolor="#882255",
                    label="tailfindr DNA / poly(A) (different dataset)",
                )
            ]
            if has_dna
            else []
        )
        + (
            [
                Patch(
                    facecolor="#CC88AA",
                    label="tailfindr DNA / poly(T) (different dataset)",
                )
            ]
            if nominal_tail
            else []
        )
        + (
            [
                Line2D(
                    [],
                    [],
                    color="#202020",
                    marker="o",
                    markerfacecolor="white",
                    linestyle="none",
                    markersize=4,
                    label="Mean",
                ),
                Line2D([], [], color="#202020", linewidth=1.1, label="Median"),
            ]
            if combined
            else []
        ),
        loc="upper center",
        bbox_to_anchor=(0.52, 0.93),
        ncol=4 if tailfindr_groups else (6 if combined else 4),
        fontsize=9 if combined else 10,
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
            showmeans=combined,
            meanprops={
                "marker": "o",
                "markersize": 3,
                "markerfacecolor": "white",
                "markeredgecolor": "#202020",
                "markeredgewidth": 0.7,
            },
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

    def record(
        cohort,
        tool,
        group,
        values,
        total=None,
        zero=None,
        nominal=None,
        position=None,
        label_source=None,
    ):
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
                "mean_nt": float(np.mean(values)) if values else None,
                "q25_nt": q[0],
                "median_nt": q[1],
                "q75_nt": q[2],
                "nominal_length_nt": nominal,
                "plot_x_nt": position,
                "label_source": label_source,
            }
        )

    ordered_lengths = sorted(lengths)
    span = (1.2 if combined else 2) * (len(local_tools) - 1)
    offsets = np.linspace(-span, span, len(local_tools))
    if nominal_tail:
        offsets = np.linspace(-2, -1, len(local_tools))
    for tool, offset in zip(local_tools, offsets, strict=True):
        for length in ordered_lengths:
            values = local[tool, length]
            box(
                axes[0],
                values,
                length + offset,
                COLORS[tool],
                0.8 if nominal_tail else (1.7 if combined else 3.2),
            )
            record(
                "local_fixed_len",
                tool,
                str(length),
                values,
                nominal=length,
                position=length + offset,
                label_source="local nominal construct",
            )
    axes[0].plot(ordered_lengths, ordered_lengths, "--", color="#555555", lw=1)
    axes[0].set(
        xticks=ordered_lengths,
        xlim=(-10, max(lengths) + 10),
        xlabel="Nominal construct length (nt)",
        title=f"a  Fixed-len: {len(local_tools)} tools, same holdout",
    )
    cdna = sorted(
        (g for kind, g in published if kind == "cDNA"),
        key=lambda g: int(g.split("_")[-1]),
    )
    rna = sorted(g for kind, g in published if kind == "RNA")
    numeric_groups = sorted(
        lengths
        | {int(g.split("_")[-1]) for g in cdna}
        | {int(g["nominal_length_nt"]) for g in nominal_tail.values()}
    )
    rna_positions = [max(numeric_groups) + 25 + 20 * index for index in range(len(rna))]
    published_axes = (axes[0], axes[0]) if combined else (axes[1], axes[2])
    for axis, kind, groups in (
        (published_axes[0], "cDNA", cdna),
        (published_axes[1], "RNA", rna),
    ):
        positions = (
            [int(g.split("_")[-1]) for g in groups]
            if kind == "cDNA"
            else (rna_positions if combined else list(range(len(groups))))
        )
        if combined and kind == "cDNA" and not nominal_tail:
            positions = [
                position + 4.8 if position in lengths else position
                for position in positions
            ]
        for group, position in zip(groups, positions, strict=True):
            reported = published[kind, group]
            values = [v for v in reported if v > 0]
            width = (
                0.8
                if nominal_tail and kind == "cDNA"
                else (1.7 if combined else (5 if kind == "cDNA" else 0.5))
            )
            box(axis, values, position, PUBLISHED_COLOR, width)
            record(
                "published_Fig3c",
                "Nano3P-seq",
                group,
                values,
                len(reported),
                sum(v == 0 for v in reported),
                nominal=int(group.split("_")[-1]) if kind == "cDNA" else None,
                position=position,
                label_source="published cDNA group suffix" if kind == "cDNA" else None,
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
        if not combined:
            axis.set(
                xticks=positions,
                xticklabels=[g.removeprefix("cDNA_") for g in groups],
                xlabel="Source group suffix (cDNA_<value>)"
                if kind == "cDNA"
                else "Source RNA group",
            )
            if kind == "cDNA":
                axis.set_xlim(min(positions) - 10, max(positions) + 10)
    if combined:
        tail_positions = [
            max([*numeric_groups, *rna_positions]) + 23 + 22 * index
            for index in range(len(categorical_tail))
        ]
        positions_by_name = dict(zip(categorical_tail, tail_positions, strict=True))
        positions_by_name.update(
            {
                name: source["nominal_length_nt"]
                + (1 if source["poly_type"] == "polyA" else 2)
                for name, source in nominal_tail.items()
            }
        )
        for group, source in (tailfindr_groups or {}).items():
            position = positions_by_name[group]
            values = source["values"]
            if any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError("tailfindr plot values must be finite and positive")
            color = "#882255" if source.get("molecule") == "DNA" else "#6655AA"
            if group in nominal_tail and source["poly_type"] == "polyT":
                color = "#CC88AA"
            box(axes[0], values, position, color, 0.8 if group in nominal_tail else 2.8)
            counts = source["status_counts"]
            record(
                source.get("cohort", "tailfindr_RNA001"),
                "tailfindr",
                group,
                values,
                sum(counts.values()),
                counts.get("zero_value", 0),
                nominal=source.get("nominal_length_nt"),
                position=position,
                label_source=source.get("label_source"),
            )
            if group not in nominal_tail:
                axes[0].text(
                    position,
                    0.98,
                    f"n+={len(values):,}",
                    rotation=90,
                    va="top",
                    ha="center",
                    fontsize=8,
                    transform=axes[0].get_xaxis_transform(),
                )
        axes[0].set(
            xticks=[*numeric_groups, *rna_positions, *tail_positions],
            xticklabels=[
                *(str(value) for value in numeric_groups),
                *(f"{group}\nRNA" for group in rna),
                *(
                    source.get("display_label", f"tailfindr\n{group}")
                    for group, source in categorical_tail.items()
                ),
            ],
            xlim=(
                min(numeric_groups) - 8,
                max([*numeric_groups, *rna_positions, *tail_positions]) + 13,
            ),
            xlabel="Construct nominal length / published cDNA group suffix (nt); RNA "
            "mixtures at right"
            if nominal_tail
            else "Construct nominal length / published cDNA group suffix (nt); "
            "RNA/DNA categories at right"
            if has_dna
            else "Construct nominal length / published cDNA group suffix (nt); "
            "RNA categories at right",
            title="Shared axis; independent RNA/DNA datasets"
            if has_dna
            else "Shared axis; independent published datasets",
        )
        if rna:
            axes[0].axvline(max(numeric_groups) + 12, color="#aaaaaa", lw=0.8, ls=":")
            axes[0].text(
                np.mean([*rna_positions, *tail_positions]),
                1.025,
                "RNA mixtures (read-to-construct mapping unavailable)"
                if nominal_tail
                else "RNA/DNA groups (no assigned length)"
                if has_dna
                else "RNA groups (no assigned length)",
                ha="center",
                fontsize=9,
                transform=axes[0].get_xaxis_transform(),
            )
    else:
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
        (
            "Separate datasets and protocols; this is not a matched four-tool "
            "benchmark or an accuracy/recall ranking.",
            "Positive outputs only. Boxes: median/IQR; circles: mean; "
            "whiskers: 1.5 IQR; dots: all outliers. One shared linear y axis.",
            "Numeric x spacing is retained. Boxes are offset within groups; published "
            "cDNA suffixes are not remapped to local construct labels.",
            "R1/R2 are RNA categories, not numerical lengths. n+ annotates published "
            "positive rows; zeros and group sizes remain in source tables.",
            "*PolyAnalysis is an uncalibrated interval/30 proxy. Local lengths are "
            "nominal weak labels; no per-read ground truth is asserted.",
        )
        if combined
        else (
            "Separate datasets and protocols; distributions are descriptive context, "
            "not a matched four-tool benchmark or an accuracy/recall ranking.",
            "All panels use positive numeric outputs only. Boxes: median/IQR; "
            "whiskers: 1.5 IQR; dots: all outliers. Y ranges differ across panels.",
            "a: actual nominal nt spacing, tool offsets -4/0/+4 nt. "
            "b: numeric spacing from source group names, "
            "with no mapping to local labels.",
            "Nano3P-seq zeros are preserved in CSV and group counts; "
            "n+ counts positive rows within the published table, "
            "not all attempted reads.",
            "R1/R2 have no nominal length assigned here. "
            "*PolyAnalysis uses an uncalibrated interval/30 proxy; "
            "local labels are nominal weak labels.",
        )
    )
    if tailfindr_groups:
        notes = (
            "Different datasets: local cDNA holdout; published Nano3P-seq Fig3c; "
            "tailfindr paper RNA001 standard-basecalling results, two replicates.",
            "Descriptive distributions only; no matched five-tool benchmark or "
            "accuracy ranking. tailfindr replicate categories have no assigned nt.",
            *notes[1:3],
            "RNA groups at right are categories. n+ counts positive source entries; "
            "native zeros, missing and invalid values are audited separately.",
            notes[-1],
        )
    if has_dna:
        notes = (
            "Different datasets: local cDNA holdout; Nano3P-seq Fig3c; "
            "tailfindr paper RNA001 RNA and LSK108/109 DNA, standard basecalling.",
            "tailfindr DNA: native valid positive lengths; poly(A)/poly(T) and "
            "replicates shown separately. Categories have no assigned nominal nt.",
            *notes[2:],
        )
    if "PolyAnalysis" not in local_tools:
        notes = tuple(note for note in notes if "PolyAnalysis" not in note) + (
            "Local construct lengths are nominal weak labels, not per-read truth. "
            "Different datasets provide descriptive context, not accuracy ranking.",
        )
    if has_dna:
        notes = tuple(
            note.replace("RNA groups at right", "RNA/DNA groups at right")
            for note in notes
        )
    if nominal_tail:
        notes = (
            "Independent datasets: local cDNA holdout; Nano3P-seq Fig3c; "
            "tailfindr paper LSK108/109 DNA and RNA001 RNA.",
            "DNA labels: flipflop-decoded barcodes joined to "
            "standard-basecalling lengths by replicate + read ID; threshold "
            "passed, no tie, agreeing A/T direction.",
            "DNA: six nominal constructs (10, 30, 40, 60, 100, 150 nt); "
            "replicates pooled within length and direction. Barcode labels are "
            "not per-read true lengths.",
            "tailfindr RNA: same six expected lengths (user supplied); no "
            "read-to-construct table available. Right-hand boxes are replicate "
            "mixtures, not six assigned groups.",
            "Actual nt spacing retained; source/type offsets -2/-1/0/+1/+2 nt. "
            "Positive lengths only; median/IQR, 1.5 IQR whiskers, all outliers; "
            "circles: mean.",
            "Group counts and excluded statuses are in source tables. "
            "Different datasets give descriptive context; no accuracy ranking.",
        )
    for index, note in enumerate(notes):
        figure.text(0.055, 0.235 - 0.032 * index, note, fontsize=8)
    stem = (
        "fixed_len_with_nano3p_tailfindr_combined"
        if tailfindr_groups
        else "fixed_len_with_nano3p_combined"
        if combined
        else "fixed_len_with_nano3p_context"
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output / f"{stem}.{suffix}",
            dpi=300,
            facecolor="white",
        )
    if combined and focus_max_nt is not None:
        axes[0].set_ylim(0, focus_max_nt)
        above = sum(
            value > focus_max_nt for values in local.values() for value in values
        )
        above += sum(
            value > focus_max_nt for values in published.values() for value in values
        )
        above += sum(
            value > focus_max_nt
            for group in (tailfindr_groups or {}).values()
            for value in group["values"]
        )
        figure.text(
            0.055,
            0.055,
            f"Focus view: 0-{focus_max_nt:g} nt; {above:,} positive values above this "
            "range. All values are retained in the full-range figure "
            "and source tables.",
            fontsize=8,
        )
        for suffix in ("png", "pdf", "svg"):
            figure.savefig(
                output / f"{stem}_focus_0_{focus_max_nt:g}nt.{suffix}",
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
    """Render descriptive plots in a new safe external work directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    parser.add_argument("--local-comparison", type=Path, required=True)
    parser.add_argument("--fig3c-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--layout", choices=("separate", "combined"), default="separate"
    )
    parser.add_argument("--focus-max-nt", type=float)
    args = parser.parse_args()
    if args.focus_max_nt is not None and (
        not math.isfinite(args.focus_max_nt)
        or args.focus_max_nt <= 0
        or args.layout != "combined"
    ):
        raise ValueError("focus-max-nt must be positive and requires combined layout")
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
    combined = args.layout == "combined"
    records = plot_context(
        *inputs, output, combined=combined, focus_max_nt=args.focus_max_nt
    )
    if [fingerprint(p) for p in inputs] != before:
        raise ValueError("source input changed")
    (output / "figure_contract_and_qa.json").write_text(
        json.dumps(
            {
                "claim": "Show positive-length distributions from distinct cohorts "
                "on one shared axis"
                if combined
                else "Show positive-length "
                "distributions in explicitly separate cohorts",
                "archetype": "quantitative grid",
                "backend": "python/matplotlib",
                "panels": ["single shared axis for four methods and RNA categories"]
                if combined
                else [
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
                "statistics": "Descriptive medians/IQR, 1.5-IQR whiskers, "
                "all outliers; means shown as circles in combined layout",
                "dimensions_inches": [16, 8] if combined else [15, 7],
                "layout": args.layout,
                "full_range_retained": True,
                "additional_focus_max_nt": args.focus_max_nt,
                "rna_categories_have_nominal_length": False,
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
