"""Plot Holypoly alone and compare unlabeled poly(A) outputs with Dorado."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from holypoly.core.config import load_config  # noqa: E402

TOOLS = ("Holypoly", "Dorado")
COLORS = {"Holypoly": "#4477AA", "Dorado": "#CC9933"}


def render(output: Path, sample: str, bin_width_nt: float) -> None:
    """Render positive-output histograms/CDFs, with full common-read states."""
    summary = json.loads((output / "summary.json").read_text())
    with (output / "holypoly_lengths.tsv").open() as stream:
        holo = [
            float(r["positive_length_nt"])
            for r in csv.DictReader(stream, delimiter="\t")
        ]
    common = {tool: [] for tool in TOOLS}
    paired = {tool: [] for tool in TOOLS}
    count = 0
    with (output / "common_read_comparison.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            count += 1
            for tool in TOOLS:
                if row[f"{tool}_positive_length_nt"]:
                    common[tool].append(float(row[f"{tool}_positive_length_nt"]))
            if all(row[f"{t}_positive_length_nt"] for t in TOOLS):
                for tool in TOOLS:
                    paired[tool].append(float(row[f"{tool}_positive_length_nt"]))
    if count != summary["common_reads"]:
        raise ValueError("extracted common cohort and summary disagree")
    for values in (holo, *common.values(), *paired.values()):
        if any(not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError("positive-output tables contain invalid lengths")
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

    def bins_for(values):
        """Keep all outputs, with a bounded number of uniform nt bins."""
        number = max(1, math.ceil(max(values, default=1) / bin_width_nt))
        if number > 10000:
            raise ValueError("too many bins; increase --bin-width-nt")
        return np.arange(number + 1) * bin_width_nt

    def cdf(axis, values, tool):
        """Plot the exact empirical CDF conditional on positive outputs."""
        if values:
            ordered = np.sort(values)
            axis.step(
                np.r_[0, ordered],
                np.r_[0, np.arange(1, len(values) + 1) / len(values)],
                where="post",
                color=COLORS[tool],
                linewidth=1.4,
                label=f"{tool} (n={len(values):,})",
            )
        axis.set(
            xlabel="Reported poly(A) length (nt)",
            ylabel="Cumulative fraction of positive outputs",
            ylim=(0, 1.02),
        )
        axis.grid(alpha=0.15)
        if axis.lines:
            axis.legend(fontsize=8)
        else:
            axis.text(
                0.5, 0.5, "No positive outputs", ha="center", transform=axis.transAxes
            )

    def export(figure, name):
        """Save PNG300dpi and editable PDF/SVG without touching inputs."""
        for suffix in ("png", "pdf", "svg"):
            figure.savefig(output / f"{name}.{suffix}", dpi=300, facecolor="white")
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.24, top=0.82, wspace=0.30)
    if holo:
        axes[0].hist(holo, bins=bins_for(holo), color=COLORS["Holypoly"], alpha=0.75)
    axes[0].set(
        xlabel="Predicted poly(A) length (nt)",
        ylabel="Positive-output reads",
        title="a  Holypoly length distribution",
    )
    cdf(axes[1], holo, "Holypoly")
    axes[1].set_title("b  Holypoly cumulative distribution")
    n = summary["tools"]["Holypoly"]["all_primary_reads"]
    figure.suptitle(f"{sample}: Holypoly poly(A), {len(holo):,}/{n:,} positive outputs")
    figure.text(
        0.08,
        0.10,
        "All selected Holypoly primary reads; positive outputs only. "
        "Missing/no-candidate/zero outputs are not biological zero tails.",
        fontsize=8,
    )
    figure.text(
        0.08,
        0.055,
        "Candidate-conditioned distribution; completeness unknown. "
        "No ground-truth accuracy or biological recall is inferred.",
        fontsize=8,
    )
    export(figure, "holypoly_polya_distribution")

    figure, axes = plt.subplots(2, 2, figsize=(12, 8.2))
    figure.subplots_adjust(
        left=0.08, right=0.98, bottom=0.24, top=0.87, hspace=0.53, wspace=0.30
    )
    figure.suptitle(
        f"{sample}: Holypoly and Dorado on {count:,} common read IDs", fontsize=13
    )
    bins = bins_for([*common["Holypoly"], *common["Dorado"]])
    for tool in TOOLS:
        values = common[tool]
        if values:
            axes[0, 0].hist(
                values,
                bins=bins,
                weights=np.full(len(values), 100 / len(values)),
                histtype="step",
                color=COLORS[tool],
                linewidth=1.5,
                label=f"{tool} (n={len(values):,})",
            )
        cdf(axes[0, 1], values, tool)
        cdf(axes[1, 0], paired[tool], tool)
    axes[0, 0].set(
        title="a  Overlaid length distributions",
        xlabel="Reported poly(A) length (nt)",
        ylabel="Positive outputs per bin (%)",
    )
    if axes[0, 0].get_legend_handles_labels()[0]:
        axes[0, 0].legend(fontsize=8)
    axes[0, 1].set_title("b  CDF: each tool's positive outputs")
    axes[1, 0].set_title(
        f"c  CDF: both positive (n={summary['common_positive_reads']:,})"
    )
    categories = (
        ("positive_length", "Positive length", None),
        ("zero_output", "Native zero output", "#E4CD9C"),
        ("no_candidate", "No candidate", "#C7CCD1"),
        ("other_or_ambiguous", "Other / ambiguous", "#9C91AD"),
        ("error_or_missing", "Error / missing", "#565B61"),
    )
    for index, tool in enumerate(TOOLS):
        left = 0
        counts = summary["tools"][tool]["common_status_counts"]
        if sum(counts.values()) != count:
            raise ValueError("status stacks do not match common cohort")
        for category, _, color in categories:
            width = 100 * counts.get(category, 0) / count
            axes[1, 1].barh(
                index, width, left=left, color=color or COLORS[tool], height=0.55
            )
            if width >= 10:
                axes[1, 1].text(
                    left + width / 2,
                    index,
                    f"{width:.1f}%",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if category == "positive_length" else "#222222",
                )
            left += width
    axes[1, 1].set(
        title="d  All common reads: native output states",
        yticks=range(2),
        yticklabels=TOOLS,
        xlim=(0, 100),
        xlabel="Common reads (%)",
    )
    axes[1, 1].invert_yaxis()
    axes[1, 1].legend(
        handles=[
            Patch(facecolor=c or "#777777", label=label) for _, label, c in categories
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.23),
        ncol=3,
        fontsize=7,
    )
    notes = (
        "a/b: same input read IDs, tool-specific positive subsets. "
        "c: identical read IDs with positive outputs from both tools.",
        "d: every common read is retained; stacks sum to 100%. "
        "Native zero / no candidate / error outputs "
        "are never imputed as biological zero tails.",
        "Histogram percentages sum to 100% per tool; uniform nt bins. "
        "All positive lengths and extremes are retained, with no nominal labels.",
        "Unlabeled cDNA: distributions and output coverage only; "
        "no biological recall, length accuracy, or tool ranking is established.",
    )
    for index, note in enumerate(notes):
        figure.text(0.08, 0.115 - index * 0.025, note, fontsize=8)
    export(figure, "holypoly_dorado_polya_comparison")
    (output / "plot_qa.json").write_text(
        json.dumps(
            {
                "claim": "Compare native poly(A) output distributions and coverage",
                "backend": "python/matplotlib",
                "archetype": "quantitative grid",
                "bin_width_nt": bin_width_nt,
                "common_reads": count,
                "common_positive_reads": summary["common_positive_reads"],
                "all_positive_extremes_retained": True,
                "zero_imputation": False,
                "accuracy_or_recall_computed": False,
                "exports": ["PNG300dpi", "PDF", "editable-text SVG"],
            },
            indent=2,
        )
        + "\n"
    )


def main() -> None:
    """Extract BAM tags in the existing Bonito environment, then plot on CPU."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    parser.add_argument("--holypoly", type=Path, required=True)
    parser.add_argument("--dorado-bam", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-name", default="cDNA")
    parser.add_argument("--bin-width-nt", type=float, default=5)
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.sample_name):
        raise ValueError("sample-name must be a simple dataset label")
    if not math.isfinite(args.bin_width_nt) or args.bin_width_nt <= 0:
        raise ValueError("bin-width-nt must be positive and finite")
    project = Path(__file__).resolve().parents[1]
    python = project / "src/bonito/.venv/bin/python"
    if not python.is_file():
        raise FileNotFoundError("existing Bonito Python environment is required")
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = str(project / "src") + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(
        [
            str(python),
            "-m",
            "holypoly.evaluation.cdna",
            "--config",
            str(args.config),
            "--holypoly",
            str(args.holypoly),
            "--dorado-bam",
            str(args.dorado_bam),
            "--output",
            str(args.output),
        ],
        env=env,
        check=True,
    )
    config = load_config(args.config)
    assert config.paths.work_root is not None
    output = (config.paths.work_root / args.output).resolve()
    render(output, args.sample_name, args.bin_width_nt)
    print(f"Finished: {output}")


if __name__ == "__main__":
    main()
