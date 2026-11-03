"""Plot unlabeled candidate lengths without inventing nominal length groups."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from holypoly.core.config import load_config  # noqa: E402


def main() -> None:
    """Export per-type histograms/CDFs and retain failure denominators in metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-width-nt", type=float, default=5)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.paths.work_root is None or config.paths.data_root is None:
        raise ValueError("data_root and work_root are required")
    work, data = config.paths.work_root.resolve(), config.paths.data_root.resolve()
    source, output = (work / args.predictions).resolve(), (work / args.output).resolve()
    project = Path(__file__).resolve().parents[1]
    if (
        not output.is_relative_to(work)
        or output == work
        or output.is_relative_to(data)
        or output.is_relative_to(project)
        or source.is_relative_to(output)
        or output.exists()
        or output.is_symlink()
    ):
        raise ValueError(
            "output must be a new external work-root directory outside inputs"
        )
    if not math.isfinite(args.bin_width_nt) or args.bin_width_nt <= 0:
        raise ValueError("bin-width-nt must be positive and finite")
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    values: dict[str, list[float]] = {}
    for row in rows:
        if row["status"] == "candidate_predicted":
            value = float(row["predicted_length_nt"])
            if not math.isfinite(value) or value < 0:
                raise ValueError("invalid exported length")
            values.setdefault(row["poly_type"], []).append(value)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for poly_type, lengths in sorted(values.items()):
        number_bins = max(1, math.ceil(max(lengths) / args.bin_width_nt))
        # Keep an anomalous model prediction from allocating millions of bins.
        if number_bins > 10000:
            raise ValueError(
                "too many bins; inspect predictions or increase bin-width-nt"
            )
        bins = [index * args.bin_width_nt for index in range(number_bins + 1)]
        axes[0].hist(
            lengths,
            bins=bins,
            histtype="step",
            linewidth=1.5,
            label=f"{poly_type} (n={len(lengths)})",
        )
        ordered = sorted(lengths)
        axes[1].step(
            ordered,
            [(index + 1) / len(ordered) for index in range(len(ordered))],
            where="post",
            label=poly_type,
        )
    axes[0].set(xlabel="Predicted length (nt)", ylabel="Candidate reads")
    axes[1].set(
        xlabel="Predicted length (nt)", ylabel="Cumulative fraction", ylim=(0, 1)
    )
    if values:
        for axis in axes:
            axis.legend()
    else:
        for axis in axes:
            axis.text(
                0.5,
                0.5,
                "No predicted candidates",
                ha="center",
                transform=axis.transAxes,
            )
    predicted = sum(map(len, values.values()))
    figure.suptitle(
        f"Exploratory cDNA candidates: {predicted}/{len(rows)} selected BAM reads"
    )
    figure.supxlabel(
        "Sequence-assisted candidates; unknown completeness; no ground-truth accuracy.",
        fontsize=9,
    )
    output.mkdir(parents=True, exist_ok=False)
    for extension in ("png", "svg", "pdf"):
        figure.savefig(output / f"length_distribution.{extension}", dpi=180)
    plt.close(figure)
    (output / "plot_summary.json").write_text(
        json.dumps(
            {
                "source_predictions": str(source),
                "selected_primary_bam_reads": len(rows),
                "predicted_candidates": predicted,
                "poly_types": sorted(values),
                "status_counts": dict(Counter(row["status"] for row in rows)),
                "bin_width_nt": args.bin_width_nt,
                "notes": (
                    "Candidate-conditioned distribution, not all biological tails; "
                    "missing predictions are not zero nt."
                ),
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
