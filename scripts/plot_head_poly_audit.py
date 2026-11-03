"""Plot the bounded raw-signal audit using the parent plotting environment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from holypoly.core.config import load_config  # noqa: E402


def main() -> None:
    """Write one contact sheet per dataset and nominal length inside work_root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--audit-dir", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    if config.paths.work_root is None or config.paths.data_root is None:
        raise ValueError("data_root/work_root are required")
    root = config.paths.work_root.resolve()
    directory = (root / args.audit_dir).resolve()
    project = Path(__file__).resolve().parents[1]
    if (
        not directory.is_relative_to(root)
        or directory == root
        or directory.is_relative_to(config.paths.data_root.resolve())
        or directory.is_relative_to(project)
    ):
        raise ValueError(
            "audit directory must be outside data_root/project in work_root"
        )
    samples = json.loads((directory / "samples.json").read_text())
    keys = sorted({(s["dataset"], s["nominal_length_nt"]) for s in samples})
    for dataset, length in keys:
        group = [
            s
            for s in samples
            if s["dataset"] == dataset
            and s["nominal_length_nt"] == length
            and s["status"] == "ok"
        ]
        if not group:
            continue
        figure, axes = plt.subplots(
            (len(group) + 3) // 4,
            4,
            figsize=(18, 2.6 * ((len(group) + 3) // 4)),
            squeeze=False,
        )
        for axis, sample in zip(axes.flat, group, strict=False):
            preview = sample["waveform"]
            axis.fill_between(
                preview["sample"],
                preview["min_pa"],
                preview["max_pa"],
                color="0.25",
                linewidth=0,
                alpha=0.7,
            )
            axis.axvspan(
                sample["raw_candidate_start_sample"],
                sample["raw_candidate_end_sample"],
                color="orange",
                alpha=0.2,
            )
            axis.axvspan(
                sample["old_chunk_start_sample"],
                sample["old_chunk_end_sample"],
                color="tab:green",
                alpha=0.25,
            )
            axis.axvline(sample["head_boundary_sample"], color="tab:red", linewidth=1)
            axis.set_title(
                f"Sample {sample['sample_index']} | {sample['old_supervision_mode']}"
            )
            axis.set_xlabel("Raw sample")
            axis.set_ylabel("Current (pA)")
        for axis in list(axes.flat)[len(group) :]:
            axis.set_visible(False)
        figure.suptitle(
            f"{dataset} | nominal {length} nt\n"
            "Red: inherited head boundary; green: v3 input; "
            "orange: candidate to raw edge (endpoint unverified)",
            fontsize=12,
        )
        figure.tight_layout(rect=(0, 0, 1, 0.95))
        figure.savefig(directory / f"{dataset}-{length}nt.png", dpi=130)
        plt.close(figure)
    print(f"Wrote audit contact sheets to {directory}")


if __name__ == "__main__":
    main()
