"""Historical v3-to-v4 audit of head-to-terminal raw signal.

This bounded audit does not infer a Poly endpoint, completeness, or new labels.
Run in the Bonito environment; its waveform summaries can be plotted separately
in the parent environment without installing plotting packages into Bonito.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from holypoly.core.config import load_config


def sample_manifest(
    path: Path, per_group: int, seed: int
) -> tuple[list[dict[str, str]], dict[int, int]]:
    """Uniformly sample each nominal group from an existing, selected manifest."""
    if per_group < 1:
        raise ValueError("per_group must be positive")
    groups: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            groups[int(row["nominal_length_nt"])].append(row)
    rng = random.Random(seed)
    selected = []
    for length in sorted(groups):
        rows = sorted(groups[length], key=lambda row: row["read_id"])
        selected.extend(rng.sample(rows, min(per_group, len(rows))))
    return selected, {length: len(rows) for length, rows in sorted(groups.items())}


def waveform_preview(signal: Any, bins: int = 600) -> dict[str, Any]:
    """Summarise float32 pA samples as min/max bins in absolute raw coordinates."""
    import numpy as np

    edges = np.linspace(0, len(signal), min(bins, len(signal)) + 1, dtype=int)
    return {
        "sample": [
            int((a + b) // 2) for a, b in zip(edges[:-1], edges[1:], strict=True)
        ],
        "min_pa": [
            float(signal[a:b].min()) for a, b in zip(edges[:-1], edges[1:], strict=True)
        ],
        "max_pa": [
            float(signal[a:b].max()) for a, b in zip(edges[:-1], edges[1:], strict=True)
        ],
    }


def audit(args: argparse.Namespace) -> dict[str, Any]:
    """Read only sampled POD5 reads and write an auditable external report."""
    import numpy as np
    from pod5 import Reader

    config = load_config(args.config)
    if config.paths.data_root is None or config.paths.work_root is None:
        raise ValueError("data_root and work_root are required")
    data_root = config.paths.data_root.resolve()
    work_root = config.paths.work_root.resolve()
    output = (work_root / args.output_dir).resolve()
    project = Path(__file__).resolve().parents[3]
    if (
        not data_root.is_dir()
        or not work_root.is_dir()
        or not output.is_relative_to(work_root)
        or output == work_root
        or output.is_relative_to(data_root)
        or output.is_relative_to(project)
    ):
        raise ValueError("output must be a work_root child outside data_root/project")
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    sources = []
    selected = []
    dataset_names: set[str] = set()
    for manifest in args.manifests:
        path = (work_root / manifest).resolve()
        if not path.is_relative_to(work_root):
            raise ValueError("manifest must be inside work_root")
        dataset = path.parent.name
        if dataset in dataset_names:
            raise ValueError("manifest parent names must be distinct")
        dataset_names.add(dataset)
        rows, counts = sample_manifest(path, args.per_group, args.seed)
        sources.append({"manifest": str(path), "available_counts": counts})
        for row in rows:
            pod5_path = Path(row["pod5_path"]).resolve()
            if not pod5_path.is_relative_to(data_root) or not pod5_path.is_file():
                raise ValueError("sampled POD5 must exist inside data_root")
            selected.append((dataset, row, pod5_path))
    summary: dict[str, Any] = {
        "seed": args.seed,
        "per_group": args.per_group,
        "sources": sources,
        "selected_reads": len(selected),
        "estimated_signal_samples": sum(
            int(row["raw_signal_length"]) for _, row, _ in selected
        ),
        "output_dir": str(output),
        "dry_run": args.dry_run,
        "limitations": [
            "Sampling uses existing v3-selected reads, not all raw candidates.",
            "Old zero_nt reads were selected by the old flank rule.",
            "Head boundaries are inherited from v3, not independently validated.",
            "Raw read edges are candidate limits, not verified Poly endpoints.",
            "No completeness, adapter identity or new training labels inferred.",
        ],
    }
    if args.dry_run:
        return summary
    output.mkdir(parents=True)
    by_file: dict[Path, list[tuple[str, dict[str, str], int]]] = defaultdict(list)
    for index, (dataset, row, pod5_path) in enumerate(selected):
        by_file[pod5_path].append((dataset, row, index))
    samples = []
    for path, requests in by_file.items():
        with Reader(path) as reader:
            for dataset, row, index in requests:
                item: dict[str, Any] = {
                    "dataset": dataset,
                    "sample_index": index,
                    "read_id": row["read_id"],
                    "pod5_path": str(path),
                    "nominal_length_nt": int(row["nominal_length_nt"]),
                    "orientation": row["orientation"],
                    "old_supervision_mode": row["supervision_mode"],
                    "head_boundary_sample": int(row["head_boundary_sample"]),
                    "old_chunk_start_sample": int(row["chunk_start_sample"]),
                    "old_chunk_end_sample": int(row["chunk_end_sample"]),
                    "status": "ok",
                }
                matches = list(
                    reader.reads(selection=[row["read_id"]], missing_ok=True)
                )
                if len(matches) != 1:
                    item.update(status="failure", failure_reason="read_not_unique")
                    samples.append(item)
                    continue
                read = matches[0]
                signal = read.signal.astype(np.float32)
                signal = read.calibration.scale * (signal + read.calibration.offset)
                raw_length = len(signal)
                boundary = item["head_boundary_sample"]
                if raw_length != int(row["raw_signal_length"]) or not (
                    0 <= boundary <= raw_length
                ):
                    item.update(status="failure", failure_reason="coordinate_mismatch")
                    samples.append(item)
                    continue
                reverse = row["orientation"] == "reverse"
                start, end = (0, boundary) if reverse else (boundary, raw_length)
                candidate = signal[start:end]
                item.update(
                    raw_signal_length=raw_length,
                    raw_candidate_start_sample=start,
                    raw_candidate_end_sample=end,
                    raw_candidate_samples=end - start,
                    samples_outside_old_chunk=(
                        item["old_chunk_start_sample"] - start
                        if reverse
                        else end - item["old_chunk_end_sample"]
                    ),
                    exceeds_old_input_limit=end - start
                    > int(row["model_input_samples"]),
                    candidate_median_pa=(
                        float(np.median(candidate)) if len(candidate) else None
                    ),
                    waveform=waveform_preview(signal),
                )
                samples.append(item)
    summary.update(
        finished_utc=datetime.now(UTC).isoformat(),
        successful_reads=sum(item["status"] == "ok" for item in samples),
        failed_reads=sum(item["status"] != "ok" for item in samples),
        groups=[],
    )
    for dataset in sorted(dataset_names):
        lengths = sorted(
            {
                item["nominal_length_nt"]
                for item in samples
                if item["dataset"] == dataset
            }
        )
        for length in lengths:
            group = [
                item
                for item in samples
                if item["dataset"] == dataset and item["nominal_length_nt"] == length
            ]
            valid = [item for item in group if item["status"] == "ok"]
            summary["groups"].append(
                {
                    "dataset": dataset,
                    "nominal_length_nt": length,
                    "reads": len(group),
                    "successful_reads": len(valid),
                    "old_supervision_counts": dict(
                        Counter(item["old_supervision_mode"] for item in group)
                    ),
                    "median_samples_outside_old_chunk": (
                        float(
                            np.median([s["samples_outside_old_chunk"] for s in valid])
                        )
                        if valid
                        else None
                    ),
                    "exceeds_old_input_limit_reads": sum(
                        item["exceeds_old_input_limit"] for item in valid
                    ),
                }
            )
    (output / "samples.json").write_text(json.dumps(samples) + "\n", encoding="utf-8")
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> None:
    """Run the bounded, explicitly requested signal audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifests", nargs="+", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--dry-run", action="store_true")
    print(json.dumps(audit(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
