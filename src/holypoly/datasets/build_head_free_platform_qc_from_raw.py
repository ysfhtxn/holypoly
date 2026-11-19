"""NEW head-free-platform-qc-v1: whole head-free input, platform QC, metadata zeros.

Start from raw BAM/POD5/reference. Positive reads retain the current strict
platform admission checks, while their entire original head-free interval is
the training input. Zero constructs retain metadata-qualified weak negatives
without signal/platform/timing checks, including reads without head sequences.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from holypoly.datasets import build
from holypoly.datasets import build_strict_poly_from_raw as pipeline
from holypoly.datasets.build_strict_poly_dataset import DatasetMethodPolicy

POLICY = DatasetMethodPolicy(
    method="head-free-platform-qc-v1",
    summary_filename="head_free_qc_dataset.json",
    raw_workflow="head-free-platform-qc-v1-from-raw",
    raw_summary_filename="head_free_qc_raw_dataset.json",
    dataset_subdirectory="head_free_qc",
    entry_filename="build_head_free_platform_qc_from_raw.py",
    full_head_free_input=True,
    metadata_only_zero=True,
)


def build_parser() -> argparse.ArgumentParser:
    """Expose a distinct raw-data entry; candidate crop options do not apply."""
    parser = build.build_parser()
    parser.description = __doc__
    parser.set_defaults(
        per_group=300, max_records=100_000, seed=20261117, crop_context_ms=0.0
    )
    parser.add_argument(
        "--rules-json",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "configs/head_free_platform_qc_v1.json",
        help="Positive platform QC rules only; zero labels are metadata-only",
    )
    parser.add_argument(
        "--target-per-group",
        type=int,
        help="Final reads per nominal length/direction, split 80/10/10",
    )
    parser.add_argument(
        "--max-signal-reads",
        type=int,
        default=2100,
        help="Positive signal attempts only; zero reads are not decoded",
    )
    parser.add_argument("--max-raw-signal-samples", type=int, default=120_000)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pod5-reader-cache-size", type=int, default=8)
    parser.add_argument("--allow-full-bam-scan", action="store_true")
    return parser


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Plan without BAM/POD5 scans, source manifests or output writes."""
    return pipeline.preflight(args, policy=POLICY)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Build new matched artifacts while preserving the two earlier methods."""
    return pipeline.run(args, policy=POLICY)


def main(argv: Sequence[str] | None = None) -> int:
    """Plan by default; require a new output and explicit --run for actual work."""
    try:
        result = run(build_parser().parse_args(argv))
    except (ValueError, OSError, KeyError) as error:
        print(f"Head-free platform QC build failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"planned", "complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
