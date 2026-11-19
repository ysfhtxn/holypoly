"""NEW: raw BAM/POD5/reference -> fresh strict-poly-candidate-v1 manifests.

No existing manifest is required. Reuse head/move/sequence metadata preparation,
then apply the strict signal rules. No training or model loading is performed.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from holypoly.datasets import build
from holypoly.datasets import build_strict_poly_dataset as strict
from holypoly.datasets.length_groups import parse_nominal_length_nt
from holypoly.datasets.prepare import load_heads
from holypoly.datasets.prepare_sequence import sha256

WORKFLOW = "strict-poly-candidate-v1-from-raw"


def build_parser() -> argparse.ArgumentParser:
    """Reuse raw metadata options; add bounded signal QC and explicit full scans."""
    parser = build.build_parser()
    parser.description = __doc__
    parser.set_defaults(per_group=300, max_records=100_000, seed=20261117)
    parser.add_argument(
        "--rules-json",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "configs/strict_poly_candidate_v1.json",
    )
    parser.add_argument("--max-signal-reads", type=int, default=2100)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU detector processes; start with 4 on HDD",
    )
    parser.add_argument(
        "--target-per-group",
        type=int,
        help="Post-QC target per length/direction; stop signals when all splits fill",
    )
    parser.add_argument("--max-raw-signal-samples", type=int, default=120_000)
    parser.add_argument("--pod5-reader-cache-size", type=int, default=4)
    parser.add_argument("--crop-context-ms", type=float, default=15.0)
    parser.add_argument("--allow-full-bam-scan", action="store_true")
    return parser


def _pool_args(args: argparse.Namespace, output: Path) -> argparse.Namespace:
    """Create a NEW source pool in this run, never read a historical manifest."""
    values = vars(args).copy()
    values.update(output_dir=output / "metadata_pool", with_timing_reference=False)
    return argparse.Namespace(**values)


def preflight(
    args: argparse.Namespace,
    *,
    policy: strict.DatasetMethodPolicy = strict.STRICT_METHOD_POLICY,
) -> dict[str, Any]:
    """Read small reference/config only, stat inputs, and report all three passes."""
    if args.with_timing_reference:
        raise ValueError("timing is used for offline QC, not a TCN timing feature")
    if args.max_records == 0 and not args.allow_full_bam_scan:
        raise ValueError("--max-records 0 requires --allow-full-bam-scan")
    # Existing preflight validates roots, all raw inputs and metadata settings.
    initial = build.preflight(args)
    output = Path(initial["output_dir"])
    data = Path(initial["data_root"])
    if data.is_relative_to(output):
        raise ValueError("output cannot contain raw data")
    if (
        args.max_signal_reads < 1
        or args.max_raw_signal_samples < 12
        or args.workers < 1
        or args.pod5_reader_cache_size < 0
        or not 1 <= args.min_mapping_quality < 255
        or len(set(args.poly_types)) != len(args.poly_types)
    ):
        raise ValueError("signal budgets must be positive")
    if not math.isfinite(args.crop_context_ms) or args.crop_context_ms < 0:
        raise ValueError("crop-context-ms must be finite and nonnegative")
    rules = strict._load_rules(args.rules_json)
    heads = load_heads(Path(initial["inputs"]["reference"]))
    lengths = sorted(
        {
            value
            for name in heads
            if (value := parse_nominal_length_nt(name)) is not None
        }
    )
    if 0 not in lengths or not any(length > 0 for length in lengths):
        raise ValueError("reference must include zero and positive constructs")
    upper = min(
        args.max_signal_reads,
        (len(lengths) - int(policy.metadata_only_zero))
        * len(args.poly_types)
        * args.per_group,
    )
    target_quotas = strict._target_quotas(
        args.target_per_group,
        args.per_group,
        args.max_signal_reads,
        (len(lengths) - int(policy.metadata_only_zero)) * len(args.poly_types),
    )
    return {
        "workflow": policy.raw_workflow,
        "method_policy": asdict(policy),
        "selection_scope": strict.SELECTION_SCOPE,
        "target_split_quotas_per_length_direction": target_quotas,
        "status": "planned",
        "output_dir": str(output),
        "training_dataset_dir": str(output / policy.dataset_subdirectory),
        "data_root": initial["data_root"],
        "work_root": initial["work_root"],
        "inputs": initial["inputs"],
        "input_metadata": initial["input_metadata"],
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "nominal_lengths_nt": lengths,
        "rules_json_sha256": sha256(args.rules_json),
        "candidate_rules": vars(rules),
        "resources": {
            **initial["resources"],
            "bonito_passes": 1,
            "dorado_passes": 2,
            "full_scan_bam_bytes": initial["resources"]["full_scan_bam_bytes"],
            "move_collection": "fused with this run's reference-head sequence scan",
            "selected_signal_reads_upper_bound": upper,
            "open_pod5_readers_cap": args.pod5_reader_cache_size,
            "signal_workers": args.workers,
            "in_flight_signal_cap": args.workers,
            "cpu": f"metadata sequential; {args.workers} ordered signal CPU worker(s)",
            "native_threads_per_worker": 1 if args.workers > 1 else "inherited",
            "target_selected_reads": args.target_per_group
            * len(lengths)
            * len(args.poly_types)
            if args.target_per_group is not None
            else None,
            "decoded_samples_upper_bound": upper * args.max_raw_signal_samples,
            "candidate_metadata_estimate_bytes": upper * 2_000_000,
            "pod5_access": "new-pool UUID metadata, selected signals; no raw copies",
            "runtime_estimate": (
                "bounded: minutes; full HDD scans plus signal QC: tens of minutes/hours"
            ),
            "gpu": False,
            "model": False,
        },
        "limitations": [
            "No old manifest or old read selection is imported.",
            "Head/sequence metadata eligibility uses existing audited helpers.",
            "The raw-edge metadata pool still excludes oversized head-free chunks.",
            "New seeded splits are assigned before strict signal QC and inherited.",
            "Nominal length/status stay construct-level weak supervision.",
            "Do not reuse a checkpoint exposed to the newly assigned splits.",
            "A bounded BAM prefix or stop-when-full pool is not a population sample.",
            "Signal early stopping is not uniform sampling of QC-passing reads.",
            "Strict thresholds are initial; no automatic tuning or fallback.",
        ],
    }


def _strict_args(
    args: argparse.Namespace,
    summary: dict[str, Any],
    *,
    policy: strict.DatasetMethodPolicy = strict.STRICT_METHOD_POLICY,
) -> argparse.Namespace:
    """Bind the new metadata pool to the existing strict exporter interface."""
    output = Path(summary["output_dir"])
    stage = strict.build_parser().parse_args(
        [
            "--source-manifest",
            str(output / "metadata_pool/length/manifest.tsv"),
            "--source-sequence-manifest",
            str(output / "metadata_pool/sequence/sequence_manifest.tsv"),
            "--dorado-bam",
            str(args.dorado_bam),
            "--output-dir",
            str(output / policy.dataset_subdirectory),
            "--poly-types",
            *args.poly_types,
            "--nominal-lengths",
            *map(str, summary["nominal_lengths_nt"]),
        ]
    )
    stage.config = args.config
    stage.data_root, stage.work_root = (
        Path(summary["data_root"]),
        Path(summary["work_root"]),
    )
    stage.pool_per_group, stage.seed = args.per_group, args.seed
    stage.target_per_group = args.target_per_group
    stage.max_source_rows = (
        len(summary["nominal_lengths_nt"]) * len(args.poly_types) * args.per_group
    )
    stage.max_bam_records = args.max_records
    stage.allow_full_bam_scan = args.allow_full_bam_scan
    stage.max_signal_reads, stage.max_signal_samples = (
        args.max_signal_reads,
        args.max_raw_signal_samples,
    )
    stage.model_input_samples = args.chunk_size_samples
    stage.min_mapping_quality = args.min_mapping_quality
    stage.crop_context_ms, stage.rules_json = args.crop_context_ms, args.rules_json
    stage.pod5_reader_cache_size = args.pod5_reader_cache_size
    stage.workers = args.workers
    stage.no_progress, stage.run = args.no_progress, True
    return stage


def run(
    args: argparse.Namespace,
    *,
    policy: strict.DatasetMethodPolicy = strict.STRICT_METHOD_POLICY,
) -> dict[str, Any]:
    """Create an independent raw-data build, keeping failed stages reviewable."""
    summary = preflight(args, policy=policy)
    print(json.dumps(summary, indent=2), flush=True)
    if not args.run:
        return summary
    output = Path(summary["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    summary.update(
        status="building", started_utc=datetime.now(UTC).isoformat(), stages={}
    )

    def save() -> None:
        """Never leave a failed or interrupted pipeline marked complete."""
        (output / policy.raw_summary_filename).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

    save()
    stage_name = "metadata_pool"
    strict_args = _strict_args(args, summary, policy=policy)
    collector: strict.SelectedMoveCollector | None = None
    display_name = (
        "Head-free platform QC" if policy.full_head_free_input else "Strict from raw"
    )

    def collect_moves(record: Any) -> None:
        """Initialize only after the NEW length pool exists; never read old tables."""
        nonlocal collector
        if collector is None:
            manifest = output / "metadata_pool/length/manifest.tsv"
            sampled, _ = strict._sample_sources(strict_args, manifest)
            collector = strict.SelectedMoveCollector(
                strict_args,
                Path(summary["inputs"]["dorado_bam"]),
                {
                    row["read_id"]: row
                    for row in sampled
                    if not (
                        policy.metadata_only_zero
                        and float(row["nominal_length_nt"]) == 0
                    )
                },
                sha256(manifest),
            )
        collector.observe(record)

    try:
        summary["current_stage"] = stage_name
        save()
        if not args.no_progress:
            print(
                f"[{display_name}] metadata_pool started", file=sys.stderr, flush=True
            )
        summary["stages"][stage_name] = build.run(
            _pool_args(args, output), sequence_record_observer=collect_moves
        )
        save()
        stage_name = policy.dataset_subdirectory
        summary["current_stage"] = stage_name
        save()
        if not args.no_progress:
            print(
                f"[{display_name}] {stage_name} signal QC started",
                file=sys.stderr,
                flush=True,
            )
        if collector is None:
            raise ValueError("no sequence-scan move collector; inspect metadata pool")
        result = strict.run(strict_args, move_collector=collector, policy=policy)
        summary["stages"][stage_name] = result
        for key, before in summary["input_metadata"].items():
            path = Path(summary["inputs"][key])
            if before != strict._fingerprint(path):
                raise ValueError(f"raw input changed across stages: {key}")
        if sha256(args.rules_json) != summary["rules_json_sha256"]:
            raise ValueError("strict rules changed across stages")
        summary.update(
            status=result["status"],
            current_stage="finished",
            selected_reads=result["selected_reads"],
            split_counts=result["split_counts"],
            missing_strata=result["missing_strata"],
            sequence_reads=result.get("sequence_reads"),
            artifacts=result.get("artifacts", {}),
            finished_utc=datetime.now(UTC).isoformat(),
            entry_code_sha256=sha256(Path(__file__).with_name(policy.entry_filename)),
        )
        save()
    except Exception as error:
        summary.update(
            status="failed",
            failed_stage=stage_name,
            error=str(error),
            finished_utc=datetime.now(UTC).isoformat(),
        )
        save()
        raise
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Plan by default; nonzero on failure or incomplete strata; never train."""
    try:
        result = run(build_parser().parse_args(argv))
    except (ValueError, OSError, KeyError) as error:
        print(f"Strict Poly from raw failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"planned", "complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
