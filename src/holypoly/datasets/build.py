"""Build a matched head–Poly dataset with one command, without version names.

Default: read-only preflight. Add --run to prepare length/status weak labels,
matched reference-head sequences and optional ordinary-window timing features.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from holypoly.core.config import load_config
from holypoly.datasets import prepare, prepare_sequence
from holypoly.datasets.length_groups import parse_nominal_length_nt
from holypoly.datasets.prepare_sequence import sha256


def build_parser() -> argparse.ArgumentParser:
    """Expose one current workflow; historical structure choices are separate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--bonito-bam", type=Path, required=True)
    parser.add_argument("--dorado-bam", type=Path, required=True)
    parser.add_argument("--pod5-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("fixed_len_ref.fa"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--poly-types", nargs="+", choices=("polyA", "polyT"), required=True
    )
    parser.add_argument("--per-group", type=int, default=1000)
    parser.add_argument(
        "--max-records", type=int, default=10_000, help="Per pass; 0 scans all"
    )
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--exclude-read-ids", type=Path)
    parser.add_argument("--stop-when-full", action="store_true")
    parser.add_argument("--chunk-size-samples", type=int, default=12000)
    parser.add_argument("--min-candidate-samples", type=int, default=12)
    parser.add_argument("--anchor-bases", type=int, default=24)
    parser.add_argument("--anchor-error-rate", type=float, default=0.1)
    parser.add_argument("--target-bases", type=int, default=96)
    parser.add_argument("--margin-bases", type=int, default=32)
    parser.add_argument("--min-base-quality", type=int, default=15)
    parser.add_argument("--min-mapping-quality", type=int, default=20)
    parser.add_argument("--max-homopolymer", type=int, default=4)
    parser.add_argument("--max-signal-samples", type=int, default=4096)
    parser.add_argument("--with-timing-reference", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument(
        "--run", action="store_true", help="Execute the reported plan"
    )
    execution.add_argument("--dry-run", action="store_true", help="Read-only default")
    return parser


def _stage_args(
    args: argparse.Namespace,
    parser: argparse.ArgumentParser,
    required: list[str],
    **overrides: Any,
) -> argparse.Namespace:
    """Use the stage's own defaults, inheriting shared command settings."""
    stage = parser.parse_args(required)
    for key, value in vars(args).items():
        if hasattr(stage, key):
            setattr(stage, key, value)
    for key, value in overrides.items():
        setattr(stage, key, value)
    return stage


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Validate paths/settings and report resources without opening BAM or POD5."""
    config = load_config(
        args.config, cli_data_root=args.data_root, cli_work_root=args.work_root
    )
    if config.paths.data_root is None or config.paths.work_root is None:
        raise ValueError("configure data_root and work_root")
    data, work = config.paths.data_root.resolve(), config.paths.work_root.resolve()
    if not data.is_dir() or not work.is_dir():
        raise ValueError(
            f"configured roots must exist: data_root={data}, work_root={work}"
        )
    project = Path(__file__).resolve().parents[3]
    output = (work / args.output_dir).resolve()
    if (
        work.is_relative_to(data)
        or work.is_relative_to(project)
        or output == work
        or not output.is_relative_to(work)
        or output.is_relative_to(data)
        or output.is_relative_to(project)
        or output.exists()
        or (work / args.output_dir).is_symlink()
    ):
        raise ValueError(
            "choose a new external work-root output outside raw data/repository"
        )
    inputs = {}
    for key in ("bonito_bam", "dorado_bam", "pod5_dir", "reference"):
        path = (data / getattr(args, key)).resolve()
        valid = path.is_dir() if key == "pod5_dir" else path.is_file()
        if not valid or not path.is_relative_to(data):
            raise ValueError(f"{key} must exist inside data_root: {path}")
        inputs[key] = str(path)
    if (
        args.max_records < 0
        or args.per_group < 3
        or args.seed < 0
        or args.chunk_size_samples < 12
        or args.chunk_size_samples % 12
        or not 1 <= args.min_candidate_samples <= args.chunk_size_samples
        or args.anchor_bases < 12
        or not 0 <= args.anchor_error_rate < 0.5
        or args.target_bases < 8
        or args.margin_bases < 1
        or args.min_base_quality < 0
        or args.min_mapping_quality < 0
        or args.max_homopolymer < 1
        or args.max_signal_samples < 12
    ):
        raise ValueError("invalid scan, quota, seed or window settings")
    if args.exclude_read_ids is not None:
        exclusion = (work / args.exclude_read_ids).resolve()
        if (
            not exclusion.is_file()
            or not exclusion.is_relative_to(work)
            or exclusion.is_relative_to(data)
        ):
            raise ValueError(
                "exclude-read-ids must exist inside work_root outside raw data"
            )
    heads = prepare.load_heads(Path(inputs["reference"]))
    lengths = {parse_nominal_length_nt(name) for name in heads}
    joint_types = set(args.poly_types) == {"polyA", "polyT"}
    if joint_types and 0 not in lengths:
        raise ValueError("joint A/T balance requires a zero_nt reference construct")
    strata = len(lengths - {0}) * len(set(args.poly_types)) + (
        (2 if joint_types else 1) if 0 in lengths else 0
    )
    fingerprints = {
        key: {
            "bytes": Path(path).stat().st_size,
            "mtime_ns": Path(path).stat().st_mtime_ns,
        }
        for key, path in inputs.items()
        if key != "pod5_dir"
    }
    bam_bytes = (
        fingerprints["bonito_bam"]["bytes"]
        + (2 + int(args.with_timing_reference)) * fingerprints["dorado_bam"]["bytes"]
    )
    return {
        "workflow": "head-poly-dataset",
        "status": "planned",
        "data_root": str(data),
        "work_root": str(work),
        "output_dir": str(output),
        "inputs": inputs,
        "input_metadata": fingerprints,
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "balancing": "strict_length_type_zero_orientation"
        if joint_types
        else "per_length",
        "resources": {
            "max_records_per_pass": args.max_records,
            "bonito_passes": 1,
            "dorado_passes": 2 + int(args.with_timing_reference),
            "full_scan_bam_bytes": bam_bytes,
            "pod5_access": "selected UUID/sample-count metadata only; no signal copies",
            "gpu": False,
            "cpu": "sequential; memory grows with read-ID mappings/audit",
            "runtime_estimate": "bounded: seconds/minutes; full HDD: minutes/hours",
            "selected_reads_upper_bound": strata * args.per_group,
            "selected_tables_estimate_bytes": strata
            * args.per_group
            * (8000 if args.with_timing_reference else 3000),
            "audit_size_estimate": "hundreds of MB per million scanned reads",
        },
        "limitations": [
            "Construct-nominal weak targets; not verified per-read truth.",
            "Raw-edge candidate; Poly completeness/location remain unknown.",
            "BAM-prefix/quota sampling may be biased; a bounded scan may miss strata.",
            "New splits; do not reuse checkpoints exposed to these reads.",
        ],
    }


def run(
    args: argparse.Namespace,
    *,
    sequence_record_observer: Callable[[Any], None] | None = None,
) -> dict[str, Any]:
    """Prepare matched artifacts; record failed/incomplete stages without overwrites."""
    summary = preflight(args)
    print(json.dumps(summary, indent=2), flush=True)
    if not args.run:
        return summary
    output = Path(summary["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    metadata = output / "dataset.json"
    summary.update(
        status="building", started_utc=datetime.now(UTC).isoformat(), stages={}
    )

    def save() -> None:
        """Persist bundle state so a partial build cannot appear complete."""
        metadata.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    save()
    stage_name = "length"
    try:
        length_args = _stage_args(
            args,
            prepare.build_parser(),
            [
                "--bonito-bam",
                str(args.bonito_bam),
                "--dorado-bam",
                str(args.dorado_bam),
                "--pod5-dir",
                str(args.pod5_dir),
            ],
            data_root=Path(summary["data_root"]),
            work_root=Path(summary["work_root"]),
            output_dir=output / "length",
            structure="head-poly",
            signal_end_policy="raw-edge",
            strict_balance=set(args.poly_types) == {"polyA", "polyT"},
        )
        summary["stages"][stage_name] = prepare.prepare(length_args)
        save()
        manifest = output / "length/manifest.tsv"
        stage_name = "sequence"
        sequence_args = _stage_args(
            args,
            prepare_sequence.build_parser(),
            [
                "--manifest",
                str(manifest),
                "--dorado-bam",
                str(args.dorado_bam),
                "--output-dir",
                str(output / "sequence"),
            ],
            data_root=Path(summary["data_root"]),
            work_root=Path(summary["work_root"]),
            output_dir=output / "sequence",
            dry_run=False,
        )
        sequence = prepare_sequence.prepare(
            sequence_args, record_observer=sequence_record_observer
        )
        summary["stages"][stage_name] = sequence
        save()
        if any(
            sequence["split_counts"].get(split, 0) == 0
            for split in ("train", "validation")
        ):
            raise ValueError(
                "sequence supervision requires accepted train and validation "
                "windows; inspect sequence/audit.tsv"
            )
        artifacts = {
            "length_manifest": manifest,
            "sequence_manifest": output / "sequence/sequence_manifest.tsv",
        }
        if args.with_timing_reference:
            from holypoly.datasets import prepare_timing_reference

            stage_name = "timing"
            timing_args = _stage_args(
                args,
                prepare_timing_reference.build_parser(),
                [
                    "--manifest",
                    str(manifest),
                    "--bam",
                    str(args.dorado_bam),
                    "--output-dir",
                    str(output / "timing"),
                ],
                data_root=Path(summary["data_root"]),
                work_root=Path(summary["work_root"]),
                output_dir=output / "timing",
                scan_bam=True,
            )
            summary["stages"][stage_name] = prepare_timing_reference.run(timing_args)
            artifacts["timing_reference_manifest"] = (
                output / "timing/timing_reference.tsv"
            )
        stage_name = "verification"
        for key, before in summary["input_metadata"].items():
            stat = Path(summary["inputs"][key]).stat()
            if before != {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}:
                raise ValueError(f"input changed during preparation: {key}")
        summary.update(
            status="complete",
            finished_utc=datetime.now(UTC).isoformat(),
            artifacts={
                key: {"path": str(path), "sha256": sha256(path)}
                for key, path in artifacts.items()
            },
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
    """Run the unified command without loading models or starting training."""
    try:
        result = run(build_parser().parse_args(argv))
    except (OSError, ValueError, KeyError) as error:
        print(f"Dataset build failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
