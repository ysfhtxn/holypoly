"""Export balanced, read-disjoint construction weak-label reads as full POD5."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

from holypoly.core.config import load_config


def file_sha256(path: Path) -> str:
    """Hash a metadata file without loading it all into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def history_exclusions(
    manifests: Sequence[Path], *, all_splits: bool = False
) -> tuple[set[str], list[dict[str, Any]]]:
    """Union historical read IDs, including sequence-supervision manifests.

    By default exclude train and validation; all_splits also reserves old tests.
    Repeated chunks in one split are allowed; cross-split duplicates are errors.
    """
    excluded: set[str] = set()
    provenance = []
    for path in sorted(set(manifests)):
        seen: dict[str, str] = {}
        counts: Counter[str] = Counter()
        with path.open(newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            if not {"read_id", "split"}.issubset(reader.fieldnames or []):
                raise ValueError(f"history manifest lacks read_id/split: {path}")
            for row in reader:
                read_id = str(UUID(row["read_id"]))
                split = row["split"]
                if split not in {"train", "validation", "test"}:
                    raise ValueError(f"unsupported historical split: {path}")
                if read_id in seen and seen[read_id] != split:
                    raise ValueError(f"historical read crosses splits: {path}")
                seen[read_id] = split
                counts[split] += 1
                if all_splits or split in {"train", "validation"}:
                    excluded.add(read_id)
        provenance.append(
            {"path": str(path), "sha256": file_sha256(path), "rows": dict(counts)}
        )
    if not excluded:
        raise ValueError("historical exclusion set is empty")
    return excluded, provenance


def select_balanced(
    rows: list[dict[str, str]],
    excluded: set[str],
    *,
    lengths: Sequence[int],
    per_group: int,
    seed: int,
) -> tuple[list[dict[str, str]], dict[int, int]]:
    """Sample equally by nominal length, including zero_nt as its own negative group."""
    if (
        per_group < 1
        or seed < 0
        or not lengths
        or len(set(lengths)) != len(lengths)
        or min(lengths) < 0
    ):
        raise ValueError("invalid count, seed or nominal lengths")
    groups: dict[int, list[dict[str, str]]] = {length: [] for length in lengths}
    seen = set()
    for row in rows:
        read_id = str(UUID(row["read_id"]))
        if read_id in seen:
            raise ValueError("duplicate source read_id")
        seen.add(read_id)
        if read_id in excluded:
            continue
        length = int(row["nominal_length_nt"])
        if length not in groups:
            continue
        if not row.get("label_source"):
            raise ValueError("weak labels must retain label_source")
        if row.get("orientation") != "forward":
            raise ValueError("polyA-only export requires forward construct orientation")
        expected_type, expected_status = (
            ("none", "zero_nt") if length == 0 else ("polyA", "poly_present")
        )
        if row["poly_type"] != expected_type or row["label_status"] != expected_status:
            raise ValueError("source nominal length, Poly type and status disagree")
        groups[length].append(row)
    counts = {length: len(group) for length, group in groups.items()}
    count = min(per_group, *counts.values())
    if count < 1:
        raise ValueError(f"missing independent nominal-length group: {counts}")
    rng = random.Random(seed)
    selected = []
    for length in sorted(groups):
        pool = sorted(groups[length], key=lambda row: row["read_id"])
        selected.extend(rng.sample(pool, count))
    return selected, counts


def _signature(record: Any) -> dict[str, Any]:
    """Fingerprint original int16 ADC samples and calibration/acquisition metadata."""
    return {
        "num_samples": record.num_samples,
        "signal_sha256": hashlib.sha256(record.signal.tobytes()).hexdigest(),
        "start_sample": record.start_sample,
        "read_number": record.read_number,
        "calibration_offset": record.calibration.offset,
        "calibration_scale": record.calibration.scale,
        "acquisition_id": record.run_info.acquisition_id,
        "sample_rate": record.run_info.sample_rate,
        "channel": record.pore.channel,
        "well": record.pore.well,
        "median_before_hex": float(record.median_before).hex(),
        "end_reason": record.end_reason.reason.name,
        "end_reason_forced": record.end_reason.forced,
    }


def export_full_reads(
    selected: list[dict[str, str]], output: Path, *, data_root: Path
) -> dict[str, dict[str, Any]]:
    """Copy whole reads without cropping/rescaling; verify every exported signal."""
    from pod5 import Reader, Writer

    by_file: dict[Path, list[str]] = defaultdict(list)
    for row in selected:
        path = (data_root / row["pod5_path"]).resolve()
        if not path.is_file() or not path.is_relative_to(data_root):
            raise ValueError("source POD5 must exist inside data_root")
        by_file[path].append(row["read_id"])
    source_stats = {
        path: (path.stat().st_size, path.stat().st_mtime_ns) for path in by_file
    }
    signatures: dict[str, dict[str, Any]] = {}
    with Writer(output, software_name="holypoly-balanced-final-holdout") as writer:
        for path, identifiers in sorted(by_file.items()):
            with Reader(path) as reader:
                for record in reader.reads(selection=identifiers, missing_ok=False):
                    read_id = str(record.read_id)
                    if read_id in signatures:
                        raise ValueError("duplicate selected read in source POD5")
                    signatures[read_id] = _signature(record)
                    writer.add_reads([record.to_read()])
    expected = {row["read_id"] for row in selected}
    if set(signatures) != expected:
        raise ValueError("not every selected read was exported")
    with Reader(output) as reader:
        if len(reader.read_ids) != len(expected) or set(reader.read_ids) != expected:
            raise ValueError("exported POD5 read IDs differ from selected labels")
        for record in reader.reads():
            if _signature(record) != signatures[str(record.read_id)]:
                raise ValueError("exported signal or acquisition metadata changed")
    for path, before in source_stats.items():
        if (path.stat().st_size, path.stat().st_mtime_ns) != before:
            raise ValueError("raw source file changed during export")
    return signatures


def build_parser() -> argparse.ArgumentParser:
    """Require an explicit source manifest and historical exclusion inventory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--exclude-manifests", type=Path, nargs="+", required=True)
    parser.add_argument("--exclude-all-history-splits", action="store_true")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=500)
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[0, 20, 40, 60, 80, 100, 150]
    )
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate metadata and export a new external evaluation bundle, without GPU."""
    config = load_config(args.config)
    if config.paths.work_root is None or config.paths.data_root is None:
        raise ValueError("configure data_root/work_root")
    data, work = config.paths.data_root.resolve(), config.paths.work_root.resolve()
    project = Path(__file__).resolve().parents[3]
    manifest = (work / args.manifest).resolve()
    histories = [(work / path).resolve() for path in args.exclude_manifests]
    output = (work / args.output_dir).resolve()
    for path in (manifest, *histories, output):
        if (
            path == work
            or not path.is_relative_to(work)
            or path.is_relative_to(data)
            or path.is_relative_to(project)
        ):
            raise ValueError("metadata/output must be external work-root children")
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    excluded, provenance = history_exclusions(
        histories, all_splits=args.exclude_all_history_splits
    )
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    selected, available = select_balanced(
        rows, excluded, lengths=args.lengths, per_group=args.per_group, seed=args.seed
    )
    selected_ids = {row["read_id"] for row in selected}
    summary: dict[str, Any] = {
        "started_utc": datetime.now(UTC).isoformat(),
        "source_manifest": str(manifest),
        "source_manifest_sha256": file_sha256(manifest),
        "historical_manifests": provenance,
        "excluded_splits": "all"
        if args.exclude_all_history_splits
        else ["train", "validation"],
        "excluded_read_ids": len(excluded),
        "selected_exclusion_overlap": len(selected_ids & excluded),
        "available_by_nominal_length": available,
        "requested_per_group": args.per_group,
        "selected_counts": dict(Counter(row["nominal_length_nt"] for row in selected)),
        "selected_reads": len(selected),
        "seed": args.seed,
        "output_dir": str(output),
        "dry_run": args.dry_run,
        "command": sys.argv,
        "limitations": (
            "Construct nominal weak labels, not per-read true lengths; zero_nt is "
            "a separate negative group. Same acquisition as development data; "
            "read-disjointness does not establish cross-run generalization. "
            "Source v4 head/raw-edge eligibility and BAM-order quota selection "
            "remain selection biases. No model/candidate predictions selected reads. "
            "All complete raw read records are copied; biological Poly completeness "
            "and Poly endpoints remain unverified. Final set must not be used to tune."
        ),
    }
    print(json.dumps(summary, sort_keys=True), flush=True)
    if args.dry_run:
        return summary
    output.mkdir(parents=True)
    pod5_path = output / "balanced.pod5"
    signatures = export_full_reads(selected, pod5_path, data_root=data)
    labels = []
    for row in selected:
        labels.append(
            {
                "read_id": row["read_id"],
                "nominal_length_nt": row["nominal_length_nt"],
                "poly_type": row["poly_type"],
                "label_status": row["label_status"],
                "label_source": row["label_source"],
                "orientation": row["orientation"],
                "reference_name": row.get("reference_name", ""),
                "split": "final_test",
                "source_manifest_split": row["split"],
                "source_pod5_path": row["pod5_path"],
                "pod5_path": str(pod5_path),
                "poly_is_complete": "unknown",
                **signatures[row["read_id"]],
            }
        )
    with (output / "labels.tsv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(labels[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(labels)
    (output / "read_ids.txt").write_text("\n".join(sorted(selected_ids)) + "\n")
    for label, repo in (
        ("git_commit", project),
        ("bonito_commit", project / "src/bonito"),
    ):
        summary[label] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    summary.update(
        completed=True,
        finished_utc=datetime.now(UTC).isoformat(),
        pod5_bytes=pod5_path.stat().st_size,
        raw_samples=sum(value["num_samples"] for value in signatures.values()),
        signal_and_metadata_verified=True,
        labels_sha256=file_sha256(output / "labels.tsv"),
        pod5_sha256=file_sha256(pod5_path),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Run metadata selection and bounded POD5 export in the Bonito environment."""
    try:
        run(build_parser().parse_args(argv))
    except (OSError, ValueError, KeyError) as error:
        print(f"Holdout export failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
