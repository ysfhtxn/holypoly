"""Extract optional ordinary-window timing features; never read POD5 signals."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pysam

from holypoly.core.config import load_config
from holypoly.datasets.prepare_sequence import sha256
from holypoly.inference.timing_reference import (
    TIMING_REFERENCE_SCHEMA,
    TimingWindowConfig,
    missing_timing_reference,
    timing_reference_from_record,
)
from holypoly.training.data import load_training_manifest


def build_parser() -> argparse.ArgumentParser:
    """Default to a read-only size report; BAM records need explicit --scan-bam."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--bam", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--scan-bam", action="store_true")
    return parser


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Export read-level or per-window audit data without sequences."""
    if not rows:
        path.write_text("")
        return
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Scan only when explicitly requested, preserving missing and failed evidence."""
    config = load_config(args.config)
    if config.paths.data_root is None or config.paths.work_root is None:
        raise ValueError("configure data_root and work_root")
    data, work = config.paths.data_root.resolve(), config.paths.work_root.resolve()
    project = Path(__file__).resolve().parents[3]
    manifest = (work / args.manifest).resolve(strict=True)
    bam = (data / args.bam).resolve(strict=True)
    output = (work / args.output_dir).resolve()
    if not manifest.is_relative_to(work) or manifest.is_relative_to(data):
        raise ValueError("source manifest must be inside work_root, outside raw data")
    if (
        work.is_relative_to(data)
        or work.is_relative_to(project)
        or output == work
        or not output.is_relative_to(work)
        or output.is_relative_to(data)
        or output.exists()
        or args.output_dir.is_symlink()
        or any(path.is_relative_to(output) for path in (manifest, bam))
    ):
        raise ValueError("choose a new external work-root output outside inputs")
    records = load_training_manifest(manifest)
    wanted = {row.read_id: row for row in records}
    if len(wanted) != len(records):
        raise ValueError("timing extraction requires unique read IDs")
    before = bam.stat()
    summary: dict[str, Any] = {
        "schema": TIMING_REFERENCE_SCHEMA,
        "source_manifest_sha256": sha256(manifest),
        "source_manifest": str(manifest),
        "bam": str(bam),
        "bam_bytes": before.st_size,
        "bam_mtime_ns": before.st_mtime_ns,
        "reads": len(records),
        "window_config": asdict(TimingWindowConfig()),
        "scan_bam": args.scan_bam,
        "command": sys.argv,
        "started_utc": datetime.now(UTC).isoformat(),
        "resources": {
            "bam_passes": int(args.scan_bam),
            "pod5_signal_reads": 0,
            "gpu": False,
            "runtime": "one sequential BAM pass; minutes for multi-GiB inputs",
            "output_estimate_bytes": len(records) * 5000,
        },
        "limitations": (
            "Candidate-independent QC windows; provisional move timing, not "
            "verified non-Poly regions or read-level truth. Extraction differs "
            "from the older candidate-anchored 10.4% diagnostic."
        ),
    }
    print(json.dumps(summary, indent=2), flush=True)
    if not args.scan_bam:
        return summary
    output.mkdir(parents=True)
    (output / "plan.json").write_text(json.dumps(summary, indent=2) + "\n")
    extracted: dict[str, dict[str, Any]] = {}
    windows: list[dict[str, Any]] = []
    scanned, started, last_progress = 0, time.monotonic(), time.monotonic()
    with pysam.AlignmentFile(str(bam), "rb", check_sq=False) as source:
        for record in source.fetch(until_eof=True):
            scanned += 1
            if time.monotonic() - last_progress >= 15:
                print(
                    f"BAM records scanned: {scanned:,}; "
                    f"selected: {len(extracted):,}/{len(wanted):,}",
                    flush=True,
                )
                last_progress = time.monotonic()
            if (
                record.is_secondary
                or record.is_supplementary
                or record.query_name not in wanted
            ):
                continue
            read_id = str(record.query_name)
            if read_id in extracted:
                raise ValueError(
                    "duplicate selected primary BAM read; refusing ambiguous evidence"
                )
            if (
                record.has_tag("ns")
                and int(record.get_tag("ns")) != wanted[read_id].raw_signal_length
            ):
                values = missing_timing_reference("raw_signal_length_mismatch")
                details: list[dict[str, Any]] = []
            else:
                values, details = timing_reference_from_record(record)
            extracted[read_id] = values
            windows.extend({"read_id": read_id, **row} for row in details)
    if (before.st_size, before.st_mtime_ns) != (
        bam.stat().st_size,
        bam.stat().st_mtime_ns,
    ):
        raise ValueError("source BAM changed during extraction")
    rows = [
        {
            "read_id": row.read_id,
            "split": row.split.value,
            "raw_signal_length": row.raw_signal_length,
            **extracted.get(row.read_id, missing_timing_reference("read_not_found")),
        }
        for row in records
    ]
    table = output / "timing_reference.tsv"
    _write_table(table, rows)
    _write_table(output / "window_audit.tsv", windows)
    summary.update(
        timing_table_sha256=sha256(table),
        records_scanned=scanned,
        elapsed_seconds=time.monotonic() - started,
        reference_status_counts=dict(
            Counter(str(row["reference_status"]) for row in rows)
        ),
        coverage_by_split={
            split: {
                "reads": sum(row["split"] == split for row in rows),
                "available": sum(
                    row["split"] == split and row["reference_available"] == 1
                    for row in rows
                ),
            }
            for split in ("train", "validation", "test")
        },
        finished_utc=datetime.now(UTC).isoformat(),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    """Expose the explicit offline preparation command."""
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
