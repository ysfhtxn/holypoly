"""Make reference-supported, non-Poly head sequence labels for selected reads.

No Poly sequence is synthesized. Dorado moves remain provisional boundaries.
Only the supplied length manifest's read IDs are eligible; splits are inherited.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from collections.abc import Sequence
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from holypoly.core.config import load_config
from holypoly.datasets.prepare import bam_records, load_heads, move_bounds, write_table

VERSION = "reference-head-sequence-v1"
LABEL_SOURCE = "reference_head_exact_match"
BOUNDARY_SOURCE = "dorado_moves_provisional"
FIELDS = (
    "manifest_version",
    "read_id",
    "pod5_path",
    "split",
    "orientation",
    "raw_signal_length",
    "chunk_start_sample",
    "chunk_end_sample",
    "sequence_target",
    "reference_name",
    "reference_start_base",
    "reference_end_base",
    "query_start_base",
    "query_end_base",
    "mapping_quality",
    "minimum_base_quality",
    "label_source",
    "boundary_source",
    "source_manifest_sha256",
    "reference_sha256",
)


def sha256(path: Path) -> str:
    """Hash small manifest/reference inputs; never hash whole BAM or POD5 files."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_head_window(
    record: Any,
    head: str,
    *,
    target_bases: int = 96,
    margin_bases: int = 32,
    min_base_quality: int = 15,
    min_mapping_quality: int = 20,
    max_homopolymer: int = 4,
) -> dict[str, Any]:
    """Select one unique, exact, indel-free head window in acquisition order.

    Reference/query coordinates are BAM SEQ coordinates, 0-based half-open.
    Raw coordinates use untrimmed acquisition samples, also half-open.
    Reverse alignments produce reverse-complement sequence targets.
    """
    if record.is_unmapped or record.mapping_quality < min_mapping_quality:
        raise ValueError("unmapped_or_low_mapping_quality")
    sequence, qualities = record.query_sequence, record.query_qualities
    if not sequence or qualities is None:
        raise ValueError("missing_sequence_or_base_qualities")
    if len(head) < target_bases + 2 * margin_bases:
        raise ValueError("head_too_short_for_interior_window")
    # Including insertions/deletions in this list makes them break exact runs.
    pairs = record.get_aligned_pairs()
    run: list[tuple[int, int]] = []
    candidates: list[list[tuple[int, int]]] = []
    for query, reference in pairs:
        valid = (
            query is not None
            and reference is not None
            and margin_bases <= reference < len(head) - margin_bases
            and sequence[query] == head[reference]
            and qualities[query] >= min_base_quality
        )
        if not valid:
            run = []
            continue
        if run and (query != run[-1][0] + 1 or reference != run[-1][1] + 1):
            run = []
        run.append((query, reference))
        if len(run) >= target_bases:
            candidates.append(run[-target_bases:])
    # Prefer a window near the head end, always inside the configured margins.
    for window in reversed(candidates):
        query_start, reference_start = window[0]
        query_end, reference_end = window[-1][0] + 1, window[-1][1] + 1
        target = head[reference_start:reference_end]
        if any(base * (max_homopolymer + 1) in target for base in "ACGT"):
            continue
        if head.find(target) != head.rfind(target) or sequence.find(
            target
        ) != sequence.rfind(target):
            continue
        start, end = move_bounds(record, query_start, query_end)
        # Do not split several bases emitted at the same move boundary.
        before_base = query_end if record.is_reverse else query_start - 1
        last_base = query_start if record.is_reverse else query_end - 1
        if (
            0 <= before_base < len(sequence)
            and move_bounds(record, before_base, before_base + 1)[0] == start
        ):
            continue
        if move_bounds(record, last_base, last_base + 1)[0] == end:
            continue
        if start >= end:
            continue
        if record.is_reverse:
            target = target.translate(str.maketrans("ACGT", "TGCA"))[::-1]
        return {
            "chunk_start_sample": start,
            "chunk_end_sample": end,
            "sequence_target": target,
            "reference_start_base": reference_start,
            "reference_end_base": reference_end,
            "query_start_base": query_start,
            "query_end_base": query_end,
            "mapping_quality": record.mapping_quality,
            "minimum_base_quality": min(qualities[query_start:query_end]),
        }
    raise ValueError("no_unique_exact_high_quality_interior_window")


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Audit selected IDs in a bounded BAM scan; write only new external metadata."""
    config = load_config(args.config)
    if config.paths.data_root is None or config.paths.work_root is None:
        raise ValueError("Configure data_root/work_root")
    data, work = (
        p.expanduser().resolve()
        for p in (config.paths.data_root, config.paths.work_root)
    )
    project = Path(__file__).resolve().parents[3]
    output = (work / args.output_dir).resolve()
    if (
        not data.is_dir()
        or not work.is_dir()
        or work.is_relative_to(data)
        or work.is_relative_to(project)
        or not output.is_relative_to(work)
        or output == work
        or output.is_relative_to(data)
        or output.is_relative_to(project)
    ):
        raise ValueError(
            "output must be inside external work_root, outside data_root/project"
        )
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"output already exists: {output}")
    manifest = (work / args.manifest).resolve()
    reference, bam = ((data / p).resolve() for p in (args.reference, args.dorado_bam))
    if not manifest.is_relative_to(work) or not manifest.is_file():
        raise ValueError("source manifest must exist inside work_root")
    for path in (reference, bam):
        if not path.is_relative_to(data) or not path.is_file():
            raise ValueError("BAM/reference must exist inside data_root")
    if (
        args.max_records < 0
        or args.target_bases < 8
        or args.margin_bases < 1
        or args.min_base_quality < 0
        or args.min_mapping_quality < 0
        or args.max_homopolymer < 1
        or args.max_signal_samples < 12
    ):
        raise ValueError("invalid scan/window/quality settings")
    heads = load_heads(reference)
    with manifest.open(newline="") as stream:
        sources = list(csv.DictReader(stream, delimiter="\t"))
    if not sources:
        raise ValueError("empty source manifest")
    by_id = {}
    for row in sources:
        read_id = row["read_id"]
        if not read_id or read_id in by_id:
            raise ValueError(
                "source manifest must have unique nonempty read IDs across splits"
            )
        if (
            row["manifest_version"] != "construct-nominal-head-poly-v4"
            or row["supervision_mode"] != "length_only"
            or row["split"] not in ("train", "validation", "test")
            or row["orientation"] not in ("forward", "reverse")
            or row["reference_name"] not in heads
        ):
            raise ValueError(
                "requires a v4 length_only manifest with known head references"
            )
        pod5 = Path(row["pod5_path"]).resolve()
        if not pod5.is_relative_to(data) or not pod5.is_file():
            raise ValueError("source POD5 paths must exist inside data_root")
        by_id[read_id] = row
    summary: dict[str, Any] = {
        "manifest_version": VERSION,
        "started_utc": datetime.now(UTC).isoformat(),
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256(manifest),
        "reference": str(reference),
        "reference_sha256": sha256(reference),
        "bam": str(bam),
        "bam_bytes": bam.stat().st_size,
        "bam_mtime_ns": bam.stat().st_mtime_ns,
        "requested_reads": len(sources),
        "settings": {
            key: getattr(args, key)
            for key in (
                "max_records",
                "target_bases",
                "margin_bases",
                "min_base_quality",
                "min_mapping_quality",
                "max_homopolymer",
                "max_signal_samples",
            )
        },
        "output_dir": str(output),
        "dry_run": args.dry_run,
        "limitations": [
            "FASTA records are assumed to be construct heads, as in v4 preparation.",
            "Exact reference/basecall agreement is selected evidence, "
            "not independent read truth.",
            "Dorado move boundaries and source POD5 metadata remain provisional.",
            "Only head interiors are labeled; no nominal Poly repeats "
            "or inferred tails.",
            "Read uniqueness is checked within the scanned BAM prefix only.",
            "Original read splits are inherited; no new reads or split reassignment.",
        ],
    }
    if args.dry_run:
        return summary
    print(
        f"CPU metadata only: <= {args.max_records or 'all'} BAM records; "
        f"file {bam.stat().st_size / 1024**3:.2f} GiB. "
        f"Only {len(sources)} selected IDs; no POD5 signals or GPU. "
        "Bounded scans typically take seconds to minutes; "
        "full scans may take tens of minutes. "
        "Output is a few MB for a 1,400-read manifest.",
        file=sys.stderr,
        flush=True,
    )
    accepted: dict[str, dict[str, Any]] = {}
    audit = dict.fromkeys(by_id, "not_seen_in_scanned_prefix")
    seen: set[str] = set()
    scanned = 0
    with closing(
        bam_records(
            bam,
            args.max_records,
            desc="Head sequence audit",
            progress=not args.no_progress,
        )
    ) as records:
        for record in records:
            scanned += 1
            if (
                record.is_secondary
                or record.is_supplementary
                or record.query_name not in by_id
            ):
                continue
            read_id = record.query_name
            if read_id in seen:
                audit[read_id] = "duplicate_primary_in_scanned_prefix"
                accepted.pop(read_id, None)
                continue
            seen.add(read_id)
            source = by_id[read_id]
            try:
                if record.reference_name != source["reference_name"]:
                    raise ValueError("source_reference_disagreement")
                orientation = "reverse" if record.is_reverse else "forward"
                if orientation != source["orientation"]:
                    raise ValueError("source_orientation_disagreement")
                window = select_head_window(
                    record,
                    heads[source["reference_name"]],
                    target_bases=args.target_bases,
                    margin_bases=args.margin_bases,
                    min_base_quality=args.min_base_quality,
                    min_mapping_quality=args.min_mapping_quality,
                    max_homopolymer=args.max_homopolymer,
                )
                raw_length = int(source["raw_signal_length"])
                stride = int(record.get_tag("mv")[0])
                if not 0 <= raw_length - int(record.get_tag("ns")) < stride:
                    raise ValueError("source_raw_length_disagreement")
                start, end = window["chunk_start_sample"], window["chunk_end_sample"]
                if (
                    not 0 <= start < end <= raw_length
                    or end - start > args.max_signal_samples
                ):
                    raise ValueError("sequence_signal_length_out_of_range")
                if (
                    orientation == "forward" and end > int(source["chunk_start_sample"])
                ) or (
                    orientation == "reverse" and start < int(source["chunk_end_sample"])
                ):
                    raise ValueError("head_window_overlaps_poly_candidate")
                accepted[read_id] = {
                    "manifest_version": VERSION,
                    "read_id": read_id,
                    "pod5_path": source["pod5_path"],
                    "split": source["split"],
                    "orientation": orientation,
                    "raw_signal_length": raw_length,
                    "reference_name": source["reference_name"],
                    **window,
                    "label_source": LABEL_SOURCE,
                    "boundary_source": BOUNDARY_SOURCE,
                    "source_manifest_sha256": summary["source_manifest_sha256"],
                    "reference_sha256": summary["reference_sha256"],
                }
                audit[read_id] = "accepted"
            except (ValueError, KeyError, IndexError) as error:
                audit[read_id] = str(error)
    rows = [accepted[key] for key in sorted(accepted)]
    summary.update(
        scanned_records=scanned,
        accepted_reads=len(rows),
        split_counts=dict(Counter(r["split"] for r in rows)),
        audit_counts=dict(Counter(audit.values())),
        finished_utc=datetime.now(UTC).isoformat(),
    )
    output.mkdir(parents=True, exist_ok=False)
    write_table(output / "sequence_manifest.tsv", rows, FIELDS)
    write_table(
        output / "audit.tsv",
        [
            {
                "read_id": key,
                "split": by_id[key]["split"],
                "status": "accepted" if audit[key] == "accepted" else "rejected",
                "failure_reason": "" if audit[key] == "accepted" else audit[key],
            }
            for key in sorted(audit)
        ],
        ("read_id", "split", "status", "failure_reason"),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Expose explicit existing-manifest, scan and head-window settings."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dorado-bam", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("fixed_len_ref.fa"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--max-records",
        type=int,
        default=100_000,
        help="0 explicitly scans all BAM records",
    )
    parser.add_argument("--target-bases", type=int, default=96)
    parser.add_argument("--margin-bases", type=int, default=32)
    parser.add_argument("--min-base-quality", type=int, default=15)
    parser.add_argument("--min-mapping-quality", type=int, default=20)
    parser.add_argument("--max-homopolymer", type=int, default=4)
    parser.add_argument("--max-signal-samples", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Prepare metadata without installing dependencies or starting training."""
    try:
        print(json.dumps(prepare(build_parser().parse_args(argv)), sort_keys=True))
        return 0
    except (OSError, ValueError, KeyError) as error:
        print(f"Sequence preparation failed: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
