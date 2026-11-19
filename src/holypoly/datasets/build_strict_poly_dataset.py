"""NEW strict-poly-candidate-v1 dataset method; old preparation is unchanged.

Filter an existing matched v4 length/reference-head dataset using selected POD5
signals and Dorado moves. Preserve read splits and nominal weak targets. Default
is a read-only resource plan; execute in src/bonito/.venv with --run.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Sequence
from contextlib import closing, nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from holypoly.core.config import load_config
from holypoly.datasets.prepare import HEAD_POLY_VERSION, bam_records, move_bounds
from holypoly.datasets.prepare_sequence import sha256
from holypoly.datasets.signal_metadata import signal_lengths_agree
from holypoly.datasets.strict_poly_candidates import (
    StrictPolyConfig,
    detect_strict_poly_platforms,
    estimate_external_speed,
    restrict_platforms_to_interval,
    select_strict_poly_candidate,
)
from holypoly.datasets.strict_poly_parallel import (
    ImmediateSignalResult,
    ordered_signal_jobs,
)

METHOD = "strict-poly-candidate-v1"
SELECTION_SCOPE = "parent_head_free_chunk_before_candidate_selection"
SPLITS = ("train", "validation", "test")


@dataclass(frozen=True)
class DatasetMethodPolicy:
    """Explicit entry-specific behavior; the historical CLI keeps its defaults."""

    method: str = METHOD
    summary_filename: str = "strict_dataset.json"
    raw_workflow: str = "strict-poly-candidate-v1-from-raw"
    raw_summary_filename: str = "raw_strict_dataset.json"
    dataset_subdirectory: str = "strict"
    entry_filename: str = "build_strict_poly_from_raw.py"
    full_head_free_input: bool = False
    metadata_only_zero: bool = False


STRICT_METHOD_POLICY = DatasetMethodPolicy()


def build_parser() -> argparse.ArgumentParser:
    """Expose bounded signal access, immutable sources and explicit new outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--source-sequence-manifest", type=Path, required=True)
    parser.add_argument("--dorado-bam", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--rules-json",
        type=Path,
        default=Path(__file__).resolve().parents[3]
        / "configs/strict_poly_candidate_v1.json",
    )
    parser.add_argument(
        "--poly-types", nargs="+", choices=("polyA", "polyT"), required=True
    )
    parser.add_argument("--nominal-lengths", nargs="+", type=int, required=True)
    parser.add_argument(
        "--pool-per-group",
        type=int,
        default=30,
        help="Pre-QC reservoir cap per nominal length/direction, split approx 80/10/10",
    )
    parser.add_argument(
        "--target-per-group",
        type=int,
        help="Post-QC target per length/direction (80/10/10); stop signal QC when full",
    )
    parser.add_argument("--max-source-rows", type=int, default=50_000)
    parser.add_argument("--max-bam-records", type=int, default=10_000)
    parser.add_argument("--allow-full-bam-scan", action="store_true")
    parser.add_argument("--max-signal-reads", type=int, default=210)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="CPU detector processes; POD5 reads and output stay ordered in the parent",
    )
    parser.add_argument("--max-signal-samples", type=int, default=120_000)
    parser.add_argument(
        "--pod5-reader-cache-size",
        type=int,
        default=4,
        help="Maximum open POD5 readers reused within this run; 0 disables reuse",
    )
    parser.add_argument("--model-input-samples", type=int, default=12_000)
    parser.add_argument("--crop-context-ms", type=float, default=15.0)
    parser.add_argument("--min-mapping-quality", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--no-progress", action="store_true")
    execution = parser.add_mutually_exclusive_group()
    execution.add_argument("--run", action="store_true")
    execution.add_argument("--dry-run", action="store_true")
    return parser


def _fingerprint(path: Path) -> dict[str, int]:
    """Cheap immutable-input guard; do not hash entire BAM/POD5 files."""
    stat = path.stat()
    return {"bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def _load_rules(path: Path) -> StrictPolyConfig:
    """Load explicit initial thresholds; reject typos instead of ignoring keys."""
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("rules JSON must be an object")
    if "window_ms" in values:
        values["window_ms"] = tuple(values["window_ms"])
    try:
        return StrictPolyConfig(**values)
    except TypeError as error:
        raise ValueError(f"invalid rules JSON: {error}") from error


def _split_quotas(per_group: int) -> dict[str, int]:
    """Allocate per-length/direction quotas, retaining all three read splits."""
    small = max(1, per_group // 10)
    return {"train": per_group - 2 * small, "validation": small, "test": small}


def _target_quotas(
    target_per_group: int | None,
    pool_per_group: int,
    max_signal_reads: int,
    group_count: int,
) -> dict[str, int]:
    """Validate optional final quotas before any BAM or signal scanning."""
    if target_per_group is None:
        return {}
    if not 3 <= target_per_group <= pool_per_group:
        raise ValueError("target-per-group must be between 3 and the pre-QC pool cap")
    if target_per_group * group_count > max_signal_reads:
        raise ValueError("max-signal-reads is smaller than the requested final target")
    return _split_quotas(target_per_group)


def preflight(
    args: argparse.Namespace, *, policy: DatasetMethodPolicy = STRICT_METHOD_POLICY
) -> dict[str, Any]:
    """Stat declared sources only; never scan BAM, POD5 or manifest rows."""
    config = load_config(
        args.config, cli_data_root=args.data_root, cli_work_root=args.work_root
    )
    if config.paths.data_root is None or config.paths.work_root is None:
        raise ValueError("configure data_root and work_root")
    data, work = config.paths.data_root.resolve(), config.paths.work_root.resolve()
    project = Path(__file__).resolve().parents[3]
    if (
        not data.is_dir()
        or not work.is_dir()
        or work.is_relative_to(data)
        or work.is_relative_to(project)
    ):
        raise ValueError("existing separated external data/work roots are required")
    output = (work / args.output_dir).resolve()
    if (
        not output.is_relative_to(work)
        or output == work
        or output.is_relative_to(data)
        or data.is_relative_to(output)
        or output.is_relative_to(project)
        or output.exists()
        or (work / args.output_dir).is_symlink()
    ):
        raise ValueError(
            "choose a new output inside external work_root, outside raw data/repository"
        )
    paths = {}
    for name, root in [
        ("source_manifest", work),
        ("source_sequence_manifest", work),
        ("dorado_bam", data),
    ]:
        path = (root / getattr(args, name)).resolve()
        if (
            not path.is_file()
            or not path.is_relative_to(root)
            or (root == work and path.is_relative_to(data))
        ):
            raise ValueError(f"invalid {name}: {path}")
        paths[name] = str(path)
    if (
        args.pool_per_group < 3
        or args.max_source_rows < 1
        or args.max_bam_records < 0
        or args.max_signal_reads < 1
        or args.max_signal_samples < 12
        or args.workers < 1
        or args.pod5_reader_cache_size < 0
        or args.model_input_samples < 12
        or args.model_input_samples % 12
        or not np.isfinite(args.crop_context_ms)
        or args.crop_context_ms < 0
        or not 1 <= args.min_mapping_quality < 255
        or args.seed < 0
        or not args.nominal_lengths
        or len(set(args.nominal_lengths)) != len(args.nominal_lengths)
        or 0 not in args.nominal_lengths
        or not any(value > 0 for value in args.nominal_lengths)
        or min(args.nominal_lengths) < 0
        or len(set(args.poly_types)) != len(args.poly_types)
    ):
        raise ValueError(
            "invalid settings; include zero and positive nominal constructs"
        )
    if args.max_bam_records == 0 and not args.allow_full_bam_scan:
        raise ValueError("0 requires explicit --allow-full-bam-scan")
    rules = _load_rules(args.rules_json)
    target_quotas = _target_quotas(
        args.target_per_group,
        args.pool_per_group,
        args.max_signal_reads,
        (len(args.nominal_lengths) - int(policy.metadata_only_zero))
        * len(args.poly_types),
    )
    upper = min(
        args.max_signal_reads,
        (len(args.nominal_lengths) - int(policy.metadata_only_zero))
        * len(args.poly_types)
        * args.pool_per_group,
    )
    return {
        "workflow": policy.method,
        "method_policy": asdict(policy),
        "selection_scope": SELECTION_SCOPE,
        "target_split_quotas_per_length_direction": target_quotas,
        "status": "planned",
        "data_root": str(data),
        "work_root": str(work),
        "output_dir": str(output),
        "inputs": paths,
        "input_metadata": {
            key: _fingerprint(Path(value)) for key, value in paths.items()
        },
        "settings": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "candidate_rules": asdict(rules),
        "rules_json_sha256": sha256(args.rules_json),
        "resources": {
            "source_rows_per_table_cap": args.max_source_rows,
            "source_table_hash_bytes": sum(
                Path(paths[key]).stat().st_size
                for key in ("source_manifest", "source_sequence_manifest")
            ),
            "bam_passes": 1,
            "bam_record_cap": args.max_bam_records or "all_explicitly_requested",
            "bam_file_bytes": Path(paths["dorado_bam"]).stat().st_size,
            "selected_signal_reads_upper_bound": upper,
            "target_selected_reads": args.target_per_group
            * len(args.nominal_lengths)
            * len(args.poly_types)
            if args.target_per_group is not None
            else None,
            "decoded_samples_upper_bound": upper * args.max_signal_samples,
            "signal_access": "selected UUIDs only; no POD5 scan or raw-signal copies",
            "open_pod5_readers_cap": args.pod5_reader_cache_size,
            "signal_workers": args.workers,
            "in_flight_signal_cap": args.workers,
            "signal_execution": "serial POD5 reads, ordered CPU process results",
            "native_threads_per_worker": 1 if args.workers > 1 else "inherited",
            "output_estimate_bytes": upper * 2_000_000,
            "runtime_estimate": (
                "metadata/I/O sequential; signal CPU work uses --workers; "
                "full HDD scans and large QC pools can take hours"
            ),
            "memory_estimate": (
                "selected-ID/move maps plus O(workers) one-read detector arrays"
            ),
            "gpu": False,
            "model": False,
        },
        "limitations": [
            "Uncalibrated rules; Poly identity/completeness remain uncertain.",
            "Construct-nominal weak targets/status; not read-level truth.",
            "Candidate coordinates are audit-only; location supervision stays off.",
            "Parent splits inherited; sequence provenance binds the NEW manifest.",
            "Signal/nominal-duration selection bias the retained distribution.",
            "Zero constructs use metadata-only weak negatives; no signal QC is run."
            if policy.metadata_only_zero
            else "Zero constructs with strong platforms are deferred, not relabeled.",
            "Full parent head-free training input; platforms are QC/audit only."
            if policy.full_head_free_input
            else "Training crops a supported platform with bounded context.",
            "Zero head-sequence labels are optional; positives require them."
            if policy.metadata_only_zero
            else "All exported length reads require paired head sequence labels.",
            "Only wholly head-free platforms compete; all platforms stay audited.",
            "No checkpoints, training, original-data writes or threshold auto-tuning.",
        ],
    }


def _sample_sources(
    args: argparse.Namespace, path: Path
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Seeded reservoir by length/direction/EXISTING split; never reassign a read."""
    rng = random.Random(args.seed)
    quota = _split_quotas(args.pool_per_group)
    groups: dict[tuple[int, str, str], list[dict[str, str]]] = defaultdict(list)
    counts: Counter[tuple[int, str, str]] = Counter()
    seen = set()
    scanned = 0
    prefix_limited = False
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {
            "read_id",
            "nominal_length_nt",
            "orientation",
            "split",
            "poly_type",
            "manifest_version",
            "supervision_mode",
            "head_excluded",
        }
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("source manifest missing v4 fields")
        for index, row in enumerate(reader):
            if index >= args.max_source_rows:
                prefix_limited = True
                break
            scanned += 1
            if not row["read_id"] or row["read_id"] in seen:
                raise ValueError(
                    "empty/duplicate read_id across source splits in scanned prefix"
                )
            seen.add(row["read_id"])
            if (
                row["manifest_version"] != HEAD_POLY_VERSION
                or row["supervision_mode"] != "length_only"
                or row["head_excluded"] != "true"
                or row["split"] not in SPLITS
                or row["orientation"] not in {"forward", "reverse"}
            ):
                raise ValueError(
                    "requires head-excluded v4 length_only with valid splits/directions"
                )
            nominal = float(row["nominal_length_nt"])
            if not np.isfinite(nominal) or nominal < 0 or nominal != int(nominal):
                raise ValueError("nominal lengths must be nonnegative integers")
            length = int(nominal)
            kind = "polyA" if row["orientation"] == "forward" else "polyT"
            if length not in args.nominal_lengths or kind not in args.poly_types:
                continue
            if length > 0 and row["poly_type"] != kind:
                raise ValueError("source Poly type disagrees with orientation")
            key = (length, row["orientation"], row["split"])
            counts[key] += 1
            reservoir = groups[key]
            if len(reservoir) < quota[key[2]]:
                reservoir.append(row)
            else:
                slot = rng.randrange(counts[key])
                if slot < quota[key[2]]:
                    reservoir[slot] = row
    # Interleave strata so a signal budget cannot silently consume only zero/train.
    ordered = []
    for index in range(max((len(group) for group in groups.values()), default=0)):
        for key in sorted(groups):
            if index < len(groups[key]):
                ordered.append(groups[key][index])
    return ordered, {
        "source_rows_scanned": scanned,
        "source_prefix_limited": prefix_limited,
        "pool_reads": len(ordered),
        "sampling": "seeded_stratified_reservoir_inherited_splits",
        "source_counts": {str(key): count for key, count in sorted(counts.items())},
    }


def _sequence_sources(
    path: Path, ids: set[str], digest: str, cap: int
) -> dict[str, dict[str, str]]:
    """Require original pair provenance before rebinding any retained sequences."""
    selected = {}
    with path.open(newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream, delimiter="\t")):
            if index >= cap:
                break
            read_id = row["read_id"]
            if read_id not in ids:
                continue
            if read_id in selected or row["source_manifest_sha256"] != digest:
                raise ValueError(
                    "duplicate sequence ID or mismatched parent length-manifest hash"
                )
            selected[read_id] = row
    return selected


class SelectedMoveCollector:
    """Collect checked moves during this run's sequence scan, without another pass.

    Only pool UUIDs are retained. Every record in the declared prefix must be
    observed, including records after all selected IDs were first encountered,
    so duplicate primary alignments cannot be hidden by the optimization.
    """

    def __init__(
        self,
        args: argparse.Namespace,
        path: Path,
        sources: dict[str, dict[str, str]],
        source_manifest_sha256: str,
    ) -> None:
        self.path = path.resolve()
        self.fingerprint = _fingerprint(self.path)
        self.source_manifest_sha256 = source_manifest_sha256
        self.max_records = args.max_bam_records
        self.min_mapping_quality = args.min_mapping_quality
        self.sources = sources
        self.found: dict[str, dict[str, Any]] = {}
        self.errors = dict.fromkeys(sources, "not_seen_in_bam_prefix")
        self.seen: set[str] = set()
        self.scanned = 0

    def observe(self, record: Any) -> None:
        """Apply the same direction, length, move and duplicate checks as a scan."""
        self.scanned += 1
        read_id = record.query_name
        if (
            read_id not in self.sources
            or record.is_secondary
            or record.is_supplementary
        ):
            return
        if read_id in self.seen:
            self.found.pop(read_id, None)
            self.errors[read_id] = "duplicate_primary_in_scanned_bam"
            return
        self.seen.add(read_id)
        source = self.sources[read_id]
        try:
            if (
                record.is_unmapped
                or record.mapping_quality == 255
                or record.mapping_quality < self.min_mapping_quality
            ):
                raise ValueError("unmapped_or_unreliable_mapping_direction")
            if ("reverse" if record.is_reverse else "forward") != source[
                "orientation"
            ] or record.reference_name != source["reference_name"]:
                raise ValueError("source_bam_orientation_or_reference_conflict")
            if (
                record.has_tag("fn")
                and Path(record.get_tag("fn")).name != Path(source["pod5_path"]).name
            ):
                raise ValueError("source_bam_pod5_filename_conflict")
            move_bounds(record, 0, record.query_length)
            moves = np.asarray(record.get_tag("mv"), dtype=np.int64)
            stride, trim, samples = (
                int(moves[0]),
                int(record.get_tag("ts")),
                int(record.get_tag("ns")),
            )
            if not signal_lengths_agree(
                int(source["raw_signal_length"]), samples, stride
            ):
                raise ValueError("source_bam_signal_length_conflict")
            positions = np.repeat(trim + np.arange(len(moves) - 1) * stride, moves[1:])
            positions = np.append(
                positions, min(samples, trim + (len(moves) - 1) * stride)
            )
            self.found[read_id] = {
                "positions": positions,
                "mapping_quality": record.mapping_quality,
            }
            self.errors[read_id] = ""
        except (ValueError, KeyError, IndexError) as error:
            self.errors[read_id] = str(error)

    def result(
        self,
        args: argparse.Namespace,
        path: Path,
        sources: dict[str, dict[str, str]],
        source_manifest_sha256: str,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, str], int]:
        """Bind in-memory moves to the exact BAM, prefix, rules and new source pool."""
        if (
            path.resolve() != self.path
            or _fingerprint(path) != self.fingerprint
            or source_manifest_sha256 != self.source_manifest_sha256
            or args.max_bam_records != self.max_records
            or args.min_mapping_quality != self.min_mapping_quality
            or any(self.sources.get(key) != row for key, row in sources.items())
        ):
            raise ValueError("selected move cache does not match this build")
        return (
            {key: row for key, row in self.found.items() if key in sources},
            {key: self.errors[key] for key in sources},
            self.scanned,
        )


def _alignments(
    args: argparse.Namespace, path: Path, sources: dict[str, dict[str, str]]
) -> tuple[dict[str, dict[str, Any]], dict[str, str], int]:
    """Standalone entry: scan once with the same collector as the fused raw path."""
    collector = SelectedMoveCollector(args, path, sources, "")
    with closing(
        bam_records(
            path,
            args.max_bam_records,
            desc="Strict candidate selected moves",
            progress=not args.no_progress,
        )
    ) as records:
        for record in records:
            collector.observe(record)
    return collector.result(args, path, sources, "")


class Pod5ReaderCache:
    """Bounded LRU of read-only POD5 handles; never cache decoded signals."""

    def __init__(self, max_open_files: int) -> None:
        self.max_open_files = max_open_files
        self.readers: OrderedDict[Path, Any] = OrderedDict()
        self.fingerprints: dict[Path, dict[str, int]] = {}

    def get(self, path: Path) -> Any:
        """Reuse a handle only while its POD5 metadata fingerprint is unchanged."""
        import pod5

        if self.max_open_files <= 0:
            raise ValueError("POD5 reader reuse is disabled")
        if path in self.readers:
            if _fingerprint(path) != self.fingerprints[path]:
                raise ValueError("POD5 changed during reader reuse")
            self.readers.move_to_end(path)
            return self.readers[path]
        if len(self.readers) >= self.max_open_files:
            previous, reader = self.readers.popitem(last=False)
            self.fingerprints.pop(previous)
            reader.close()
        self.readers[path] = pod5.Reader(path)
        self.fingerprints[path] = _fingerprint(path)
        return self.readers[path]

    def close(self) -> None:
        """Release all remaining POD5 handles, including when QC raises or stops."""
        while self.readers:
            _, reader = self.readers.popitem(last=False)
            reader.close()
        self.fingerprints.clear()


def _load_signal(
    source: dict[str, str],
    data: Path,
    max_samples: int,
    *,
    reader_cache: Pod5ReaderCache | None = None,
) -> tuple[Any, float, Path]:
    """Read one calibrated UUID only, checking sample count BEFORE decoding."""
    import pod5

    path = Path(source["pod5_path"]).expanduser().resolve()
    if not path.is_relative_to(data) or not path.is_file():
        raise ValueError("source POD5 must exist inside data_root")
    reader_context = (
        nullcontext(reader_cache.get(path))
        if reader_cache is not None and reader_cache.max_open_files > 0
        else pod5.Reader(path)
    )
    with reader_context as reader:
        reads = list(reader.reads(selection=[source["read_id"]], missing_ok=True))
        if len(reads) != 1:
            raise ValueError("pod5_uuid_missing_or_duplicated")
        read = reads[0]
        if read.num_samples != int(source["raw_signal_length"]):
            raise ValueError("pod5_source_signal_length_conflict")
        if not 0 < read.num_samples <= max_samples:
            raise ValueError("signal_sample_budget_exceeded")
        return read.signal_pa, float(read.run_info.sample_rate), path


def _analyze_strict_signal(
    values: Any,
    fs: float,
    orientation: str,
    positions: Any,
    nominal_length_nt: int,
    allowed_start: int,
    allowed_end: int,
    rules: StrictPolyConfig,
) -> dict[str, Any]:
    """Pure one-read CPU job: decoded raw samples in, candidate diagnostics out.

    This top-level callable can run in a spawned process. It opens no files,
    changes no quotas and uses the same detector for one or many workers.
    """
    detection = detect_strict_poly_platforms(values, fs, orientation, rules)
    speed = estimate_external_speed(positions, detection["regions"], fs, rules)
    scoped = restrict_platforms_to_interval(detection, allowed_start, allowed_end)
    decision = select_strict_poly_candidate(scoped, nominal_length_nt, speed, rules)
    return {
        "detection": detection,
        "speed": speed,
        "decision": decision,
        "fs": fs,
        "allowed_start": allowed_start,
        "allowed_end": allowed_end,
        "eligible_candidate_count": len(scoped["regions"]),
        "supported_candidate_count": sum(
            r["morphology_supported"] for r in scoped["regions"]
        ),
    }


def _training_row(
    source: dict[str, str],
    decision: dict[str, Any],
    fs: float | None,
    args: argparse.Namespace,
    *,
    policy: DatasetMethodPolicy = STRICT_METHOD_POLICY,
) -> dict[str, Any]:
    """Preserve weak targets; retain the full chunk or crop under explicit policy."""
    row: dict[str, Any] = source.copy()
    selected = decision["selected"]
    if selected is not None:
        start, end = selected["start"], selected["end"]
        allowed_start, allowed_end = (
            int(source["chunk_start_sample"]),
            int(source["chunk_end_sample"]),
        )
        if not allowed_start <= start < end <= allowed_end:
            raise ValueError("review_candidate_outside_parent_head_free_chunk")
        if not policy.full_head_free_input:
            if fs is None:
                raise ValueError("sample rate required for candidate cropping")
            margin = round(args.crop_context_ms * fs / 1000)
            row.update(
                chunk_start_sample=max(allowed_start, start - margin),
                chunk_end_sample=min(allowed_end, end + margin),
                candidate_start_sample=start,
                candidate_end_sample=end,
            )
    if (
        int(row["chunk_end_sample"]) - int(row["chunk_start_sample"])
        > args.model_input_samples
    ):
        raise ValueError("review_chunk_exceeds_model_input_no_truncation")
    row.update(
        dataset_method=policy.method,
        selection_reason=policy.method + ":" + decision["status"],
        model_input_samples=args.model_input_samples,
        # Candidate boundaries are diagnostics, never supervised Poly truth.
        poly_start_sample="",
        poly_end_sample="",
        poly_is_complete="",
        strict_candidate_status=decision["status"],
        strict_candidate_start_sample=selected["start"] if selected else "",
        strict_candidate_end_sample=selected["end"] if selected else "",
        strict_boundary_status=selected["boundary_status"]
        if selected
        else "not_applicable",
        strict_duration_ratio=decision.get("duration_ratio", ""),
    )
    if policy.full_head_free_input:
        for name in list(row):
            if name.startswith("strict_"):
                row["platform_qc_" + name.removeprefix("strict_")] = row.pop(name)
        row.update(
            training_interval_policy="full_head_free_raw_edge",
            platform_qc_zero_policy="metadata_only_construct_nominal",
            platform_qc_signal_evaluated="false"
            if decision["status"] == "weak_zero_metadata_only"
            else "true",
        )
    return row


def _write_table(path: Path, rows: list[dict[str, Any]], fallback: list[str]) -> None:
    """Write new TSV metadata only; never overwrite an existing artifact."""
    columns = list(dict.fromkeys(key for row in rows for key in row)) or fallback
    with path.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def run(
    args: argparse.Namespace,
    *,
    move_collector: SelectedMoveCollector | None = None,
    policy: DatasetMethodPolicy = STRICT_METHOD_POLICY,
) -> dict[str, Any]:
    """Build an auditable matched subset; never silently loosen a strict rule."""
    summary = preflight(args, policy=policy)
    print(json.dumps(summary, indent=2), flush=True)
    if not args.run:
        return summary
    output, data = Path(summary["output_dir"]), Path(summary["data_root"])
    output.mkdir(parents=True, exist_ok=False)
    summary.update(
        status="building",
        stage="sampling_metadata",
        started_utc=datetime.now(UTC).isoformat(),
    )
    rules = _load_rules(args.rules_json)

    def save() -> None:
        """Keep failures distinguishable from a complete dataset."""
        (output / policy.summary_filename).write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )

    save()
    try:
        paths = {key: Path(value) for key, value in summary["inputs"].items()}
        source_hash, sequence_hash = (
            sha256(paths["source_manifest"]),
            sha256(paths["source_sequence_manifest"]),
        )
        summary["source_table_sha256"] = {
            "length": source_hash,
            "sequence": sequence_hash,
        }
        sampled, sampling = _sample_sources(args, paths["source_manifest"])
        summary.update(sampling)
        from holypoly.training.data import load_training_manifest

        if sampled:
            pool_path = output / "source_pool.tsv"
            _write_table(pool_path, sampled, [])
            load_training_manifest(pool_path)
        by_id = {row["read_id"]: row for row in sampled}
        sequences = _sequence_sources(
            paths["source_sequence_manifest"],
            set(by_id),
            source_hash,
            args.max_source_rows,
        )
        # No signals are opened for reads without matched sequence evidence.
        requested = {
            key: value
            for key, value in by_id.items()
            if key in sequences
            and not (
                policy.metadata_only_zero and float(value["nominal_length_nt"]) == 0
            )
        }
        summary["stage"] = "dorado_moves"
        save()
        if move_collector is None:
            moves, alignment_errors, scanned = _alignments(
                args, paths["dorado_bam"], requested
            )
            summary["move_source"] = "separate_bam_scan"
        else:
            moves, alignment_errors, scanned = move_collector.result(
                args, paths["dorado_bam"], requested, source_hash
            )
            summary["move_source"] = "this_run_sequence_scan_in_memory"
            summary["resources"]["bam_passes"] = 0
        summary["bam_records_scanned"] = scanned
        accepted: list[dict[str, Any]] = []
        audit: list[dict[str, Any]] = []
        signal_attempts = 0
        pod5_metadata: dict[str, dict[str, int]] = {}
        orientations = [
            "forward" if kind == "polyA" else "reverse" for kind in args.poly_types
        ]
        strata = [
            (length, direction, split)
            for length in args.nominal_lengths
            for direction in orientations
            for split in SPLITS
        ]
        target_quotas = summary["target_split_quotas_per_length_direction"]
        accepted_counts: Counter[tuple[int, str, str]] = Counter()
        target_reached = False
        from tqdm import tqdm

        started = perf_counter()
        status_counts: Counter[str] = Counter()
        export_counts: Counter[str] = Counter()
        summary.update(
            stage="signal_qc",
            progress={
                "processed_reads": 0,
                "total_reads": len(sampled),
                "signal_reads_attempted": 0,
                "eligible_reads": 0,
                "deferred_reads": 0,
                "excluded_reads": 0,
                "not_evaluated_reads": 0,
                "status_counts": {},
                "elapsed_seconds": 0.0,
            },
        )
        save()

        def remaining_quota(source: dict[str, str]) -> int | None:
            """Reserve only still-needed slots while keeping read selection ordered."""
            if not target_quotas:
                return None
            key = (
                int(float(source["nominal_length_nt"])),
                source["orientation"],
                source["split"],
            )
            return max(0, int(target_quotas[key[2]]) - accepted_counts[key])

        def prepare_signal(
            source: dict[str, str],
        ) -> tuple[Any, ...] | ImmediateSignalResult:
            """Read one UUID in the parent; never share POD5 handles with workers."""
            nonlocal signal_attempts
            read_id = source["read_id"]
            if policy.metadata_only_zero and float(source["nominal_length_nt"]) == 0:
                return ImmediateSignalResult(
                    {
                        "metadata_only_zero": True,
                        "detection": {
                            "status": "not_evaluated_metadata_only_zero",
                            "regions": [],
                        },
                        "speed": {
                            "status": "not_evaluated_metadata_only_zero",
                            "window_count": None,
                            "samples_per_nt": None,
                        },
                        "decision": {
                            "status": "weak_zero_metadata_only",
                            "selected": None,
                        },
                        "fs": None,
                        "allowed_start": int(source["chunk_start_sample"]),
                        "allowed_end": int(source["chunk_end_sample"]),
                        "eligible_candidate_count": "",
                        "supported_candidate_count": "",
                    }
                )
            if read_id not in sequences:
                raise ValueError("matched_sequence_missing_in_source_prefix")
            if read_id not in moves:
                raise ValueError(alignment_errors[read_id])
            if signal_attempts >= args.max_signal_reads:
                raise ValueError("signal_read_budget_exhausted")
            signal_attempts += 1
            path = Path(source["pod5_path"]).resolve()
            before = _fingerprint(path)
            if str(path) in pod5_metadata and before != pod5_metadata[str(path)]:
                raise ValueError("POD5 changed between signal reads")
            values, fs, path = _load_signal(
                source, data, args.max_signal_samples, reader_cache=reader_cache
            )
            if before != _fingerprint(path):
                raise ValueError("POD5 changed during signal read")
            pod5_metadata[str(path)] = before
            return (
                values,
                fs,
                source["orientation"],
                moves[read_id]["positions"],
                int(float(source["nominal_length_nt"])),
                int(source["chunk_start_sample"]),
                int(source["chunk_end_sample"]),
                rules,
            )

        with (
            (output / "candidate_audit.jsonl").open("x") as details,
            closing(Pod5ReaderCache(args.pod5_reader_cache_size)) as reader_cache,
            ordered_signal_jobs(
                sampled,
                prepare_signal,
                _analyze_strict_signal,
                remaining_quota,
                args.workers,
            ) as jobs,
            tqdm(
                jobs,
                total=len(sampled),
                desc="Head-free platform QC"
                if policy.full_head_free_input
                else "Strict signal QC",
                unit="read",
                disable=args.no_progress,
            ) as progress,
        ):

            def record_progress(entry: dict[str, Any]) -> None:
                """Persist every visited pool row, including skipped full strata."""
                audit.append(entry)
                details.flush()
                status_counts[entry["status"]] += 1
                export_counts[entry["export_status"]] += 1
                summary["progress"] = {
                    "processed_reads": len(audit),
                    "total_reads": len(sampled),
                    "signal_reads_attempted": signal_attempts,
                    "eligible_reads": len(accepted),
                    "deferred_reads": export_counts["deferred"],
                    "excluded_reads": export_counts["excluded"],
                    "not_evaluated_reads": export_counts["not_evaluated"],
                    "status_counts": dict(status_counts),
                    "elapsed_seconds": round(perf_counter() - started, 2),
                }
                save()
                progress.set_postfix(
                    eligible=len(accepted),
                    deferred=export_counts["deferred"],
                    excluded=export_counts["excluded"],
                    skipped=export_counts["not_evaluated"],
                    refresh=False,
                )

            for source, outcome in progress:
                read_id = source["read_id"]
                entry: dict[str, Any] = {
                    "read_id": read_id,
                    "split": source["split"],
                    "nominal_length_nt": source["nominal_length_nt"],
                    "orientation": source["orientation"],
                    "status": "failure",
                    "failure_reason": "",
                }
                stratum_key = (
                    int(float(source["nominal_length_nt"])),
                    source["orientation"],
                    source["split"],
                )
                if (
                    target_quotas
                    and accepted_counts[stratum_key] >= target_quotas[stratum_key[2]]
                ):
                    entry.update(
                        status="not_evaluated_stratum_quota_met",
                        export_status="not_evaluated",
                    )
                    record_progress(entry)
                    continue
                try:
                    result = outcome.result()
                    detection, speed, decision = (
                        result["detection"],
                        result["speed"],
                        result["decision"],
                    )
                    fs = result["fs"]
                    metadata_zero = result.get("metadata_only_zero", False)
                    allowed_start, allowed_end = (
                        result["allowed_start"],
                        result["allowed_end"],
                    )
                    entry.update(
                        status=decision["status"],
                        candidate_count=""
                        if metadata_zero
                        else len(detection["regions"]),
                        eligible_candidate_count=result["eligible_candidate_count"],
                        supported_candidate_count=result["supported_candidate_count"],
                        selection_start_sample=allowed_start,
                        selection_end_sample=allowed_end,
                        fs="" if metadata_zero else fs,
                        timing_window_count=speed["window_count"],
                        samples_per_nt=speed["samples_per_nt"],
                        duration_ratio=decision.get("duration_ratio", ""),
                    )
                    if policy.full_head_free_input:
                        entry["signal_evaluated"] = "false" if metadata_zero else "true"
                        entry["reference_head_sequence_available"] = (
                            "true" if read_id in sequences else "false"
                        )
                    details.write(
                        json.dumps(
                            {
                                "read_id": read_id,
                                "detection": detection,
                                "selection_scope": {
                                    "method": SELECTION_SCOPE,
                                    "start_sample": allowed_start,
                                    "end_sample": allowed_end,
                                    "eligible_candidate_count": entry[
                                        "eligible_candidate_count"
                                    ],
                                    "supported_candidate_count": entry[
                                        "supported_candidate_count"
                                    ],
                                },
                                "speed": speed,
                                "decision": decision,
                            }
                        )
                        + "\n"
                    )
                    if decision["status"] in {
                        "preferred_suspected_poly",
                        "weak_zero_no_supported_candidate",
                    } or (
                        policy.metadata_only_zero
                        and decision["status"] == "weak_zero_metadata_only"
                    ):
                        candidate_row = _training_row(
                            source, decision, fs, args, policy=policy
                        )
                        if policy.full_head_free_input:
                            candidate_row["reference_head_sequence_available"] = (
                                "true" if read_id in sequences else "false"
                            )
                        accepted.append(candidate_row)
                        accepted_counts[
                            (
                                int(float(source["nominal_length_nt"])),
                                source["orientation"],
                                source["split"],
                            )
                        ] += 1
                        entry.update(
                            export_status="eligible",
                            candidate_start_sample=candidate_row[
                                "platform_qc_candidate_start_sample"
                                if policy.full_head_free_input
                                else "strict_candidate_start_sample"
                            ],
                            candidate_end_sample=candidate_row[
                                "platform_qc_candidate_end_sample"
                                if policy.full_head_free_input
                                else "strict_candidate_end_sample"
                            ],
                        )
                    else:
                        entry["export_status"] = "deferred"
                except (ValueError, KeyError, OSError, IndexError) as error:
                    entry.update(
                        status="failure_or_deferred",
                        failure_reason=str(error),
                        export_status="excluded",
                    )
                record_progress(entry)
                if target_quotas and all(
                    accepted_counts[key] >= target_quotas[key[2]] for key in strata
                ):
                    target_reached = True
                    break
        processed_reads = len(audit)
        # Retain a row for every sampled read, including those never evaluated.
        # These are neither failures nor negatives and never enter training.
        for source in sampled[processed_reads:]:
            audit.append(
                {
                    "read_id": source["read_id"],
                    "split": source["split"],
                    "nominal_length_nt": source["nominal_length_nt"],
                    "orientation": source["orientation"],
                    "status": "not_evaluated_target_quota_met",
                    "failure_reason": "",
                    "export_status": "not_evaluated",
                }
            )
        summary["early_stop"] = {
            "enabled": bool(target_quotas),
            "target_reached": target_reached,
            "stopped_before_pool_end": processed_reads < len(sampled),
            "processed_reads": processed_reads,
            "not_evaluated_reads": export_counts["not_evaluated"]
            + len(sampled)
            - processed_reads,
        }
        if target_reached and not args.no_progress:
            print(
                "Strict signal QC target reached after "
                f"{processed_reads}/{len(sampled)} pool rows; finalizing manifests.",
                file=sys.stderr,
                flush=True,
            )
        summary["stage"] = "balancing_and_verification"
        save()
        summary["signal_reads_attempted"] = signal_attempts
        summary["audit_status_counts"] = dict(Counter(row["status"] for row in audit))
        groups: dict[tuple[int, str, str], list[dict[str, Any]]] = {
            key: [] for key in strata
        }
        for row in accepted:
            groups[
                (int(float(row["nominal_length_nt"])), row["orientation"], row["split"])
            ].append(row)
        missing = [
            str(key)
            for key, group in groups.items()
            if len(group) < target_quotas.get(key[2], 1)
        ]
        summary["missing_strata"] = missing
        summary["eligible_stratum_counts"] = {
            str(key): len(group) for key, group in groups.items()
        }
        selected_rows = []
        if not missing:
            # Balance length/type AND zero orientation independently inside each
            # original split. No read ever migrates into another split.
            for split in SPLITS:
                count = target_quotas.get(split) or min(
                    len(group) for key, group in groups.items() if key[2] == split
                )
                for key, group in sorted(groups.items()):
                    if key[2] == split:
                        selected_rows.extend(group[:count])
        exported = {row["read_id"] for row in selected_rows}
        for entry in audit:
            if entry.get("export_status") == "eligible":
                entry["export_status"] = (
                    "selected"
                    if entry["read_id"] in exported
                    else "not_exported_balance_or_incomplete_strata"
                )
        _write_table(
            output / "read_audit.tsv", audit, ["read_id", "status", "failure_reason"]
        )
        _write_table(
            output / "eligible_before_balance.tsv",
            accepted,
            ["read_id", "dataset_method"],
        )
        summary["eligible_reads"] = len(accepted)
        summary["selected_reads"] = len(selected_rows)
        summary["split_counts"] = dict(Counter(row["split"] for row in selected_rows))
        if selected_rows:
            from holypoly.training.sequence_data import load_sequence_manifest

            length_dir, sequence_dir = output / "length", output / "sequence"
            length_dir.mkdir()
            sequence_dir.mkdir()
            manifest = length_dir / "manifest.tsv"
            _write_table(manifest, selected_rows, [])
            digest = sha256(manifest)
            sequence_rows = []
            for source in selected_rows:
                if source["read_id"] not in sequences:
                    if not (
                        policy.metadata_only_zero
                        and float(source["nominal_length_nt"]) == 0
                    ):
                        raise ValueError(
                            "selected positive read is missing its head sequence"
                        )
                    continue
                row = sequences[source["read_id"]].copy()
                row.update(
                    source_manifest_sha256=digest,
                    dataset_method=policy.method,
                    parent_source_manifest_sha256=source_hash,
                    parent_sequence_manifest_sha256=sequence_hash,
                )
                sequence_rows.append(row)
            sequence_manifest = sequence_dir / "sequence_manifest.tsv"
            _write_table(sequence_manifest, sequence_rows, [])
            records = load_training_manifest(manifest)
            sequence_records = load_sequence_manifest(
                sequence_manifest,
                records,
                source_manifest_sha256=digest,
                data_root=data,
            )
            assert len(records) == len(selected_rows)
            assert len(sequence_records) == len(sequence_rows)
            if not policy.metadata_only_zero:
                assert len(sequence_records) == len(records)
            summary["sequence_reads"] = len(sequence_records)
            summary["sequence_split_counts"] = dict(
                Counter(r.split.value for r in sequence_records)
            )
            summary["selected_reads_without_sequence"] = len(records) - len(
                sequence_records
            )
            summary["artifacts"] = {
                "length_manifest": {"path": str(manifest), "sha256": digest},
                "sequence_manifest": {
                    "path": str(sequence_manifest),
                    "sha256": sha256(sequence_manifest),
                },
            }
        for key, before in summary["input_metadata"].items():
            if before != _fingerprint(paths[key]):
                raise ValueError(f"source changed during build: {key}")
        if source_hash != sha256(paths["source_manifest"]) or sequence_hash != sha256(
            paths["source_sequence_manifest"]
        ):
            raise ValueError("source table content changed during build")
        if summary["rules_json_sha256"] != sha256(args.rules_json):
            raise ValueError("candidate rules changed during build")
        if any(
            before != _fingerprint(Path(path)) for path, before in pod5_metadata.items()
        ):
            raise ValueError("selected POD5 changed during build")
        summary.update(
            status="complete" if selected_rows else "incomplete_no_training_manifest",
            stage="finished",
            finished_utc=datetime.now(UTC).isoformat(),
        )
        project = Path(__file__).resolve().parents[3]
        summary["provenance"] = {
            "python": sys.version,
            "numpy": np.__version__,
            "packages": {name: version(name) for name in ("pod5", "pysam", "torch")},
            "command": sys.argv,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=project, text=True
            ).strip(),
            "bonito_commit": subprocess.check_output(
                ["git", "-C", "src/bonito", "rev-parse", "HEAD"], cwd=project, text=True
            ).strip(),
            "code_sha256": {
                Path(__file__).name: sha256(Path(__file__)),
                "strict_poly_candidates.py": sha256(
                    Path(__file__).with_name("strict_poly_candidates.py")
                ),
                "strict_poly_parallel.py": sha256(
                    Path(__file__).with_name("strict_poly_parallel.py")
                ),
            },
            "git_dirty": subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=project, text=True
            ),
        }
        save()
    except Exception as error:
        summary.update(
            status="failed",
            error=str(error),
            finished_utc=datetime.now(UTC).isoformat(),
        )
        save()
        raise
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Return nonzero for incomplete builds; never start training automatically."""
    try:
        result = run(build_parser().parse_args(argv))
    except (ValueError, OSError, KeyError) as error:
        print(f"Strict Poly dataset failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0 if result["status"] in {"planned", "complete"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
