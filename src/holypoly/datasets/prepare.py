"""Build balanced weak-label manifests from Bonito, Dorado moves and POD5.

Run in the Bonito environment: python -m holypoly.datasets.prepare --help.
No signal copies or model weights are needed: training reads the POD5 chunks.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Collection, Generator, Mapping, Sequence
from contextlib import ExitStack, closing
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any
from uuid import UUID

from holypoly.core.config import load_config
from holypoly.datasets.length_groups import parse_nominal_length_nt

VERSION = "construct-nominal-head-excluded-v3"
HEAD_POLY_VERSION = "construct-nominal-head-poly-v4"
FIELDS = (  # noqa: SIM905 -- Compact declaration of a wide scientific table.
    "manifest_version read_id pod5_path bam_path pod5_file_id split orientation "
    "raw_signal_length chunk_start_sample chunk_end_sample poly_start_sample "
    "poly_end_sample poly_type training_length_nt true_length_nt nominal_length_nt "
    "observed_sequence_length_nt label_status poly_is_complete failure_reason "
    "label_source flank_source flank_status selection_reason reference_name "
    "head_end_base flank_start_base move_stride trim_start_samples "
    "supervision_mode candidate_start_sample candidate_end_sample "
    "dorado_signal_samples model_input_samples head_excluded head_boundary_sample"
).split()


def read_uuid(name: str) -> str:
    """Restore UUID from Bonito's UUID:chunk_index:chunk_count read names."""
    parts = name.split(":")
    if len(parts) != 1 and (
        len(parts) != 3 or not all(part.isdigit() for part in parts[1:])
    ):
        raise ValueError("unexpected Bonito read name")
    return str(UUID(parts[0]))


def load_heads(path: Path) -> dict[str, str]:
    """Load construct heads; the complete FASTA sequence precedes Poly."""
    heads: dict[str, str] = {}
    name = ""
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            if name in heads:
                raise ValueError("duplicate FASTA reference")
            heads[name] = ""
        elif line.strip():
            if not name:
                raise ValueError("FASTA sequence before header")
            heads[name] += line.strip().upper()
    if not heads or any(not seq or set(seq) - set("ACGT") for seq in heads.values()):
        raise ValueError("heads must be nonempty A/C/G/T sequences")
    if any(parse_nominal_length_nt(name) is None for name in heads):
        raise ValueError("each head name must contain exactly one <number>A token")
    return heads


def bam_records(
    path: Path, limit: int, *, desc: str = "BAM", progress: bool = True
) -> Generator[Any, None, None]:
    """Stream records with progress; full scans use compressed byte offsets.

    Limited scans show the record cap, not a pre-counted file total. BGZF byte
    offsets approximate full-scan progress without an extra scan or BAM index.
    """
    import pysam
    from tqdm import tqdm

    total = limit or path.stat().st_size
    count = 0
    with (
        pysam.AlignmentFile(str(path), "rb") as bam,
        tqdm(
            total=total,
            desc=desc,
            unit="record" if limit else "B",
            unit_scale=not limit,
            mininterval=1.0,
            dynamic_ncols=True,
            disable=not progress,
            file=sys.stderr,
        ) as bar,
    ):
        records = bam.fetch(until_eof=True)
        for count, record in enumerate(
            records if limit == 0 else islice(records, limit), start=1
        ):
            if not progress:
                yield record
                continue
            if limit:
                bar.update(1)
            else:
                # BAM tell() is a BGZF virtual offset: high bits are file bytes.
                offset = min(bam.tell() >> 16, total)
                bar.update(max(0, offset - bar.n))
                if count == 1 or count % 1000 == 0:
                    bar.set_postfix(records=count, refresh=False)
            yield record
        if not limit:
            bar.set_postfix(records=count, refresh=False)
            bar.update(total - bar.n)


def construct_assignments(
    path: Path, heads: dict[str, str], limit: int, *, progress: bool = True
) -> tuple[dict[str, str], set[str], Counter[str]]:
    """Link UUID to construct, excluding inconsistent chunk assignments."""
    assigned: dict[str, str] = {}
    conflicts: set[str] = set()
    counts: Counter[str] = Counter()
    for record in bam_records(
        path, limit, desc="1/3 Bonito constructs", progress=progress
    ):
        counts["bonito_records"] += 1
        if record.is_secondary or record.is_supplementary:
            continue
        read_id = read_uuid(record.query_name)
        reference = record.reference_name or ""
        if read_id in assigned and assigned[read_id] != reference:
            conflicts.add(read_id)
        assigned[read_id] = reference
    return assigned, conflicts, counts


def head_boundary(record: Any, head: str, anchor_bases: int, error_rate: float) -> int:
    """Find head end in BAM SEQ coordinates, checking its terminal alignment.

    BAM SEQ is reference-oriented even for reverse alignments. Coordinates here
    are 0-based bases; reverse-to-signal conversion happens only in move_bounds.
    """
    if record.is_unmapped or record.reference_end != len(head):
        raise ValueError("head_end_not_covered")
    sequence = record.query_sequence
    if not sequence or len(head) < anchor_bases:
        raise ValueError("missing_sequence_or_short_head")
    start = len(head) - anchor_bases
    pairs = [
        (query, reference)
        for query, reference in record.get_aligned_pairs()
        if reference is not None and reference >= start
    ]
    matches = sum(
        query is not None and sequence[query] == head[reference]
        for query, reference in pairs
    )
    if matches < anchor_bases * (1 - error_rate):
        raise ValueError("head_anchor_low_identity")
    endpoint = [query for query, ref in pairs if ref == len(head) - 1]
    if len(endpoint) != 1 or endpoint[0] is None:
        raise ValueError("head_terminal_base_deleted")
    return int(endpoint[0]) + 1


def infer_flank(candidates: Sequence[str], min_support: int, error_rate: float) -> str:
    """Pick a supported observed 0A suffix, allowing substitution errors only."""
    counts = Counter(candidates)
    if not counts:
        raise ValueError("no_usable_0A_flank_candidates")
    scored = []
    for candidate in counts:
        support = sum(
            count
            for other, count in counts.items()
            if sum(a != b for a, b in zip(candidate, other, strict=True))
            <= math.floor(len(candidate) * error_rate)
        )
        scored.append((support, counts[candidate], candidate))
    scored.sort(reverse=True)
    if scored[0][0] < min_support:
        raise ValueError("insufficient_0A_flank_support")
    if len(scored) > 1 and scored[0][:2] == scored[1][:2]:
        raise ValueError("ambiguous_0A_flank_consensus")
    return scored[0][2]


def poly_like(sequence: str, min_fraction: float) -> bool:
    """Require A-rich sequence with no run of more than two non-A bases."""
    if not sequence or sequence.count("A") / len(sequence) < min_fraction:
        return False
    interrupted = 0
    for base in sequence:
        interrupted = 0 if base == "A" else interrupted + 1
        if interrupted > 2:
            return False
    return True


def flank_boundary(
    sequence: str,
    head_end: int,
    flank: str,
    nominal_length: int,
    error_rate: float,
    min_fraction: float,
) -> int:
    """Find a unique downstream anchor with a plausible intervening Poly."""
    hits = []
    for position in range(head_end, len(sequence) - len(flank) + 1):
        segment = sequence[position : position + len(flank)]
        if sum(a != b for a, b in zip(segment, flank, strict=True)) > math.floor(
            len(flank) * error_rate
        ):
            continue
        gap = sequence[head_end:position]
        if (nominal_length == 0 and not gap) or (
            nominal_length > 0 and poly_like(gap, min_fraction)
        ):
            hits.append(position)
    if len(hits) != 1:
        raise ValueError(
            "missing_flank_or_invalid_poly" if not hits else "ambiguous_flank"
        )
    return hits[0]


def move_bounds(record: Any, start_base: int, end_base: int) -> tuple[int, int]:
    """Map a base interval to untrimmed raw samples [start, end).

    mv is int8: first element is stride; remaining nonnegative counts are moves
    in acquisition order. A reverse BAM SEQ interval must be reversed first.
    Multiple bases in one move block share a provisional sample coordinate.
    """
    if record.has_tag("pi") or (record.has_tag("dx") and record.get_tag("dx") != 0):
        raise ValueError("split_or_duplex_read_not_supported")
    if not all(record.has_tag(tag) for tag in ("mv", "ts", "ns")):
        raise ValueError("missing_move_tags")
    moves = list(record.get_tag("mv"))
    stride = int(moves[0]) if moves else 0
    trim = int(record.get_tag("ts"))
    samples = int(record.get_tag("ns"))
    if stride <= 0 or trim < 0 or any(move < 0 for move in moves[1:]):
        raise ValueError("invalid_move_table")
    positions = [
        trim + block * stride
        for block, move in enumerate(moves[1:])
        for _ in range(move)
    ]
    if len(positions) != record.query_length or not positions:
        raise ValueError("move_sequence_length_mismatch")
    if (
        positions[-1] >= samples
        or trim + (len(moves) - 1) * stride > samples + stride - 1
    ):
        raise ValueError("moves_outside_raw_signal")
    if record.is_reverse:
        start_base, end_base = len(positions) - end_base, len(positions) - start_base
    if not 0 <= start_base <= end_base <= len(positions):
        raise ValueError("base_bounds_outside_moves")
    positions.append(min(samples, trim + (len(moves) - 1) * stride))
    return positions[start_base], positions[end_base]


def length_only_row(row: dict[str, Any], reason: str) -> None:
    """Remove unresolved boundaries; retain only the nominal length target."""
    row.update(
        supervision_mode="length_only",
        poly_start_sample="",
        poly_end_sample="",
        poly_is_complete="",
        observed_sequence_length_nt="",
        flank_start_base="",
        flank_status=f"unresolved:{reason}",
        flank_source="not_used_for_length_target",
    )


def head_free_bounds(
    record: Any, head_end_base: int, model_input_samples: int
) -> tuple[int, int, int]:
    """Keep the called post-head interval; never borrow head signal for padding.

    Return absolute raw [start, end) and the head-side boundary in samples.
    Reject a move block shared by the last head and first downstream base;
    removing that block could otherwise silently truncate a complete Poly.
    Head exclusion is relative to the accepted Dorado alignment/move table.
    """
    if not 0 < head_end_base < record.query_length:
        raise ValueError("no_post_head_signal")
    start, end = move_bounds(record, head_end_base, record.query_length)
    adjacent_start, adjacent_end = move_bounds(
        record,
        head_end_base if record.is_reverse else head_end_base - 1,
        head_end_base + 1 if record.is_reverse else head_end_base,
    )
    if adjacent_start == adjacent_end:
        raise ValueError("head_downstream_share_move_block")
    if not 0 < end - start <= model_input_samples:
        raise ValueError("head_free_signal_empty_or_exceeds_chunk")
    return start, end, end if record.is_reverse else start


def head_poly_bounds(
    record: Any,
    head_end_base: int,
    raw_length: int,
    model_input_samples: int,
    min_candidate_samples: int = 12,
) -> tuple[int, int, int]:
    """Keep a head-excluded raw-edge candidate, not a labeled Poly interval.

    Coordinates are absolute raw samples [start, end). The raw edge can include
    adapter/noise; completeness and Poly endpoint remain unknown. A head-only
    read is usable only if real signal remains on the candidate side.
    """
    boundary, _ = move_bounds(record, head_end_base, head_end_base)
    if head_end_base < record.query_length:
        adjacent_start, adjacent_end = move_bounds(
            record,
            head_end_base if record.is_reverse else head_end_base - 1,
            head_end_base + 1 if record.is_reverse else head_end_base,
        )
        if adjacent_start == adjacent_end:
            raise ValueError("head_downstream_share_move_block")
    start, end = (0, boundary) if record.is_reverse else (boundary, raw_length)
    if not 0 <= start < end <= raw_length:
        raise ValueError("no_real_post_head_signal")
    if end - start < min_candidate_samples:
        raise ValueError("post_head_signal_below_min_samples")
    if end - start > model_input_samples:
        raise ValueError("head_free_signal_empty_or_exceeds_chunk")
    return start, end, boundary


def _sampling_group(
    row: Mapping[str, Any], *, split_zero_orientation: bool = False
) -> tuple[int, str]:
    """Return a sampling stratum without changing biological label fields."""
    length = int(row["nominal_length_nt"])
    if split_zero_orientation and length == 0:
        orientation = row["orientation"]
        if orientation not in {"forward", "reverse"}:
            raise ValueError("zero_nt requires forward/reverse orientation")
        return 0, f"none:{orientation}"
    return length, str(row["poly_type"])


def balance_and_split(
    rows: Sequence[dict[str, Any]],
    per_group: int,
    rng: random.Random,
    *,
    poly_types: Sequence[str] = ("polyA", "polyT"),
    balance_to_smallest: bool = True,
    strict_groups: Collection[tuple[int, str]] | None = None,
) -> list[dict[str, Any]]:
    """Balance selected positive types, retain zero_nt, and split unique UUIDs.

    Excluded positive types never limit the common per-stratum sample count.
    Zero-length reads retain their original orientations and are capped at that
    count, without limiting the positive strata when negatives are scarce.
    With ``balance_to_smallest=False`` (v4), cap each group independently and
    preserve shortages for the summary instead of shrinking every other group.
    Explicit ``strict_groups`` instead requires all listed strata, including
    zero_nt orientations, and downsamples all of them to a common count.
    """
    if not poly_types or set(poly_types) - {"polyA", "polyT"}:
        raise ValueError("poly_types must contain polyA and/or polyT")
    groups: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["nominal_length_nt"] > 0 and row["poly_type"] not in poly_types:
            continue
        groups[
            _sampling_group(row, split_zero_orientation=strict_groups is not None)
        ].append(row)
    if strict_groups is not None:
        if not strict_groups:
            raise ValueError("strict balance requires nonempty target strata")
        insufficient = {
            f"{length}:{kind}": len(groups[(length, kind)])
            for length, kind in sorted(strict_groups)
            if len(groups[(length, kind)]) < 3
        }
        if insufficient:
            raise ValueError(
                f"strict balance requires at least 3 reads in every stratum: "
                f"{insufficient}"
            )
        groups = {key: groups[key] for key in sorted(strict_groups)}
        if len({row["read_id"] for group in groups.values() for row in group}) != sum(
            len(group) for group in groups.values()
        ):
            raise ValueError("strict balance requires unique read_id values")
    positive_groups = [group for key, group in groups.items() if key[0] > 0]
    if not positive_groups:
        raise ValueError(
            "no_positive_reads_with_both_anchors"
            if balance_to_smallest
            else "no_positive_reads_with_valid_head"
        )
    supported_groups = [group for group in positive_groups if len(group) >= 3]
    if not supported_groups:
        raise ValueError("need_at_least_3_reads_per_present_positive_stratum")
    if strict_groups is not None:
        count = min(per_group, *(len(group) for group in groups.values()))
    else:
        count = (
            min(per_group, *(len(group) for group in supported_groups))
            if balance_to_smallest
            else per_group
        )
    selected = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: row["read_id"])
        rng.shuffle(group)
        group = group[:count]
        if len(group) < 3:
            continue
        validation_count = max(1, len(group) // 10)
        test_count = max(1, len(group) // 10)
        for index, row in enumerate(group):
            row["split"] = (
                "validation"
                if index < validation_count
                else "test"
                if index < validation_count + test_count
                else "train"
            )
            row["selection_reason"] = (
                "strict_balanced_length_type_and_zero_orientation"
                if strict_groups is not None
                else "balanced_nominal_length_and_poly_type"
                if balance_to_smallest
                else "capped_nominal_length_and_poly_type"
            )
            selected.append(row)
    return selected


def _metadata_quotas_met(
    group_counts: Mapping[tuple[int, str], int],
    target_groups: Collection[tuple[int, str]],
    per_group: int,
    cross_lengths: Mapping[int, int],
    min_flank_reads: int,
    *,
    require_flank: bool = True,
) -> bool:
    """Require all quotas and, for legacy v3 only, provisional flank support."""
    return (
        bool(target_groups)
        and all(group_counts.get(group, 0) >= per_group for group in target_groups)
        and (
            not require_flank
            or len(cross_lengths) >= 2
            and sum(cross_lengths.values()) >= min_flank_reads
        )
    )


def write_table(
    path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]
) -> None:
    """Write metadata only; never write sequencing inputs or signal arrays."""
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_audit(output: Path, assigned: dict[str, str], audit: dict[str, str]) -> None:
    """Keep UUID-to-length links and rejection reasons even when QC fails."""
    write_table(
        output / "audit.tsv",
        [
            {
                "read_id": read_id,
                "reference_name": name,
                "nominal_length_nt": parse_nominal_length_nt(name),
                "status": audit[read_id],
            }
            for read_id, name in sorted(assigned.items())
        ],
        ("read_id", "reference_name", "nominal_length_nt", "status"),
    )


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    """Build head-Poly v4 or legacy flank-based v3 weak-label manifests."""
    from pod5 import Reader

    from holypoly.training.data import load_training_manifest

    head_poly = args.structure == "head-poly"
    strict_balance = args.strict_balance
    if strict_balance and (not head_poly or set(args.poly_types) != {"polyA", "polyT"}):
        raise ValueError("--strict-balance requires head-poly and both polyA/polyT")
    if head_poly and args.signal_end_policy != "raw-edge":
        raise ValueError("head-poly requires explicit --signal-end-policy raw-edge")
    if not head_poly and args.signal_end_policy is not None:
        raise ValueError("--signal-end-policy is only supported for head-poly")
    if not 1 <= args.min_candidate_samples <= args.chunk_size_samples:
        raise ValueError("min_candidate_samples must fit inside chunk_size_samples")
    started = datetime.now(UTC).isoformat()
    config = load_config(
        args.config, cli_data_root=args.data_root, cli_work_root=args.work_root
    )
    if config.paths.data_root is None or config.paths.work_root is None:
        raise ValueError("data_root and work_root are required")
    data_root = config.paths.data_root.resolve()
    work_root = config.paths.work_root.resolve()
    project = Path(__file__).resolve().parents[3]
    output = (work_root / args.output_dir).resolve()
    if (
        work_root.is_relative_to(data_root)
        or not output.is_relative_to(work_root)
        or output.is_relative_to(data_root)
        or output.is_relative_to(project)
        or output == work_root
    ):
        raise ValueError(
            "output must be a work_root child outside data_root and the repository"
        )
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    if not data_root.is_dir() or not work_root.is_dir():
        raise FileNotFoundError("configured data_root/work_root does not exist")
    bonito = (data_root / args.bonito_bam).resolve()
    dorado = (data_root / args.dorado_bam).resolve()
    reference = (data_root / args.reference).resolve()
    pod5_dir = (data_root / args.pod5_dir).resolve()
    for path in (bonito, dorado, reference, pod5_dir):
        if not path.is_relative_to(data_root) or not path.exists():
            raise ValueError(f"input must exist inside data_root: {path}")
    if args.chunk_size_samples < 12 or args.chunk_size_samples % 12:
        raise ValueError("chunk_size_samples must be positive and divisible by 12")
    if args.max_records < 0 or args.per_group < 3 or args.min_flank_reads < 2:
        raise ValueError("invalid record/group/support limits")
    if args.anchor_bases < 12 or not 0 <= args.anchor_error_rate < 0.5:
        raise ValueError("anchor_bases >= 12 and 0 <= anchor_error_rate < 0.5 required")
    exclusion_path = getattr(args, "exclude_read_ids", None)
    excluded_ids: set[str] = set()
    exclusion_sha256 = None
    if exclusion_path is not None:
        exclusion_path = (work_root / exclusion_path).resolve()
        if (
            not exclusion_path.is_file()
            or not exclusion_path.is_relative_to(work_root)
            or exclusion_path.is_relative_to(data_root)
        ):
            raise ValueError("exclude-read-ids must exist inside work_root")
        identifiers = exclusion_path.read_text().splitlines()
        excluded_ids = {str(UUID(identifier)) for identifier in identifiers}
        if not identifiers or len(excluded_ids) != len(identifiers):
            raise ValueError("exclude-read-ids must contain unique nonempty UUIDs")
        exclusion_sha256 = hashlib.sha256(exclusion_path.read_bytes()).hexdigest()
    if not 0.5 <= args.min_poly_fraction <= 1 or args.seed < 0:
        raise ValueError("invalid Poly fraction or seed")
    dorado_passes = 1 if head_poly else 2
    full_scan_gib = (
        bonito.stat().st_size + dorado_passes * dorado.stat().st_size
    ) / 1024**3
    print(
        f"CPU metadata preparation; {args.max_records or 'all'} records per pass. "
        f"Bonito once, Dorado {dorado_passes} pass(es); "
        f"full BAM scan <= {full_scan_gib:.2f} GiB. "
        "POD5 metadata only, no signal copies/GPU. Full HDD runs may take "
        "minutes to hours; audit output may reach hundreds of MB per million "
        "reads. Start with a bounded run.",
        file=sys.stderr,
        flush=True,
    )
    heads = load_heads(reference)
    positive_lengths = {
        length
        for name in heads
        if (length := parse_nominal_length_nt(name)) is not None and length > 0
    }
    target_groups = {
        (length, kind) for length in positive_lengths for kind in args.poly_types
    }
    if any(parse_nominal_length_nt(name) == 0 for name in heads):
        if strict_balance:
            target_groups.update({(0, "none:forward"), (0, "none:reverse")})
        else:
            target_groups.add((0, "none"))
    elif strict_balance:
        raise ValueError("--strict-balance requires a zero_nt construct in reference")
    if args.stop_when_full:
        print(
            f"Metadata collection will pause at {args.per_group} eligible reads "
            "per requested stratum"
            + (". " if head_poly else " once flank QC passes. ")
            + "Remaining BAM records "
            "still undergo lightweight UUID checks; sampling is from evaluated "
            "candidates, not uniformly from the whole file.",
            file=sys.stderr,
            flush=True,
        )
    assigned, conflicts, counts = construct_assignments(
        bonito, heads, args.max_records, progress=not args.no_progress
    )
    excluded_assigned = len(assigned.keys() & excluded_ids)
    for read_id in excluded_ids:
        assigned.pop(read_id, None)
    conflicts.difference_update(excluded_ids)
    output.mkdir(parents=True)
    audit = {
        read_id: "conflicting_bonito_constructs"
        if read_id in conflicts
        else "not_seen_in_dorado_scan"
        for read_id in assigned
    }
    write_audit(output, assigned, audit)
    candidates = []
    candidate_ids: set[str] = set()
    if not head_poly:
        with closing(
            bam_records(
                dorado,
                args.max_records,
                desc="2/3 Dorado flank candidates",
                progress=not args.no_progress,
            )
        ) as records:
            for record in records:
                if record.is_secondary or record.is_supplementary:
                    continue
                read_id = record.query_name
                name = assigned.get(read_id)
                if (
                    name is None
                    or read_id in conflicts
                    or read_id in candidate_ids
                    or name not in heads
                    or parse_nominal_length_nt(name) != 0
                    or record.reference_name != name
                ):
                    continue
                try:
                    end = head_boundary(
                        record, heads[name], args.anchor_bases, args.anchor_error_rate
                    )
                    candidate = record.query_sequence[end : end + args.anchor_bases]
                    if len(candidate) == args.anchor_bases and not set(candidate) - set(
                        "ACGT"
                    ):
                        candidates.append(candidate)
                        candidate_ids.add(read_id)
                except ValueError:
                    continue
                if len(candidates) >= 200:
                    break
    flank_reason = ""
    flank = ""
    if not head_poly:
        try:
            flank = infer_flank(
                candidates, args.min_flank_reads, args.anchor_error_rate
            )
        except ValueError as error:
            flank_reason = str(error)
    rng = random.Random(args.seed)
    rows: dict[str, dict[str, Any]] = {}
    cross_lengths: Counter[int] = Counter()
    group_counts: Counter[tuple[int, str]] = Counter()
    metadata_paused = False
    quota_reached = False
    quota_resumptions = 0
    metadata_records = 0
    metadata_skipped = 0
    seen: set[str] = set()
    with ExitStack() as stack:
        readers: dict[Path, Any] = {}
        records = stack.enter_context(
            closing(
                bam_records(
                    dorado,
                    args.max_records,
                    desc=(
                        "2/2 Dorado head / POD5 metadata"
                        if head_poly
                        else "3/3 Dorado boundaries / POD5 metadata"
                    ),
                    progress=not args.no_progress,
                )
            )
        )
        for record in records:
            counts["dorado_records"] += 1
            if record.is_secondary or record.is_supplementary:
                continue
            read_id = record.query_name
            if read_id not in assigned or read_id in conflicts:
                continue
            if read_id in seen:
                removed = rows.pop(read_id, None)
                if removed is not None:
                    group = _sampling_group(
                        removed, split_zero_orientation=strict_balance
                    )
                    group_counts[group] -= 1
                    if group_counts[group] == 0:
                        del group_counts[group]
                    if (
                        removed["nominal_length_nt"] > 0
                        and removed["supervision_mode"] == "full"
                    ):
                        length = removed["nominal_length_nt"]
                        cross_lengths[length] -= 1
                        if cross_lengths[length] == 0:
                            del cross_lengths[length]
                audit[read_id] = "duplicate_dorado_primary"
                if metadata_paused and not _metadata_quotas_met(
                    group_counts,
                    target_groups,
                    args.per_group,
                    cross_lengths,
                    args.min_flank_reads,
                    require_flank=not head_poly,
                ):
                    metadata_paused = False
                    quota_resumptions += 1
                    print(
                        "Duplicate primary invalidated a quota or quality support; "
                        "resuming boundary/POD5 metadata collection.",
                        file=sys.stderr,
                        flush=True,
                    )
                continue
            seen.add(read_id)
            if metadata_paused:
                audit[read_id] = "not_evaluated_quota_met"
                metadata_skipped += 1
                continue
            metadata_records += 1
            name = assigned[read_id]
            nominal = parse_nominal_length_nt(name)
            try:
                if name not in heads or nominal is None:
                    raise ValueError("unassigned_bonito_construct")
                if record.reference_name != name:
                    raise ValueError("bonito_dorado_construct_disagreement")
                requested_type = "polyT" if record.is_reverse else "polyA"
                if head_poly and requested_type not in args.poly_types:
                    raise ValueError(
                        "not_selected_poly_type"
                        if nominal > 0
                        else "not_selected_zero_orientation"
                    )
                head_end = head_boundary(
                    record, heads[name], args.anchor_bases, args.anchor_error_rate
                )
                boundary_reason = flank_reason
                flank_start = None
                if flank:
                    try:
                        flank_start = flank_boundary(
                            record.query_sequence,
                            head_end,
                            flank,
                            nominal,
                            args.anchor_error_rate,
                            args.min_poly_fraction,
                        )
                    except ValueError as error:
                        boundary_reason = str(error)
                if not head_poly and flank_start is None and nominal == 0:
                    raise ValueError(f"zero_nt_boundary_unverified:{boundary_reason}")
                # v4 waits for POD5 raw length before defining a raw-edge candidate.
                candidate_start, candidate_end, head_boundary_sample = (
                    (0, 0, 0)
                    if head_poly
                    else head_free_bounds(record, head_end, args.chunk_size_samples)
                )
                poly_start, poly_end = (
                    move_bounds(record, head_end, flank_start)
                    if flank_start is not None
                    else (candidate_start, candidate_end)
                )
                if not head_poly and nominal > 0 and poly_start >= poly_end:
                    raise ValueError("poly_bounds_share_move_block")
                if not record.has_tag("fn"):
                    raise ValueError("missing_pod5_filename")
                filename = str(record.get_tag("fn"))
                pod5_path = (pod5_dir / filename).resolve()
                if Path(filename).name != filename or not pod5_path.is_relative_to(
                    pod5_dir
                ):
                    raise ValueError("invalid_pod5_filename")
                if not pod5_path.is_file():
                    raise ValueError("missing_pod5_file")
                if pod5_path not in readers:
                    readers[pod5_path] = stack.enter_context(Reader(pod5_path))
                matches = list(
                    readers[pod5_path].reads(selection=[read_id], missing_ok=True)
                )
                if len(matches) != 1:
                    raise ValueError("missing_or_duplicate_pod5_read")
                raw_length = matches[0].num_samples
                # Dorado may omit a terminal fractional move block. Coordinates
                # still refer to the original signal; never tolerate a full block.
                dorado_samples = int(record.get_tag("ns"))
                move_stride = int(record.get_tag("mv")[0])
                if not 0 <= raw_length - dorado_samples < move_stride:
                    raise ValueError("pod5_dorado_signal_length_mismatch")
                if head_poly:
                    candidate_start, candidate_end, head_boundary_sample = (
                        head_poly_bounds(
                            record,
                            head_end,
                            raw_length,
                            args.chunk_size_samples,
                            args.min_candidate_samples,
                        )
                    )
                rows[read_id] = {
                    "manifest_version": HEAD_POLY_VERSION if head_poly else VERSION,
                    "read_id": read_id,
                    "pod5_path": str(pod5_path),
                    "bam_path": str(dorado),
                    "pod5_file_id": filename,
                    "split": "",
                    "orientation": "reverse" if record.is_reverse else "forward",
                    "raw_signal_length": raw_length,
                    "chunk_start_sample": candidate_start,
                    "chunk_end_sample": candidate_end,
                    "poly_start_sample": poly_start if nominal else "",
                    "poly_end_sample": poly_end if nominal else "",
                    "poly_type": ("polyT" if record.is_reverse else "polyA")
                    if nominal
                    else "none",
                    "training_length_nt": nominal,
                    "true_length_nt": nominal,
                    "nominal_length_nt": nominal,
                    "observed_sequence_length_nt": (
                        flank_start - head_end if flank_start is not None else ""
                    ),
                    "label_status": "poly_present" if nominal else "zero_nt",
                    "poly_is_complete": "true",
                    "failure_reason": "",
                    "label_source": "construct_nominal_weak_supervision",
                    "flank_source": "0A_downstream_consensus_cross_length_checked",
                    "flank_status": "provisional_both_anchors",
                    "selection_reason": "",
                    "reference_name": name,
                    "head_end_base": head_end,
                    "flank_start_base": flank_start,
                    "move_stride": int(record.get_tag("mv")[0]),
                    "trim_start_samples": int(record.get_tag("ts")),
                    "supervision_mode": "full",
                    "candidate_start_sample": candidate_start,
                    "candidate_end_sample": candidate_end,
                    "dorado_signal_samples": dorado_samples,
                    "model_input_samples": args.chunk_size_samples,
                    "head_excluded": "true",
                    "head_boundary_sample": head_boundary_sample,
                }
                if head_poly:
                    length_only_row(rows[read_id], "not_applicable_head_poly")
                    rows[read_id].update(
                        flank_status="not_applicable_no_flank",
                        flank_source="not_applicable_no_flank",
                    )
                elif flank_start is None:
                    length_only_row(rows[read_id], boundary_reason)
                row = rows[read_id]
                group_counts[
                    _sampling_group(row, split_zero_orientation=strict_balance)
                ] += 1
                if nominal > 0 and row["supervision_mode"] == "full":
                    cross_lengths[nominal] += 1
                audit[read_id] = "eligible"
                if args.stop_when_full and _metadata_quotas_met(
                    group_counts,
                    target_groups,
                    args.per_group,
                    cross_lengths,
                    args.min_flank_reads,
                    require_flank=not head_poly,
                ):
                    metadata_paused = True
                    quota_reached = True
                    print(
                        "All requested quotas and quality checks satisfied; pausing "
                        "boundary/POD5 metadata collection. Continuing lightweight "
                        "UUID validation to the scan limit/EOF.",
                        file=sys.stderr,
                        flush=True,
                    )
            except ValueError as error:
                audit[read_id] = str(error)
    write_audit(output, assigned, audit)
    flank_validated = (
        len(cross_lengths) >= 2 and sum(cross_lengths.values()) >= args.min_flank_reads
    )
    if not head_poly and not flank_validated:
        flank_reason = flank_reason or "insufficient_cross_length_flank_support"
        for read_id, row in list(rows.items()):
            if row["nominal_length_nt"] == 0:
                del rows[read_id]
                audit[read_id] = "zero_nt_flank_not_cross_validated"
            elif row["supervision_mode"] == "full":
                length_only_row(row, flank_reason)
    write_audit(output, assigned, audit)
    selected = balance_and_split(
        list(rows.values()),
        args.per_group,
        rng,
        poly_types=args.poly_types,
        balance_to_smallest=not head_poly,
        strict_groups=target_groups if strict_balance else None,
    )
    selected_ids = {row["read_id"] for row in selected}
    eligible_counts = Counter(
        _sampling_group(row, split_zero_orientation=strict_balance)
        for row in rows.values()
    )
    selected_group_counts = Counter(
        _sampling_group(row, split_zero_orientation=strict_balance) for row in selected
    )
    for read_id in assigned:
        if read_id not in audit:
            audit[read_id] = "not_seen_in_dorado_scan"
        elif read_id in selected_ids:
            audit[read_id] = f"selected_{rows[read_id]['supervision_mode']}"
        elif audit[read_id] == "eligible":
            row = rows[read_id]
            if row["nominal_length_nt"] > 0 and row["poly_type"] not in args.poly_types:
                audit[read_id] = "not_selected_poly_type"
                continue
            group_count = eligible_counts[
                _sampling_group(row, split_zero_orientation=strict_balance)
            ]
            audit[read_id] = (
                "not_selected_insufficient_group"
                if group_count < 3
                else "not_selected_balancing"
            )
    summary: dict[str, Any] = {
        "manifest_version": HEAD_POLY_VERSION if head_poly else VERSION,
        "construct_structure": args.structure,
        "flank_sequence": flank,
        "flank_status": (
            "not_applicable_no_flank"
            if head_poly
            else "provisional_cross_checked"
            if flank_validated
            else "unverified"
        ),
        "flank_qc_reason": flank_reason,
        "0A_candidate_reads": len(candidates),
        "cross_length_flank_support": dict(cross_lengths),
        "seed": args.seed,
        "balancing": {
            "mode": "strict_length_type_zero_orientation"
            if strict_balance
            else "independent_caps"
            if head_poly
            else "legacy_positive_minimum",
            "requested_per_group": args.per_group,
            "effective_per_group": min(selected_group_counts.values())
            if strict_balance
            else None,
            "zero_nt_grouping": "orientation" if strict_balance else "pooled",
        },
        "input_policy": (
            "post_head_raw_edge_candidate_right_padded"
            if head_poly
            else "post_head_called_signal_right_padded"
        ),
        "poly_endpoint_verified": False if head_poly else None,
        "head_excluded_reads": sum(row["head_excluded"] == "true" for row in selected),
        "padded_reads": sum(
            row["chunk_end_sample"] - row["chunk_start_sample"]
            < row["model_input_samples"]
            for row in selected
        ),
        "arguments": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "inputs": [str(path) for path in (bonito, dorado, reference, pod5_dir)],
        "scan_counts": dict(counts),
        "metadata_collection": {
            "stop_when_full": args.stop_when_full,
            "quota_reached": quota_reached,
            "final_quotas_met": _metadata_quotas_met(
                group_counts,
                target_groups,
                args.per_group,
                cross_lengths,
                args.min_flank_reads,
                require_flank=not head_poly,
            ),
            "target_groups": [
                f"{length}:{kind}" for length, kind in sorted(target_groups)
            ],
            "records_evaluated": metadata_records,
            "records_skipped_after_quota": metadata_skipped,
            "quota_resumptions": quota_resumptions,
            "candidate_counts_scope": "evaluated_records_only"
            if metadata_skipped
            else "all_scanned_candidates",
            "uuid_validation_scope": "full_file"
            if args.max_records == 0
            else "record_limit",
            "sampling_policy": "random_within_evaluated_candidates"
            if metadata_skipped
            else "random_within_all_scanned_candidates",
        },
        "audit_counts": dict(Counter(audit.values())),
        "selected_counts": dict(
            Counter(
                f"{row['split']}:{row['nominal_length_nt']}:{row['poly_type']}"
                for row in selected
            )
        ),
        "selected_stratum_counts": dict(
            Counter(
                f"{row['split']}:{length}:{kind}"
                for row in selected
                for length, kind in [
                    _sampling_group(row, split_zero_orientation=strict_balance)
                ]
            )
        ),
        "supervision_counts": dict(
            Counter(row["supervision_mode"] for row in selected)
        ),
        "eligible_counts": {
            f"{length}:{kind}": count
            for (length, kind), count in sorted(eligible_counts.items())
        },
        "excluded_poly_type_counts": {
            f"{length}:{kind}": count
            for (length, kind), count in sorted(eligible_counts.items())
            if length > 0 and kind not in args.poly_types
        },
        "missing_positive_strata": [
            f"{length}:{kind}"
            for length in sorted(positive_lengths)
            for kind in dict.fromkeys(args.poly_types)
            if not any(
                row["nominal_length_nt"] == length and row["poly_type"] == kind
                for row in selected
            )
        ],
        "quota_deficits": {
            f"{length}:{kind}": max(
                0, args.per_group - selected_group_counts[(length, kind)]
            )
            for length, kind in sorted(target_groups)
        },
        "started_utc": started,
        "finished_utc": datetime.now(UTC).isoformat(),
        "python_version": sys.version,
        "limitations": (
            (
                "Head-poly structure supplied by the user. All targets are construct "
                "nominal weak labels, including zero_nt. No flank was inferred. "
                "The raw-edge candidate can include adapter/noise. "
                "It does not establish a Poly endpoint or completeness. "
                "Only length is supervised. "
                "Head exclusion follows Dorado alignment/moves and is provisional. "
                "Zero reads without real candidate signal are excluded. "
                "No formal accuracy claims."
            )
            if head_poly
            else (
                "Construct lengths and move/anchor boundaries are weak labels, "
                "not read-level ground truth. Head exclusion follows the accepted "
                "Dorado boundary; alignment/move errors remain possible. Retained "
                "signal includes possible flank and downstream background. "
                "No formal accuracy claims."
            )
        )
        + (
            " Metadata quotas limited candidate evaluation to BAM-order subsets; "
            "these are not a uniform sample of all eligible reads."
            if metadata_skipped
            else ""
        ),
    }
    if exclusion_path is not None:
        summary["read_exclusion"] = {
            "path": str(exclusion_path),
            "sha256": exclusion_sha256,
            "requested_ids": len(excluded_ids),
            "excluded_assigned_ids": excluded_assigned,
            "selected_overlap": len(
                {row["read_id"] for row in selected} & excluded_ids
            ),
        }
    for label, repo in (
        ("git_commit", project),
        ("bonito_commit", project / "src/bonito"),
    ):
        summary[label] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
    write_table(output / "manifest.tsv", selected, FIELDS)
    load_training_manifest(output / "manifest.tsv")
    write_audit(output, assigned, audit)
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    """Expose dataset-specific paths and a small set of auditable thresholds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument(
        "--structure",
        choices=("head-poly-flank", "head-poly"),
        default="head-poly-flank",
        help="head-poly selects v4 without flank inference, using length-only labels.",
    )
    parser.add_argument(
        "--signal-end-policy",
        choices=("raw-edge",),
        help="Required for v4: raw edge bounds a candidate, not a verified Poly.",
    )
    parser.add_argument("--bonito-bam", type=Path, required=True)
    parser.add_argument("--dorado-bam", type=Path, required=True)
    parser.add_argument("--pod5-dir", type=Path, required=True)
    parser.add_argument("--reference", type=Path, default=Path("fixed_len_ref.fa"))
    parser.add_argument(
        "--output-dir", type=Path, default=Path("metadata/poly-dataset-v3")
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=10_000,
        help="Per BAM pass; 0 scans all. Dorado is read twice.",
    )
    parser.add_argument(
        "--no-progress", action="store_true", help="Disable progress bars on stderr."
    )
    parser.add_argument("--per-group", type=int, default=1000)
    parser.add_argument(
        "--exclude-read-ids",
        type=Path,
        help="Work-root text file of UUIDs to exclude before candidate checks/quotas",
    )
    parser.add_argument(
        "--strict-balance",
        action="store_true",
        help=(
            "v4 with both types: require every reference positive length/type and "
            "both zero_nt orientations; downsample all to the smallest eligible "
            "count, capped by --per-group. Fail if any stratum has fewer than 3. "
            "Split each stratum separately. Existing defaults remain unchanged."
        ),
    )
    parser.add_argument(
        "--stop-when-full",
        action="store_true",
        help=(
            "Pause boundary/POD5 metadata checks when every requested stratum "
            "(including zero_nt, when referenced) reaches --per-group. Legacy v3 "
            "also requires flank QC. Continue UUID checks; sampling uses a BAM-order "
            "subset, not all eligible reads."
        ),
    )
    parser.add_argument(
        "--poly-types",
        nargs="+",
        choices=("polyA", "polyT"),
        default=["polyA", "polyT"],
        help=(
            "Positive types to balance and retain; zero_nt is retained separately. "
            "Both types still contribute to the unchanged flank quality checks."
        ),
    )
    parser.add_argument("--chunk-size-samples", type=int, default=12000)
    parser.add_argument(
        "--min-candidate-samples",
        type=int,
        default=12,
        help="v4 minimum real candidate size; a technical floor, not proof of Poly.",
    )
    parser.add_argument("--anchor-bases", type=int, default=24)
    parser.add_argument("--anchor-error-rate", type=float, default=0.1)
    parser.add_argument("--min-poly-fraction", type=float, default=0.8)
    parser.add_argument("--min-flank-reads", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260923)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run preparation without loading a model or using a GPU."""
    try:
        summary = prepare(build_parser().parse_args(argv))
    except (OSError, ValueError) as error:
        print(f"Dataset preparation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
