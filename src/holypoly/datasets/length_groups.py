"""Classify Dorado-aligned reads by reference-name nominal Poly length."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import sys
from collections import Counter
from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, TextIO

MANIFEST_VERSION = "nominal-length-groups-v1"
NOMINAL_LENGTH_PATTERN = re.compile(r"(?:^|_)(\d+)A(?:_|$)")
READ_FIELDS = (
    "read_id",
    "source_batch",
    "nominal_length_nt",
    "reference_name",
    "classification_status",
    "bam_path",
    "pod5_path",
    "pod5_file_id",
    "alignment_orientation",
    "mapping_quality",
    "flag",
    "query_length",
    "reference_start",
    "reference_end",
    "cigar",
    "signal_samples",
    "trim_start_samples",
    "move_stride",
    "label_source",
    "manifest_version",
)


@dataclass(frozen=True)
class ReferenceRecord:
    """Reference metadata with a nominal length parsed from its name."""

    reference_name: str
    reference_length_bases: int
    nominal_length_nt: int | None


def parse_nominal_length_nt(reference_name: str) -> int | None:
    """Extract ``<number>A`` from a reference name without inferring truth."""
    matches = NOMINAL_LENGTH_PATTERN.findall(reference_name)
    if len(matches) != 1:
        return None
    return int(matches[0])


def load_reference_catalog(reference_path: Path) -> tuple[ReferenceRecord, ...]:
    """Read FASTA names and sequence lengths without modifying the reference."""
    if not reference_path.is_file():
        raise FileNotFoundError(f"reference FASTA does not exist: {reference_path}")
    records: list[ReferenceRecord] = []
    name: str | None = None
    sequence_length = 0
    with reference_path.open(encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if name is not None:
                    records.append(
                        ReferenceRecord(
                            reference_name=name,
                            reference_length_bases=sequence_length,
                            nominal_length_nt=parse_nominal_length_nt(name),
                        )
                    )
                name = line[1:].split()[0]
                sequence_length = 0
            else:
                if name is None:
                    raise ValueError("reference FASTA sequence appears before a header")
                sequence_length += len(line)
    if name is not None:
        records.append(
            ReferenceRecord(
                reference_name=name,
                reference_length_bases=sequence_length,
                nominal_length_nt=parse_nominal_length_nt(name),
            )
        )
    if not records:
        raise ValueError(f"reference FASTA contains no sequences: {reference_path}")
    duplicate_names = [
        reference_name
        for reference_name, count in Counter(
            record.reference_name for record in records
        ).items()
        if count > 1
    ]
    if duplicate_names:
        raise ValueError(
            "reference FASTA contains duplicate names: "
            + ", ".join(sorted(duplicate_names))
        )
    return tuple(records)


def discover_dorado_bams(data_root: Path) -> tuple[Path, ...]:
    """Return one Dorado BAM per immediate source-batch directory."""
    bam_root = data_root / "dorado_calls"
    bams = tuple(sorted(bam_root.glob("*/dorado_basecalls.bam")))
    if not bams:
        raise FileNotFoundError(f"no Dorado BAM files found under {bam_root}")
    return bams


def build_length_groups(
    *,
    data_root: Path,
    reference_path: Path,
    bam_paths: Sequence[Path],
    output_directory: Path,
    progress_every: int = 100_000,
) -> dict[str, Any]:
    """Create compressed per-length read tables without copying raw inputs."""
    data_root = data_root.expanduser().resolve()
    reference_path = reference_path.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    if output_directory == data_root or not output_directory.is_relative_to(data_root):
        raise ValueError("output_directory must be a dedicated child of data_root")
    if output_directory.exists():
        raise FileExistsError(f"output directory already exists: {output_directory}")
    if progress_every < 1:
        raise ValueError("progress_every must be positive")

    references = load_reference_catalog(reference_path)
    reference_lengths = {
        record.reference_name: record.nominal_length_nt for record in references
    }
    group_names = {
        f"{length_nt}nt"
        for length_nt in reference_lengths.values()
        if length_nt is not None
    }
    group_names.add("unassigned")
    staging_directory = output_directory.with_name(output_directory.name + ".tmp")
    if staging_directory.exists():
        raise FileExistsError(f"staging directory already exists: {staging_directory}")
    staging_directory.mkdir(parents=True)

    summary: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "reference_path": str(reference_path),
        "data_root": str(data_root),
        "output_directory": str(output_directory),
        "bam_paths": [str(path.expanduser().resolve()) for path in bam_paths],
        "records_seen": 0,
        "unique_reads": 0,
        "ignored_secondary_alignments": 0,
        "ignored_supplementary_alignments": 0,
        "duplicate_primary_read_ids": 0,
        "counts_by_group": {},
        "counts_by_status": {},
        "counts_by_source_batch": {},
        "counts_by_orientation": {},
        "missing_pod5_files": [],
    }
    counts_by_group: Counter[str] = Counter()
    counts_by_status: Counter[str] = Counter()
    counts_by_source_batch: Counter[str] = Counter()
    counts_by_orientation: Counter[str] = Counter()
    seen_read_ids: set[str] = set()
    checked_pod5_paths: dict[Path, bool] = {}

    try:
        with ExitStack() as stack:
            (staging_directory / "metadata").mkdir(parents=True, exist_ok=True)
            writers = {
                group_name: stack.enter_context(
                    _open_group_writer(staging_directory, group_name)
                )
                for group_name in sorted(group_names)
            }
            duplicate_handle = stack.enter_context(
                gzip.open(
                    staging_directory / "metadata" / "duplicate_read_ids.txt.gz",
                    "wt",
                    encoding="utf-8",
                )
            )
            for row in _iter_classified_reads(
                data_root=data_root,
                bam_paths=bam_paths,
                reference_lengths=reference_lengths,
                summary=summary,
                seen_read_ids=seen_read_ids,
                checked_pod5_paths=checked_pod5_paths,
                duplicate_handle=duplicate_handle,
                progress_every=progress_every,
            ):
                group_name = _group_name(row["nominal_length_nt"])
                writers[group_name].writerow(row)
                counts_by_group[group_name] += 1
                counts_by_status[str(row["classification_status"])] += 1
                counts_by_source_batch[str(row["source_batch"])] += 1
                counts_by_orientation[str(row["alignment_orientation"])] += 1

        summary["unique_reads"] = len(seen_read_ids)
        summary["counts_by_group"] = dict(sorted(counts_by_group.items()))
        summary["counts_by_status"] = dict(sorted(counts_by_status.items()))
        summary["counts_by_source_batch"] = dict(sorted(counts_by_source_batch.items()))
        summary["counts_by_orientation"] = dict(sorted(counts_by_orientation.items()))
        summary["missing_pod5_files"] = sorted(
            str(path) for path, exists in checked_pod5_paths.items() if not exists
        )
        _write_reference_catalog(staging_directory, references)
        _write_summary(staging_directory, summary)
        _write_dataset_readme(staging_directory)
        staging_directory.replace(output_directory)
    except Exception:
        shutil.rmtree(staging_directory, ignore_errors=True)
        raise
    return summary


def _iter_classified_reads(
    *,
    data_root: Path,
    bam_paths: Sequence[Path],
    reference_lengths: dict[str, int | None],
    summary: dict[str, Any],
    seen_read_ids: set[str],
    checked_pod5_paths: dict[Path, bool],
    duplicate_handle: TextIO,
    progress_every: int,
) -> Iterator[dict[str, object]]:
    import pysam

    for bam_path_value in bam_paths:
        bam_path = bam_path_value.expanduser().resolve()
        if not bam_path.is_file():
            raise FileNotFoundError(f"Dorado BAM does not exist: {bam_path}")
        source_batch = bam_path.parent.name
        with pysam.AlignmentFile(str(bam_path), "rb") as bam_file:
            for alignment in bam_file.fetch(until_eof=True):
                summary["records_seen"] += 1
                records_seen = int(summary["records_seen"])
                if records_seen % progress_every == 0:
                    print(
                        f"processed {records_seen:,} BAM records",
                        file=sys.stderr,
                        flush=True,
                    )
                if alignment.is_secondary:
                    summary["ignored_secondary_alignments"] += 1
                    continue
                if alignment.is_supplementary:
                    summary["ignored_supplementary_alignments"] += 1
                    continue

                read_id = alignment.query_name
                if read_id in seen_read_ids:
                    summary["duplicate_primary_read_ids"] += 1
                    duplicate_handle.write(f"{read_id}\t{bam_path}\n")
                    continue
                seen_read_ids.add(read_id)

                reference_name = alignment.reference_name or ""
                nominal_length_nt = reference_lengths.get(reference_name)
                status = _classification_status(
                    is_unmapped=alignment.is_unmapped,
                    reference_name=reference_name,
                    nominal_length_nt=nominal_length_nt,
                )
                pod5_file_id = _optional_tag(alignment, "fn")
                pod5_path = ""
                if pod5_file_id:
                    candidate = data_root / "pod5" / source_batch / str(pod5_file_id)
                    if candidate not in checked_pod5_paths:
                        checked_pod5_paths[candidate] = candidate.is_file()
                    exists = checked_pod5_paths[candidate]
                    pod5_path = str(candidate)
                    if not exists and status == "assigned":
                        status = "missing_pod5_file"
                elif status == "assigned":
                    status = "missing_pod5_file_tag"

                move_table = _optional_tag(alignment, "mv")
                move_stride: int | str = ""
                if move_table is not None and len(move_table) > 0:
                    move_stride = int(move_table[0])
                yield {
                    "read_id": read_id,
                    "source_batch": source_batch,
                    "nominal_length_nt": (
                        "" if nominal_length_nt is None else nominal_length_nt
                    ),
                    "reference_name": reference_name,
                    "classification_status": status,
                    "bam_path": str(bam_path),
                    "pod5_path": pod5_path,
                    "pod5_file_id": pod5_file_id or "",
                    "alignment_orientation": (
                        "reverse" if alignment.is_reverse else "forward"
                    ),
                    "mapping_quality": alignment.mapping_quality,
                    "flag": alignment.flag,
                    "query_length": alignment.query_length or 0,
                    "reference_start": (
                        "" if alignment.is_unmapped else alignment.reference_start
                    ),
                    "reference_end": (
                        "" if alignment.is_unmapped else alignment.reference_end
                    ),
                    "cigar": alignment.cigarstring or "",
                    "signal_samples": _optional_tag(alignment, "ns") or "",
                    "trim_start_samples": _optional_tag(alignment, "ts") or "",
                    "move_stride": move_stride,
                    "label_source": "reference_name_nominal",
                    "manifest_version": MANIFEST_VERSION,
                }


def _classification_status(
    *, is_unmapped: bool, reference_name: str, nominal_length_nt: int | None
) -> str:
    if is_unmapped:
        return "unmapped"
    if not reference_name:
        return "missing_reference_name"
    if nominal_length_nt is None:
        return "unknown_reference_length"
    return "assigned"


def _optional_tag(alignment: Any, tag: str) -> Any | None:
    return alignment.get_tag(tag) if alignment.has_tag(tag) else None


def _group_name(nominal_length_nt: object) -> str:
    return "unassigned" if nominal_length_nt == "" else f"{nominal_length_nt}nt"


@contextmanager
def _open_group_writer(
    staging_directory: Path, group_name: str
) -> Iterator[csv.DictWriter[str]]:
    group_directory = staging_directory / group_name
    group_directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(
        group_directory / "reads.tsv.gz",
        "wt",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=READ_FIELDS, delimiter="\t")
        writer.writeheader()
        yield writer


def _write_reference_catalog(
    staging_directory: Path, references: Sequence[ReferenceRecord]
) -> None:
    metadata_directory = staging_directory / "metadata"
    metadata_directory.mkdir(parents=True, exist_ok=True)
    with (metadata_directory / "reference_catalog.tsv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "reference_name",
                "reference_length_bases",
                "nominal_length_nt",
                "label_source",
            ),
            delimiter="\t",
        )
        writer.writeheader()
        for reference in references:
            writer.writerow(
                {
                    **asdict(reference),
                    "nominal_length_nt": (
                        ""
                        if reference.nominal_length_nt is None
                        else reference.nominal_length_nt
                    ),
                    "label_source": "reference_name_nominal",
                }
            )


def _write_summary(staging_directory: Path, summary: dict[str, Any]) -> None:
    metadata_directory = staging_directory / "metadata"
    metadata_directory.mkdir(parents=True, exist_ok=True)
    with (metadata_directory / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_dataset_readme(staging_directory: Path) -> None:
    (staging_directory / "README.md").write_text(
        "# Nominal Poly length read groups\n\n"
        "Reads are grouped by the `<number>A` token in the aligned reference name.\n"
        "`nominal_length_nt` is a construct-level weak label, not independently\n"
        "validated read-level ground truth. Original BAM and POD5 files are\n"
        "referenced by path and are not copied or modified. These tables do not\n"
        "contain signal-level Poly interval labels required by location-head "
        "training.\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the nominal-length grouping command parser."""
    parser = argparse.ArgumentParser(
        description="Group Dorado-aligned reads by reference-name nominal Poly length."
    )
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bam", type=Path, action="append", dest="bam_paths")
    parser.add_argument("--progress-every", type=int, default=100_000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run read classification and print the final JSON summary."""
    args = build_parser().parse_args(argv)
    try:
        data_root = args.data_root.expanduser().resolve()
        bam_paths = (
            discover_dorado_bams(data_root)
            if not args.bam_paths
            else tuple(args.bam_paths)
        )
        summary = build_length_groups(
            data_root=data_root,
            reference_path=args.reference,
            bam_paths=bam_paths,
            output_directory=args.output_dir,
            progress_every=args.progress_every,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as error:
        print(f"Length grouping failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
