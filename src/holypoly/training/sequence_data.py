"""Consume separate reference-head labels without altering Poly length targets."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from holypoly.datasets.prepare_sequence import BOUNDARY_SOURCE, LABEL_SOURCE, VERSION
from holypoly.training.data import ManifestSplit, PolyTrainingRecord, SignalReader


@dataclass(frozen=True)
class SequenceRecord:
    """One head-only sequence and its provisional raw-sample interval."""

    read_id: str
    pod5_path: Path
    split: ManifestSplit
    raw_signal_length: int
    chunk_start_sample: int
    chunk_end_sample: int
    sequence_target: str
    source_manifest_sha256: str
    reference_sha256: str


@dataclass(frozen=True)
class SequenceExample:
    """Unpadded float32 signal [S] and int64 A/C/G/T targets [L] in raw order."""

    signal: Tensor
    targets: Tensor
    record: SequenceRecord


def load_sequence_manifest(
    path: Path,
    length_records: Sequence[PolyTrainingRecord],
    *,
    source_manifest_sha256: str,
    data_root: Path,
) -> tuple[SequenceRecord, ...]:
    """Enforce read identity, split, provenance and disjoint head/Poly intervals."""
    by_id = {row.read_id: row for row in length_records}
    if len(by_id) != len(length_records):
        raise ValueError(
            "joint training requires unique length read IDs across all splits"
        )
    required = {
        "manifest_version",
        "read_id",
        "pod5_path",
        "split",
        "orientation",
        "raw_signal_length",
        "chunk_start_sample",
        "chunk_end_sample",
        "sequence_target",
        "reference_start_base",
        "reference_end_base",
        "label_source",
        "boundary_source",
        "source_manifest_sha256",
        "reference_sha256",
    }
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("sequence manifest is missing required fields")
        rows = list(reader)
    if not rows:
        raise ValueError("sequence manifest is empty; inspect preparation audit")
    result, seen = [], set()
    references = set()
    for row in rows:
        read_id = row["read_id"]
        if read_id in seen or read_id not in by_id:
            raise ValueError(
                "sequence read ID duplicated or absent from length manifest"
            )
        seen.add(read_id)
        source = by_id[read_id]
        path_value = Path(row["pod5_path"]).expanduser().resolve()
        if (
            path_value != source.pod5_path.resolve()
            or not path_value.is_relative_to(data_root)
            or not path_value.is_file()
            or row["split"] != source.split.value
            or row["orientation"] != source.orientation
        ):
            raise ValueError(
                "sequence source path/split/orientation differs from length manifest"
            )
        if (
            row["manifest_version"] != VERSION
            or row["label_source"] != LABEL_SOURCE
            or row["boundary_source"] != BOUNDARY_SOURCE
            or row["source_manifest_sha256"] != source_manifest_sha256
        ):
            raise ValueError("sequence label provenance differs from expected source")
        digest = row["reference_sha256"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("invalid reference SHA256")
        references.add(digest)
        raw, start, end = (
            int(row[key])
            for key in ("raw_signal_length", "chunk_start_sample", "chunk_end_sample")
        )
        target = row["sequence_target"]
        ref_start, ref_end = (
            int(row["reference_start_base"]),
            int(row["reference_end_base"]),
        )
        if (
            raw != source.raw_signal_length
            or not 0 <= start < end <= raw
            or not target
            or set(target) - set("ACGT")
            or ref_start < 0
            or ref_end - ref_start != len(target)
        ):
            raise ValueError("invalid sequence target, interval or raw signal length")
        if (source.orientation == "forward" and end > source.chunk_start_sample) or (
            source.orientation == "reverse" and start < source.chunk_end_sample
        ):
            raise ValueError("sequence interval overlaps Poly candidate")
        result.append(
            SequenceRecord(
                read_id,
                path_value,
                source.split,
                raw,
                start,
                end,
                target,
                source_manifest_sha256,
                digest,
            )
        )
    if len(references) != 1:
        raise ValueError("sequence manifest mixes reference versions")
    return tuple(result)


class SequenceDataset(Dataset[SequenceExample]):
    """Read explicit head intervals without reversing or truncating targets."""

    def __init__(
        self,
        records: Sequence[SequenceRecord],
        reader: SignalReader,
        *,
        split: ManifestSplit,
    ) -> None:
        self.records = tuple(record for record in records if record.split == split)
        self.reader = reader
        if not self.records:
            raise ValueError(f"no accepted sequence reads in {split.value} split")

    def __len__(self) -> int:
        """Return the number of accepted sequence chunks in this split."""
        return len(self.records)

    def __getitem__(self, index: int) -> SequenceExample:
        """Return an unpadded calibrated interval and acquisition-order targets."""
        record = self.records[index]
        full = self.reader(record.pod5_path, record.read_id)
        if full.ndim != 1 or len(full) != record.raw_signal_length:
            raise ValueError("sequence POD5 signal length differs from manifest")
        signal = (
            full[record.chunk_start_sample : record.chunk_end_sample]
            .float()
            .contiguous()
        )
        if not torch.isfinite(signal).all():
            raise ValueError("sequence signal contains NaN or Inf")
        targets = torch.tensor(
            ["NACGT".index(base) for base in record.sequence_target], dtype=torch.long
        )
        return SequenceExample(signal, targets, record)


def collate_sequence_examples(
    examples: Sequence[SequenceExample],
) -> list[SequenceExample]:
    """Keep variable raw lengths separate so CRF loss never includes batch padding."""
    return list(examples)
