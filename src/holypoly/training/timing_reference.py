"""Strict manifest-bound timing-table loading, without fitting statistics."""

from __future__ import annotations

import csv
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from holypoly.datasets.prepare_sequence import sha256
from holypoly.inference.timing_reference import (
    TIMING_REFERENCE_SCHEMA,
    TimingWindowConfig,
    timing_vector,
)
from holypoly.training.data import PolyTrainingRecord


def load_timing_references(
    path: Path,
    records: Sequence[PolyTrainingRecord],
    *,
    source_manifest: Path,
) -> dict[str, tuple[float, float, float]]:
    """Require every source read, including explicit unavailable rows and splits.

    SHA256 ties this table to its source manifest and extraction summary. Old
    diagnostic tables are deliberately not interchangeable with this schema.
    """
    metadata = json.loads((path.parent / "summary.json").read_text())
    if (
        metadata.get("schema") != TIMING_REFERENCE_SCHEMA
        or metadata.get("source_manifest_sha256") != sha256(source_manifest)
        or metadata.get("timing_table_sha256") != sha256(path)
        or metadata.get("window_config") != asdict(TimingWindowConfig())
    ):
        raise ValueError("timing reference provenance/config/hash mismatch")
    expected = {row.read_id: row for row in records}
    if len(expected) != len(records):
        raise ValueError("timing reference requires unique source read IDs")
    result: dict[str, tuple[float, float, float]] = {}
    with path.open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            read_id = row["read_id"]
            if read_id in result or read_id not in expected:
                raise ValueError("duplicate/unexpected read in timing reference table")
            record = expected[read_id]
            if (
                row["split"] != record.split.value
                or int(row["raw_signal_length"]) != record.raw_signal_length
            ):
                raise ValueError("timing reference read split/raw length mismatch")
            result[read_id] = timing_vector(dict(row))
    if result.keys() != expected.keys():
        raise ValueError(
            "timing table must retain every source read; use explicit missing rows"
        )
    return result
