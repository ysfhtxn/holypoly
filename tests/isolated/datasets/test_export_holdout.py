"""Synthetic balanced-holdout selection, exclusion and lossless POD5 export."""

from __future__ import annotations

import csv
import inspect
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID

import numpy as np
import pod5
import pytest

from holypoly.datasets.export_holdout import (
    export_full_reads,
    file_sha256,
    history_exclusions,
    select_balanced,
)


def rows() -> list[dict[str, str]]:
    """Make synthetic weak-label groups with unequal sizes."""
    return [
        {
            "read_id": str(UUID(int=100 * length + index + 1)),
            "nominal_length_nt": str(length),
            "poly_type": "polyA" if length else "none",
            "label_status": "poly_present" if length else "zero_nt",
            "label_source": "synthetic_construct_nominal",
            "orientation": "forward",
        }
        for length, count in ((0, 4), (20, 6), (40, 5))
        for index in range(count)
    ]


def test_balancing_excludes_before_count_and_is_order_independent() -> None:
    source = rows()
    excluded = {source[0]["read_id"]}
    options = {"lengths": [0, 20, 40], "per_group": 500, "seed": 7}
    selected, counts = select_balanced(source, excluded, **options)
    repeated, _ = select_balanced(list(reversed(source)), excluded, **options)
    assert selected == repeated
    assert counts == {0: 3, 20: 6, 40: 5}
    assert len(selected) == 9
    assert not {row["read_id"] for row in selected} & excluded
    assert [
        sum(row["nominal_length_nt"] == str(n) for row in selected) for n in (0, 20, 40)
    ] == [3, 3, 3]
    with pytest.raises(ValueError, match="missing independent"):
        select_balanced(source, set(), lengths=[0, 150], per_group=500, seed=1)
    source[-1]["poly_type"] = "polyT"
    with pytest.raises(ValueError, match="disagree"):
        select_balanced(source, set(), **options)


def test_history_exclusions_union_and_cross_split_safety(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.tsv"
    with manifest.open("w", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["read_id", "split"])
        writer.writerows(
            (str(UUID(int=index)), split)
            for index, split in (
                (1, "train"),
                (1, "train"),
                (2, "validation"),
                (3, "test"),
            )
        )
    excluded, provenance = history_exclusions([manifest])
    assert excluded == {str(UUID(int=1)), str(UUID(int=2))}
    assert provenance[0]["sha256"] == file_sha256(manifest)
    all_ids, _ = history_exclusions([manifest], all_splits=True)
    assert all_ids == excluded | {str(UUID(int=3))}
    with manifest.open("a") as stream:
        stream.write(f"{UUID(int=1)}\ttest\n")
    with pytest.raises(ValueError, match="crosses splits"):
        history_exclusions([manifest], all_splits=True)


def test_whole_signal_export_roundtrip(tmp_path: Path) -> None:
    source_root = tmp_path / "raw"
    source_root.mkdir()
    source = source_root / "synthetic.pod5"
    metadata = {
        name: "synthetic" for name in inspect.signature(pod5.RunInfo).parameters
    }
    metadata.update(
        acquisition_start_time=datetime(2026, 1, 1, tzinfo=UTC),
        protocol_start_time=datetime(2026, 1, 1, tzinfo=UTC),
        adc_max=8191,
        adc_min=-8192,
        context_tags={},
        tracking_id={},
        sample_rate=4000,
    )
    run_info = pod5.RunInfo(**metadata)
    signals = [np.arange(123 + index * 17, dtype=np.int16) - 60 for index in range(3)]
    with pod5.Writer(source) as writer:
        writer.add_reads(
            [
                pod5.Read(
                    read_id=UUID(int=index + 1),
                    pore=pod5.Pore(channel=index + 1, well=1, pore_type="synthetic"),
                    calibration=pod5.Calibration(offset=-5, scale=0.25),
                    read_number=index,
                    start_sample=index * 1000,
                    median_before=float("nan"),
                    end_reason=pod5.EndReason(pod5.EndReasonEnum.UNKNOWN, False),
                    run_info=run_info,
                    signal=signal,
                )
                for index, signal in enumerate(signals)
            ]
        )
    before = file_sha256(source)
    output = tmp_path / "export.pod5"
    selected = [
        {"read_id": str(UUID(int=index)), "pod5_path": source.name} for index in (1, 3)
    ]
    signatures = export_full_reads(selected, output, data_root=source_root)
    assert len(signatures) == 2
    assert file_sha256(source) == before
    with pod5.Reader(output) as reader:
        assert set(reader.read_ids) == {row["read_id"] for row in selected}
        for record in reader.reads():
            np.testing.assert_array_equal(
                record.signal, signals[record.read_id.int - 1]
            )
            assert record.start_sample == (record.read_id.int - 1) * 1000
    with pytest.raises(ValueError, match="source POD5"):
        export_full_reads(
            [{"read_id": str(UUID(int=1)), "pod5_path": "../absent.pod5"}],
            tmp_path / "invalid.pod5",
            data_root=source_root,
        )
