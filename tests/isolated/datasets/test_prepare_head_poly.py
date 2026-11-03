"""Synthetic v4 head-Poly preparation and weak zero-label regression checks."""

from __future__ import annotations

import csv
import random
from array import array
from collections import Counter
from pathlib import Path
from uuid import UUID

import pysam
import pytest
import torch

from holypoly.datasets import prepare as module
from holypoly.training.data import (
    PolyTrainingDataset,
    collate_poly_training_examples,
    load_training_manifest,
)
from tests.isolated.datasets.test_prepare import HEAD, _quota_setup, alignment


def terminal_alignment(
    index: int, length: int, *, reverse: bool = False, suffix: str | None = None
) -> pysam.AlignedSegment:
    """Make a head-terminal construct, optionally with called terminal adapter."""
    record = alignment(str(UUID(int=index)), length, reverse)
    tail = ("A" * length if length else "CGTACG") if suffix is None else suffix
    record.query_sequence = HEAD + tail
    record.cigar = [(0, len(HEAD))] + ([(4, len(tail))] if tail else [])
    record.set_tag("mv", array("b", [6] + [1, 0] * record.query_length))
    record.set_tag("ns", 10 + record.query_length * 12)
    return record


@pytest.mark.parametrize("reverse", [False, True])
def test_raw_candidates_include_uncalled_edges_without_labeling_them(
    reverse: bool,
) -> None:
    record = terminal_alignment(1, 20, reverse=reverse)
    raw_length = record.get_tag("ns") + 2
    start, end, boundary = module.head_poly_bounds(record, len(HEAD), raw_length, 960)
    if reverse:
        assert start == 0
        assert end == boundary == 250
    else:
        assert start == boundary == 298
        assert end == raw_length == 540
    with pytest.raises(ValueError, match="exceeds_chunk"):
        module.head_poly_bounds(record, len(HEAD), raw_length, 120)


def test_head_only_without_real_candidate_is_not_zero_filled() -> None:
    record = terminal_alignment(1, 0, suffix="")
    with pytest.raises(ValueError, match="no_real_post_head_signal"):
        module.head_poly_bounds(record, len(HEAD), record.get_tag("ns"), 960)
    with pytest.raises(ValueError, match="below_min_samples"):
        module.head_poly_bounds(record, len(HEAD), record.get_tag("ns") + 2, 960)


@pytest.mark.parametrize("reverse", [False, True])
def test_v4_rejects_head_and_poly_sharing_a_move_block(reverse: bool) -> None:
    record = terminal_alignment(1, 20, reverse=reverse)
    moves = [1, 0] * record.query_length
    index = record.query_length - len(HEAD) - 1 if reverse else len(HEAD) - 1
    moves[index * 2] = 2
    moves[(index + 1) * 2] = 0
    record.set_tag("mv", array("b", [6] + moves))
    with pytest.raises(ValueError, match="head_downstream_share_move_block"):
        module.head_poly_bounds(record, len(HEAD), record.get_tag("ns") + 2, 960)


def test_v4_caps_groups_independently() -> None:
    rows = [
        {
            "read_id": f"synthetic-{length}-{index}",
            "nominal_length_nt": length,
            "poly_type": "polyA",
        }
        for length, count in ((20, 3), (40, 7))
        for index in range(count)
    ]
    selected = module.balance_and_split(
        rows, 6, random.Random(1), balance_to_smallest=False
    )
    assert sum(row["nominal_length_nt"] == 20 for row in selected) == 3
    assert sum(row["nominal_length_nt"] == 40 for row in selected) == 6


def test_exclusions_do_not_fill_quotas_or_request_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        terminal_alignment(index + 1, length)
        for index, length in enumerate([0] * 4 + [20] * 4 + [40] * 4)
    ]
    args, requested = _quota_setup(tmp_path, monkeypatch, records)
    args.structure = "head-poly"
    args.signal_end_policy = "raw-edge"
    excluded = {records[index].query_name for index in (0, 4, 8)}
    path = Path(args.work_root) / "exclusions.txt"
    path.write_text("\n".join(sorted(excluded)) + "\n")
    args.exclude_read_ids = path
    summary = module.prepare(args)
    assert len(requested) == 9
    assert not set(requested) & excluded
    assert summary["metadata_collection"]["final_quotas_met"]
    assert summary["read_exclusion"]["excluded_assigned_ids"] == 3
    assert summary["read_exclusion"]["selected_overlap"] == 0


def test_v4_no_flank_pipeline_masks_all_location_and_status_labels(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        terminal_alignment(index + 1, length)
        for index, length in enumerate([0] * 3 + [20] * 3 + [40] * 4)
    ]
    records += [
        terminal_alignment(101, 20, reverse=True),
        terminal_alignment(102, 0, reverse=True),
        terminal_alignment(103, 0, suffix=""),
    ]
    args, requested = _quota_setup(tmp_path, monkeypatch, records)
    args.structure = "head-poly"
    args.signal_end_policy = "raw-edge"
    args.stop_when_full = False
    args.per_group = 4

    def forbidden(*args: object, **kwargs: object) -> str:
        raise AssertionError("v4 must not infer or search for flank")

    monkeypatch.setattr(module, "infer_flank", forbidden)
    monkeypatch.setattr(module, "flank_boundary", forbidden)
    summary = module.prepare(args)
    assert summary["manifest_version"] == module.HEAD_POLY_VERSION
    assert summary["flank_status"] == "not_applicable_no_flank"
    assert summary["0A_candidate_reads"] == 0
    assert summary["supervision_counts"] == {"length_only": 10}
    assert summary["quota_deficits"] == {"0:none": 1, "20:polyA": 1, "40:polyA": 0}
    assert summary["audit_counts"]["post_head_signal_below_min_samples"] == 1
    assert summary["audit_counts"]["not_selected_zero_orientation"] == 1
    assert records[-3].query_name not in requested
    path = Path(args.work_root) / args.output_dir / "manifest.tsv"
    parsed = load_training_manifest(path)
    assert len(parsed) == 10
    assert all(record.poly_is_complete is None for record in parsed)
    assert all(record.orientation == "forward" for record in parsed)
    dataset = PolyTrainingDataset(
        parsed,
        lambda path, read_id: torch.arange(
            next(r.raw_signal_length for r in parsed if r.read_id == read_id),
            dtype=torch.float32,
        ),
    )
    batch = collate_poly_training_examples([dataset[i] for i in range(len(dataset))])
    assert torch.all(batch.location_targets == -100)
    assert batch.status_targets.tolist() == [-100] * 10
    assert sorted(batch.true_length_nt.tolist()) == [0] * 3 + [20] * 3 + [40] * 4
    assert torch.all(batch.raw_lengths_samples > 0)
    # A v4 row must never acquire a full interval/completeness label by accident.
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    rows[0]["poly_is_complete"] = "true"
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="v4 length_only"):
        load_training_manifest(path)


def test_v4_quota_pause_needs_no_flank_and_resumes_after_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = [
        terminal_alignment(index + 1, length)
        for index, length in enumerate([0] * 3 + [20] * 3 + [40] * 3)
    ]
    replacement = terminal_alignment(101, 20)
    records = prefix + [terminal_alignment(100, 40), prefix[3], replacement]
    args, requested = _quota_setup(tmp_path, monkeypatch, records)
    args.structure = "head-poly"
    args.signal_end_policy = "raw-edge"
    summary = module.prepare(args)
    assert requested == [r.query_name for r in prefix] + [replacement.query_name]
    assert summary["metadata_collection"]["quota_resumptions"] == 1
    assert summary["metadata_collection"]["final_quotas_met"]
    assert summary["metadata_collection"]["records_skipped_after_quota"] == 1
    assert "BAM-order subsets" in summary["limitations"]


def test_v4_requires_explicit_candidate_policy(tmp_path: Path) -> None:
    args = module.build_parser().parse_args(
        [
            "--bonito-bam",
            "unused.bam",
            "--dorado-bam",
            "unused.bam",
            "--pod5-dir",
            ".",
            "--structure",
            "head-poly",
        ]
    )
    with pytest.raises(ValueError, match="explicit"):
        module.prepare(args)


STRICT_GROUPS = {
    (length, kind) for length in (20, 40) for kind in ("polyA", "polyT")
} | {(0, "none:forward"), (0, "none:reverse")}


def balanced_rows(
    scarce: tuple[int, str] | None = None, count: int = 13
) -> list[dict[str, object]]:
    """Make independent synthetic reads, optionally with one scarce stratum."""
    return [
        {
            "read_id": f"synthetic-{length}-{kind}-{index}",
            "nominal_length_nt": length,
            "poly_type": kind if length else "none",
            "orientation": "reverse"
            if kind in {"polyT", "none:reverse"}
            else "forward",
        }
        for length, kind in sorted(STRICT_GROUPS)
        for index in range(count if (length, kind) == scarce else 25)
    ]


@pytest.mark.parametrize("scarce", [None, (40, "polyT"), (0, "none:reverse")])
def test_strict_balance_in_every_split_and_repeatability(
    scarce: tuple[int, str] | None,
) -> None:
    rows = balanced_rows(scarce)
    selected = module.balance_and_split(
        rows,
        20,
        random.Random(20260923),
        balance_to_smallest=False,
        strict_groups=STRICT_GROUPS,
    )
    count = 20 if scarce is None else 13
    validation_count = max(1, count // 10)
    expected_splits = {
        "train": count - validation_count * 2,
        "validation": validation_count,
        "test": validation_count,
    }
    actual = Counter(
        (row["split"], module._sampling_group(row, split_zero_orientation=True))
        for row in selected
    )
    assert actual == {
        (split, group): split_count
        for split, split_count in expected_splits.items()
        for group in STRICT_GROUPS
    }
    assert len({row["read_id"] for row in selected}) == 6 * count
    assert all(
        row["poly_type"] == "none" for row in selected if not row["nominal_length_nt"]
    )
    repeated = module.balance_and_split(
        list(reversed(balanced_rows(scarce))),
        20,
        random.Random(20260923),
        strict_groups=STRICT_GROUPS,
    )
    assert selected == repeated


@pytest.mark.parametrize("count", [0, 1, 2])
@pytest.mark.parametrize("group", [(0, "none:reverse"), (40, "polyT")])
def test_strict_balance_rejects_missing_or_unsplittable_strata(
    count: int, group: tuple[int, str]
) -> None:
    with pytest.raises(ValueError, match=f"at least 3.*{group[1]}"):
        module.balance_and_split(
            balanced_rows(group, count),
            20,
            random.Random(1),
            strict_groups=STRICT_GROUPS,
        )


def test_strict_balance_rejects_duplicate_read_ids() -> None:
    rows = balanced_rows()
    rows.append(rows[0].copy())
    with pytest.raises(ValueError, match="unique read_id"):
        module.balance_and_split(
            rows, 20, random.Random(1), strict_groups=STRICT_GROUPS
        )


def test_strict_quota_requires_both_zero_orientations_and_handles_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prefix = [
        terminal_alignment(
            index * 3 + offset + 1,
            length,
            reverse=kind in {"polyT", "none:reverse"},
        )
        for index, (length, kind) in enumerate(sorted(STRICT_GROUPS))
        for offset in range(3)
    ]
    # The reverse zero stratum is last: enough forward zeros cannot fill it.
    reverse_zero = [r for r in prefix if r.is_reverse and r.reference_id == 0]
    prefix = [r for r in prefix if r not in reverse_zero] + reverse_zero
    skipped = terminal_alignment(100, 40)
    replacement = terminal_alignment(101, 0, reverse=True)
    records = prefix + [skipped, reverse_zero[0], replacement]
    args, requested = _quota_setup(
        tmp_path, monkeypatch, records, poly_types=("polyA", "polyT")
    )
    args.structure = "head-poly"
    args.signal_end_policy = "raw-edge"
    args.strict_balance = True
    summary = module.prepare(args)
    assert requested == [r.query_name for r in prefix] + [replacement.query_name]
    collection = summary["metadata_collection"]
    assert collection["quota_resumptions"] == 1
    assert collection["final_quotas_met"]
    assert collection["records_skipped_after_quota"] == 1
    assert set(collection["target_groups"]) == {
        f"{length}:{kind}" for length, kind in STRICT_GROUPS
    }
    assert summary["balancing"]["effective_per_group"] == 3
    assert all(count == 0 for count in summary["quota_deficits"].values())
    assert len(summary["selected_stratum_counts"]) == 18
    assert set(summary["selected_stratum_counts"].values()) == {1}
    parsed = load_training_manifest(
        Path(args.work_root) / args.output_dir / "manifest.tsv"
    )
    assert len(parsed) == 18
    assert len({r.read_id for r in parsed}) == 18
    assert reverse_zero[0].query_name not in {r.read_id for r in parsed}
    assert replacement.query_name in {r.read_id for r in parsed}
    assert Counter(r.orientation for r in parsed if r.label_status == "zero_nt") == {
        "forward": 3,
        "reverse": 3,
    }
    assert all(r.supervision_mode == "length_only" for r in parsed)


def test_strict_pipeline_reports_common_count_after_shortage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        terminal_alignment(
            index * 5 + offset + 1, length, reverse=kind in {"polyT", "none:reverse"}
        )
        for index, (length, kind) in enumerate(sorted(STRICT_GROUPS))
        for offset in range(3 if kind == "none:reverse" else 5)
    ]
    args, _ = _quota_setup(
        tmp_path, monkeypatch, records, poly_types=("polyA", "polyT")
    )
    args.structure = "head-poly"
    args.signal_end_policy = "raw-edge"
    args.strict_balance = True
    args.per_group = 5
    summary = module.prepare(args)
    assert not summary["metadata_collection"]["final_quotas_met"]
    assert summary["balancing"]["effective_per_group"] == 3
    assert set(summary["quota_deficits"].values()) == {2}
    assert sum(summary["selected_counts"].values()) == 18


def test_strict_pipeline_rejects_missing_zero_orientation_without_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [
        terminal_alignment(
            index * 3 + offset + 1,
            length,
            reverse=kind == "polyT",
        )
        for index, (length, kind) in enumerate(sorted(STRICT_GROUPS))
        if kind != "none:reverse"
        for offset in range(3)
    ]
    args, _ = _quota_setup(
        tmp_path, monkeypatch, records, poly_types=("polyA", "polyT")
    )
    args.structure = "head-poly"
    args.signal_end_policy = "raw-edge"
    args.strict_balance = True
    with pytest.raises(ValueError, match="at least 3.*none:reverse"):
        module.prepare(args)
    output = Path(args.work_root) / args.output_dir
    assert (output / "audit.tsv").is_file()
    assert not (output / "manifest.tsv").exists()


def test_strict_balance_requires_zero_reference_construct(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, requested = _quota_setup(
        tmp_path, monkeypatch, [], poly_types=("polyA", "polyT")
    )
    (Path(args.data_root) / "heads.fa").write_text(f">2_20A\n{HEAD}\n>3_40A\n{HEAD}\n")
    args.structure = "head-poly"
    args.signal_end_policy = "raw-edge"
    args.strict_balance = True
    with pytest.raises(ValueError, match="requires a zero_nt construct"):
        module.prepare(args)
    assert not requested
    assert not (Path(args.work_root) / args.output_dir).exists()


@pytest.mark.parametrize(
    "structure,types", [("head-flank", ["polyA", "polyT"]), ("head-poly", ["polyA"])]
)
def test_strict_balance_requires_v4_and_both_types(
    structure: str, types: list[str]
) -> None:
    args = module.build_parser().parse_args(
        ["--bonito-bam", "unused.bam", "--dorado-bam", "unused.bam", "--pod5-dir", "."]
    )
    args.structure = structure
    args.poly_types = types
    args.strict_balance = True
    with pytest.raises(ValueError, match="requires head-poly and both"):
        module.prepare(args)
