"""Small-manifest selection uses synthetic metadata only, without Torch or POD5."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import pytest

from holypoly.datasets.subset_manifest import HEAD_POLY_VERSION, main, select_subset


def _rows() -> list[dict[str, str]]:
    """Include zero, multiple positive lengths, and both positive orientations."""
    rows = []
    for length, kind in ((0, "none"), (20, "polyA"), (20, "polyT"), (150, "polyA")):
        for split, count in (("train", 40), ("validation", 5), ("test", 5)):
            for index in range(count):
                rows.append(
                    {
                        "read_id": f"synthetic-{length}-{kind}-{split}-{index}",
                        "split": split,
                        "nominal_length_nt": str(length),
                        "poly_type": kind,
                        "manifest_version": HEAD_POLY_VERSION,
                        "supervision_mode": "length_only",
                        "training_length_nt": str(length),
                        "label_source": "synthetic_construct_weak_label",
                        "chunk_start_sample": "12",
                        "chunk_end_sample": "120",
                        "extra_audit_field": "unchanged",
                    }
                )
    return rows


def test_balanced_subset_preserves_rows_splits_and_reproducible_selection() -> None:
    rows = _rows()
    before = [dict(row) for row in rows]
    selected = select_subset(rows, per_group=20, seed=17)
    assert len(selected) == 80
    assert Counter(row["split"] for row in selected) == {
        "train": 64,
        "validation": 8,
        "test": 8,
    }
    strata = Counter(
        (row["nominal_length_nt"], row["poly_type"], row["split"]) for row in selected
    )
    assert all(
        count == (16 if split == "train" else 2)
        for (_, _, split), count in strata.items()
    )
    assert len({row["read_id"] for row in selected}) == len(selected)
    original = {row["read_id"]: row for row in before}
    assert all(row == original[row["read_id"]] for row in selected)
    assert rows == before
    assert selected == select_subset(list(reversed(rows)), per_group=20, seed=17)
    assert selected != select_subset(rows, per_group=20, seed=18)


@pytest.mark.parametrize(
    "case", ["duplicate", "too_few", "wrong_version", "bad_split", "empty"]
)
def test_invalid_source_is_rejected(case: str) -> None:
    rows = _rows()
    if case == "duplicate":
        rows.append({**rows[0], "split": "test"})
    elif case == "too_few":
        rows = [row for row in rows if row["split"] != "test"]
    elif case == "wrong_version":
        rows[0]["manifest_version"] = "legacy"
    elif case == "bad_split":
        rows[0]["split"] = "unknown"
    else:
        rows = []
    with pytest.raises(ValueError):
        select_subset(rows, per_group=20, seed=17)


def test_cli_dry_run_new_outputs_and_existing_source_protection(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    data, work = tmp_path / "raw", tmp_path / "work"
    data.mkdir()
    work.mkdir()
    source = work / "source.csv"
    rows = _rows()
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    before = source.read_bytes()
    arguments = [
        "--data-root",
        str(data),
        "--work-root",
        str(work),
        "--manifest",
        "source.csv",
        "--output-dir",
        "metadata/small",
        "--per-group",
        "20",
        "--seed",
        "17",
    ]
    assert main([*arguments, "--dry-run"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["selected_rows"] == 80
    assert not (work / "metadata").exists()
    assert main(arguments) == 0
    capsys.readouterr()
    output = work / "metadata" / "small"
    with (output / "manifest.tsv").open() as stream:
        subset = list(csv.DictReader(stream, delimiter="\t"))
    assert subset == select_subset(rows, per_group=20, seed=17)
    assert (output / "summary.json").is_file()
    assert source.read_bytes() == before
    saved = (output / "manifest.tsv").read_bytes()
    assert main(arguments) == 2
    assert "choose a new output" in capsys.readouterr().err
    assert (output / "manifest.tsv").read_bytes() == saved
    arguments[arguments.index("metadata/small")] = str(data / "unsafe")
    assert main(arguments) == 2
    assert not (data / "unsafe").exists()
    assert source.read_bytes() == before
