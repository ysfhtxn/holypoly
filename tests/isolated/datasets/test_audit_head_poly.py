"""Check deterministic sampling and calibrated signal audits with synthetic data."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pod5
import pytest

from holypoly.datasets.audit_head_poly import audit, sample_manifest, waveform_preview


def test_sampling_is_reproducible_and_per_length(tmp_path: Path) -> None:
    path = tmp_path / "manifest.tsv"
    rows = [
        {"read_id": str(index), "nominal_length_nt": str(length)}
        for length in (0, 20, 150)
        for index in range(length + 1, length + 11)
    ]
    with path.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    sampled, counts = sample_manifest(path, 3, 19)
    assert (sampled, counts) == sample_manifest(path, 3, 19)
    assert len(sampled) == 9
    assert counts == {0: 10, 20: 10, 150: 10}
    assert len({row["read_id"] for row in sampled}) == 9
    with pytest.raises(ValueError, match="positive"):
        sample_manifest(path, 0, 19)


def test_preview_preserves_extrema_and_raw_coordinates() -> None:
    preview = waveform_preview(np.array([1.0, -10.0, 20.0, 2.0]), bins=2)
    assert preview == {"sample": [1, 3], "min_pa": [-10.0, 2.0], "max_pa": [1.0, 20.0]}


def test_audit_reads_only_selected_ids_and_dry_run_does_not_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root, work_root = tmp_path / "data", tmp_path / "work"
    data_root.mkdir()
    directory = work_root / "metadata" / "synthetic"
    directory.mkdir(parents=True)
    pod5_path = data_root / "synthetic.pod5"
    pod5_path.touch()
    config = tmp_path / "config.yaml"
    config.write_text(f"paths:\n  data_root: {data_root}\n  work_root: {work_root}\n")
    rows = []
    for reverse in (False, True):
        rows.append(
            {
                "read_id": "reverse" if reverse else "forward",
                "nominal_length_nt": "20",
                "pod5_path": str(pod5_path),
                "raw_signal_length": "100",
                "orientation": "reverse" if reverse else "forward",
                "head_boundary_sample": "60" if reverse else "40",
                "chunk_start_sample": "10" if reverse else "40",
                "chunk_end_sample": "60" if reverse else "90",
                "model_input_samples": "48",
                "supervision_mode": "length_only",
            }
        )
    manifest = directory / "manifest.tsv"
    with manifest.open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    requested = []

    class Reader:
        def __init__(self, path: Path) -> None:
            assert path == pod5_path

        def __enter__(self) -> Reader:
            return self

        def __exit__(self, *args: object) -> None:
            pass

        def reads(
            self, *, selection: list[str], missing_ok: bool
        ) -> list[SimpleNamespace]:
            assert missing_ok
            requested.extend(selection)
            return [
                SimpleNamespace(
                    signal=np.arange(100, dtype=np.int16),
                    calibration=SimpleNamespace(scale=2.0, offset=-5.0),
                )
            ]

    monkeypatch.setattr(pod5, "Reader", Reader)
    args = argparse.Namespace(
        config=config,
        manifests=[manifest],
        per_group=20,
        seed=1,
        output_dir=Path("results/audit"),
        dry_run=True,
    )
    assert audit(args)["selected_reads"] == 2
    assert requested == []
    assert not (work_root / args.output_dir).exists()
    args.dry_run = False
    summary = audit(args)
    assert summary["successful_reads"] == 2
    assert sorted(requested) == ["forward", "reverse"]
    samples = json.loads((work_root / args.output_dir / "samples.json").read_text())
    assert all(s["samples_outside_old_chunk"] == 10 for s in samples)
    assert all(s["exceeds_old_input_limit"] for s in samples)
    assert all(s["waveform"]["min_pa"][0] == -10 for s in samples)
    with pytest.raises(FileExistsError):
        audit(args)
