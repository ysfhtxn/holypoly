"""Unlabeled BAM candidates and unchanged-network inference on synthetic signals."""

from __future__ import annotations

import argparse
import array
from pathlib import Path
from typing import Any

import pysam
import pytest
import torch

from holypoly.inference.predict import (
    build_parser,
    candidate_rows,
    locate_signals,
    predict_candidates,
    run,
    summarize,
)
from holypoly.models.model import HolypolyModel
from tests.isolated.helpers import tiny_v5_basecaller


def _bam(path: Path) -> None:
    """Write unlabeled synthetic records with primary/secondary and reverse cases."""
    header = {"HD": {"VN": "1.6"}, "SQ": []}
    sequences = [
        ("forward", "CG" * 10 + "A" * 12, 4, True),
        ("reverse", "T" * 12 + "CG" * 10, 20, True),
        ("absent", "CG" * 20, 4, True),
        ("missing-moves", "CG" * 10 + "A" * 12, 4, False),
        ("ambiguous", "T" * 12 + "CG" * 10 + "A" * 12, 4, True),
        ("t-direction", "CG" * 10 + "T" * 12, 4, True),
        ("secondary", "CG" * 10 + "A" * 12, 260, True),
    ]
    with pysam.AlignmentFile(str(path), "wb", header=header) as writer:
        for name, sequence, flag, with_moves in sequences:
            record = pysam.AlignedSegment(writer.header)
            record.query_name = name
            record.query_sequence = sequence
            record.flag = flag
            if with_moves:
                record.set_tag("mv", array.array("b", [6] + [1] * len(sequence)))
                record.set_tag("ts", 0)
                record.set_tag("ns", 6 * len(sequence))
            writer.write(record)


def _args() -> argparse.Namespace:
    """Use a chunk that fits test candidates, without supplying any truth labels."""
    return build_parser().parse_args(
        [
            "--pod5-dir",
            "unused",
            "--bam",
            "unused.bam",
            "--checkpoint",
            "unused.pt",
            "--output-dir",
            "unused",
            "--chunk-size-samples",
            "120",
        ]
    )


def test_bam_candidates_keep_absent_ambiguous_reverse_and_mapping_failures(
    tmp_path: Path,
) -> None:
    """Reference-oriented reverse SEQ must be restored before raw move mapping."""
    path = tmp_path / "synthetic.bam"
    _bam(path)
    args = _args()
    rows = candidate_rows(path, args)
    assert len(rows) == 6
    assert [row["status"] for row in rows] == [
        "pending",
        "pending",
        "no_candidate",
        "candidate_mapping_failure",
        "ambiguous_candidate",
        "type_not_selected",
    ]
    for row in rows[:2]:
        assert row["poly_type"] == "polyA"
        assert (row["start_base"], row["end_base"]) == (20, 32)
        assert (row["start_sample"], row["end_sample"]) == (120, 192)
    assert all(row["predicted_length_nt"] == "" for row in rows)
    assert not any("true_length" in key or "nominal_length" in key for key in rows[0])
    args.max_reads = 1
    assert len(candidate_rows(path, args)) == 1
    args.chunk_size_samples = 60
    rows = candidate_rows(path, args)
    assert rows[0]["status"] == "candidate_mapping_failure"
    assert "not_clipped" in rows[0]["failure_reason"]


def test_inference_preserves_raw_lengths_masks_and_model_parameters(
    tmp_path: Path,
) -> None:
    """Padding invariance, absent/missing rows, and zero-status ungated regression."""
    path = tmp_path / "synthetic.bam"
    _bam(path)
    rows = candidate_rows(path, _args())
    rows[0]["pod5_path"] = "synthetic.pod5"
    rows[1]["pod5_path"] = "synthetic.pod5"
    torch.manual_seed(7)
    model = HolypolyModel(
        tiny_v5_basecaller(), hidden_features=16, length_head_mode="sum"
    )
    with torch.no_grad():
        for head in (model.location_head, model.length_head):
            head.status_classifier.weight.zero_()
            head.status_classifier.bias.copy_(torch.tensor([0.0, 20.0, -20.0]))
    before = {key: value.clone() for key, value in model.state_dict().items()}
    seen = []

    def reader(path: Path, read_id: str) -> torch.Tensor:
        seen.append(read_id)
        if read_id == "reverse":
            raise KeyError("synthetic missing signal")
        return torch.linspace(-2, 2, 192)

    predict_candidates(
        model, rows, reader, device=torch.device("cpu"), chunk_size_samples=120
    )
    assert seen == ["forward", "reverse"]
    assert rows[0]["status"] == "candidate_predicted"
    assert rows[0]["predicted_length_nt"] > 0
    assert rows[0]["model_status"] == "zero_nt"
    assert rows[1]["status"] == "input_failure"
    assert rows[1]["predicted_length_nt"] == rows[2]["predicted_length_nt"] == ""
    assert all(
        torch.equal(value, model.state_dict()[key]) for key, value in before.items()
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    summary = summarize(rows)
    assert summary["selected_primary_bam_reads"] == 6
    assert summary["distributions_by_type"]["polyA"]["reads"] == 1
    assert "mae_nt" not in summary


def test_locating_signal_reports_missing_ids_without_reading_samples(
    monkeypatch: Any,
) -> None:
    """Metadata mapping cannot silently drop unavailable reads or load signal arrays."""

    class FakeReader:
        read_ids = ["found"]

        def __init__(self, path: Path):
            pass

        def __enter__(self) -> Any:
            return self

        def __exit__(self, *args: Any) -> None:
            pass

    monkeypatch.setattr("holypoly.inference.predict.Reader", FakeReader)
    rows = [{"read_id": name, "status": "pending"} for name in ("found", "missing")]
    locate_signals(rows, [Path("synthetic.pod5")])
    assert rows[0]["pod5_path"] == "synthetic.pod5"
    assert rows[1]["status"] == "input_failure"
    with pytest.raises(ValueError, match="multiple POD5"):
        locate_signals(
            [{"read_id": "found", "status": "pending"}],
            [Path("one.pod5"), Path("two.pod5")],
        )


def test_dry_run_does_not_load_bam_signals_weights_or_create_output(
    tmp_path: Path, monkeypatch: Any
) -> None:
    """Preflight enforces new external outputs without loading real inputs."""
    data, work = tmp_path / "raw", tmp_path / "work"
    data.mkdir()
    work.mkdir()
    pod5 = data / "pod5"
    pod5.mkdir()
    (pod5 / "synthetic.pod5").touch()
    (work / "synthetic.bam").touch()
    (work / "weights.pt").touch()
    model = work / "model"
    model.mkdir()
    (model / "config.toml").touch()
    config = tmp_path / "config.yaml"
    config.write_text(f"paths:\n  data_root: {data}\n  work_root: {work}\n")
    args = build_parser().parse_args(
        [
            "--config",
            str(config),
            "--pod5-dir",
            "pod5",
            "--bam",
            "synthetic.bam",
            "--checkpoint",
            "weights.pt",
            "--model-dir",
            "model",
            "--output-dir",
            "out",
            "--dry-run",
        ]
    )

    def forbidden(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("dry-run attempted to load real inputs")

    monkeypatch.setattr("holypoly.inference.predict.candidate_rows", forbidden)
    monkeypatch.setattr(torch, "load", forbidden)
    summary = run(args)
    assert summary["pod5_files"] == 1
    assert not (work / "out").exists()
    args.output_dir = data / "unsafe"
    with pytest.raises(ValueError, match="output must"):
        run(args)
