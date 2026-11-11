"""Synthetic checks for held-out evaluation; no real POD5 or GPU required."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from holypoly.models.model import HolypolyModel
from holypoly.training import evaluate
from holypoly.training.data import (
    ManifestLabelStatus,
    ManifestPolyType,
    ManifestSplit,
    ManifestSupervisionMode,
    PolyTrainingDataset,
    PolyTrainingRecord,
)
from holypoly.training.evaluate import (
    _ranks,
    _validate_read_ids,
    agreement_metrics,
    main,
    predict_reads,
    status_agreement_metrics,
)
from tests.isolated.helpers import tiny_v5_basecaller
from tests.isolated.training.test_command import FIELDNAMES, _row


def _record(read_id: str) -> PolyTrainingRecord:
    """Create a complete synthetic read with an absolute raw-sample offset."""
    return PolyTrainingRecord(
        read_id=read_id,
        pod5_path=Path("synthetic.pod5"),
        pod5_file_id="synthetic",
        split=ManifestSplit.TEST,
        orientation="forward",
        raw_signal_length=240,
        chunk_start_sample=60,
        chunk_end_sample=180,
        poly_start_sample=84,
        poly_end_sample=144,
        poly_type=ManifestPolyType.POLY_A,
        true_length_nt=10,
        nominal_length_nt=10,
        label_status=ManifestLabelStatus.POLY_PRESENT,
        poly_is_complete=True,
        failure_reason="",
    )


def test_timing_evaluation_maps_references_and_retains_failures() -> None:
    """Available and unavailable reads share batches without mixing references."""
    records = [_record(name) for name in ("a", "t", "unreadable")]

    def reader(path: Path, read_id: str) -> torch.Tensor:
        if read_id == "unreadable":
            raise KeyError("synthetic missing signal")
        return torch.linspace(-2, 2, 240)

    model = HolypolyModel(
        tiny_v5_basecaller(),
        hidden_features=6,
        length_head_mode="sum",
        use_timing_reference=True,
    ).eval()
    with torch.no_grad():
        model.length_head.timing_residual[-1].bias.fill_(3)
    dataset = PolyTrainingDataset(records, reader)
    references = {
        "a": (math.log1p(12), 0.1, 1),
        "t": (0, 0, 0),
        "unreadable": (0, 0, 0),
    }
    plain = predict_reads(model, dataset, device=torch.device("cpu"), batch_size=2)
    timed = predict_reads(
        model,
        dataset,
        device=torch.device("cpu"),
        batch_size=2,
        timing_references=references,
    )
    single = predict_reads(
        model,
        dataset,
        device=torch.device("cpu"),
        batch_size=1,
        timing_references=references,
    )
    assert [row["read_id"] for row in timed] == [row.read_id for row in records]
    assert timed[0]["regression_length_nt"] > plain[0]["regression_length_nt"]
    assert timed[1]["regression_length_nt"] == plain[1]["regression_length_nt"]
    assert timed[2] == plain[2] == single[2]
    assert timed[2]["prediction_status"] == "input_failure"
    for row, expected in zip(timed[:2], single[:2], strict=True):
        assert row["regression_length_nt"] == pytest.approx(
            expected["regression_length_nt"]
        )
        assert row["shared_poly_present_probability"] == pytest.approx(
            plain[timed.index(row)]["shared_poly_present_probability"]
        )


@pytest.mark.parametrize("head_free", [False, True])
def test_predict_preserves_failures_and_never_updates_parameters(
    head_free: bool,
) -> None:
    torch.manual_seed(7)
    records = [
        _record("a"),
        replace(_record("t"), orientation="reverse", poly_type=ManifestPolyType.POLY_T),
        replace(
            _record("weak"),
            supervision_mode=ManifestSupervisionMode.LENGTH_ONLY,
            poly_start_sample=None,
            poly_end_sample=None,
            poly_is_complete=None,
        ),
        _record("unreadable"),
    ]
    if head_free:
        records = [
            replace(
                record,
                manifest_version="construct-nominal-head-excluded-v3",
                chunk_end_sample=157,
                model_input_samples=120,
                head_excluded=True,
                head_boundary_sample=180 if record.orientation == "reverse" else 60,
            )
            for record in records
        ]

    def reader(path: Path, read_id: str) -> torch.Tensor:
        if read_id == "unreadable":
            raise KeyError("synthetic missing signal")
        return torch.linspace(-2, 2, 240)

    dataset = PolyTrainingDataset(records, reader)
    model = HolypolyModel(tiny_v5_basecaller(), hidden_features=16, dropout=0.3)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rows = predict_reads(model, dataset, device=torch.device("cpu"), batch_size=2)
    repeated = predict_reads(model, dataset, device=torch.device("cpu"), batch_size=2)
    assert rows == repeated
    assert [row["read_id"] for row in rows] == [record.read_id for record in records]
    assert all(row["prediction_status"] == "ok" for row in rows[:3])
    assert rows[3]["prediction_status"] == "input_failure"
    assert "KeyError" in rows[3]["failure_reason"]
    assert rows[3]["regression_length_nt"] is None
    assert not model.training
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        torch.equal(value, before[key]) for key, value in model.state_dict().items()
    )
    for record, row in zip(records[:3], rows[:3], strict=True):
        assert sum(
            row[key]
            for key in (
                "shared_poly_present_probability",
                "shared_zero_nt_probability",
                "shared_failure_probability",
            )
        ) == pytest.approx(1)
        assert row["regression_length_nt"] >= 0
        assert row["model_input_samples"] == 120
        assert row["head_excluded"] == head_free
        assert row["head_boundary_sample"] == record.head_boundary_sample
        if row["predicted_start_sample"] is not None:
            assert (
                record.chunk_start_sample
                <= row["predicted_start_sample"]
                < row["predicted_end_sample"]
                <= record.chunk_end_sample
            )


def _metric_row(target: float, prediction: float | None, **overrides: object) -> dict:
    """Make one labeled prediction for testing metric denominators."""
    return {
        "training_length_nt": target,
        "regression_length_nt": prediction,
        "prediction_status": "ok",
        "label_status": "poly_present",
        "shared_status": "poly_present",
        **overrides,
    }


def test_status_metrics_keep_failures_separate_and_do_not_gate_regression() -> None:
    rows = [
        _metric_row(0, 40, label_status="zero_nt", shared_status="zero_nt"),
        _metric_row(0, 50, label_status="zero_nt", shared_status="poly_present"),
        _metric_row(0, 60, label_status="zero_nt", shared_status="failure"),
        _metric_row(20, 20),
        _metric_row(20, 20, shared_status="zero_nt"),
        _metric_row(20, 20, shared_status="failure"),
        _metric_row(20, None, prediction_status="input_failure", shared_status=""),
    ]
    metrics = status_agreement_metrics(rows)
    assert metrics["zero_nt_false_positive_fraction"] == pytest.approx(1 / 3)
    assert metrics["poly_present_false_zero_fraction"] == pytest.approx(1 / 3)
    assert metrics["poly_present_recall"] == pytest.approx(1 / 3)
    assert metrics["zero_nt_predicted_failure_fraction"] == pytest.approx(1 / 3)
    assert metrics["accuracy"] == pytest.approx(2 / 6)
    assert metrics["input_or_prediction_failures"] == 1
    assert metrics["confusion_true_rows_predicted_columns"]["failure"] == {
        "poly_present": 0,
        "zero_nt": 0,
        "failure": 0,
    }
    assert agreement_metrics(rows)["mae_nt"] == 25
    assert status_agreement_metrics([])["accuracy"] is None
    assert (
        status_agreement_metrics([_metric_row(20, 20)])[
            "zero_nt_false_positive_fraction"
        ]
        is None
    )


def test_agreement_metrics_include_zero_and_exclude_failed_or_unknown_targets() -> None:
    rows = [
        _metric_row(10, 8),
        _metric_row(20, 24),
        _metric_row(0, 1, label_status="zero_nt"),
        _metric_row(math.nan, 999, label_status="failure"),
        _metric_row(30, None, prediction_status="input_failure", shared_status=""),
    ]
    metrics = agreement_metrics(rows)
    assert metrics["evaluable_reads"] == 3
    assert metrics["prediction_failures"] == 1
    assert metrics["mae_nt"] == pytest.approx(7 / 3)
    assert metrics["median_ae_nt"] == 2
    assert metrics["rmse_nt"] == pytest.approx(math.sqrt(7))
    assert metrics["bias_nt"] == 1
    assert metrics["within_1nt_fraction"] == pytest.approx(1 / 3)
    assert metrics["within_5nt_fraction"] == 1
    assert metrics["spearman"] == pytest.approx(1)
    assert agreement_metrics([rows[0]])["pearson"] is None
    assert agreement_metrics([rows[-1]])["mae_nt"] is None
    assert _ranks([5, 5, 9]) == [0.5, 0.5, 2]


def test_rejects_shared_uuid_even_with_different_chunks() -> None:
    record = _record("same")
    with pytest.raises(ValueError, match="more than once"):
        _validate_read_ids(
            [
                record,
                replace(
                    record,
                    split=ManifestSplit.TRAIN,
                    chunk_start_sample=0,
                    chunk_end_sample=120,
                ),
            ]
        )


def test_evaluation_dry_run_loads_no_checkpoint_or_signal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in (
        "HOLYPOLY_DATA_ROOT",
        "HOLYPOLY_WORK_ROOT",
        "HOLYPOLY_M3_MANIFEST",
        "HOLYPOLY_M3_CHECKPOINT",
        "HOLYPOLY_BONITO_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)
    data_root, work_root = tmp_path / "data", tmp_path / "work"
    data_root.mkdir()
    model_dir = work_root / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text("[standardisation]\nmean=0\nstdev=1\n")
    checkpoint = work_root / "checkpoint.pt"
    checkpoint.write_text("not a torch checkpoint")
    manifest = work_root / "manifest.tsv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(
            [
                _row(
                    f"test-{index}",
                    data_root / "nonexistent.pod5",
                    split="test",
                    pod5_file_id="synthetic",
                    poly_type="polyA",
                    length_nt=10,
                )
                for index in range(3)
            ]
        )
    args = [
        "--data-root",
        str(data_root),
        "--work-root",
        str(work_root),
        "--manifest",
        str(manifest),
        "--model-dir",
        str(model_dir),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        "results/eval",
        "--max-reads",
        "2",
        "--dry-run",
    ]
    assert main(args) == 0
    summary = json.loads(capsys.readouterr().out.splitlines()[0])
    assert summary["partial_evaluation"] and summary["selected_reads"] == 2
    assert not (work_root / "results").exists()
    assert checkpoint.read_text() == "not a torch checkpoint"
    args[args.index("results/eval")] = str(data_root / "unsafe-output")
    assert main(args) == 2
    assert "outside raw data" in capsys.readouterr().err


@pytest.mark.parametrize("mode", ["mean", "sum", "duration_only", "mean_duration"])
def test_evaluation_exports_test_only_and_restores_checkpoint_hidden_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    """Exercise checkpoint loading and all outputs using CPU synthetic inputs."""
    for name in (
        "HOLYPOLY_DATA_ROOT",
        "HOLYPOLY_WORK_ROOT",
        "HOLYPOLY_M3_MANIFEST",
        "HOLYPOLY_M3_CHECKPOINT",
        "HOLYPOLY_BONITO_MODEL",
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "CUDA_CACHE_PATH",
    ):
        monkeypatch.delenv(name, raising=False)
    data_root, work_root = tmp_path / "data", tmp_path / "work"
    data_root.mkdir()
    model_dir = work_root / "model"
    model_dir.mkdir(parents=True)
    (model_dir / "config.toml").write_text("[standardisation]\nmean=0\nstdev=1\n")
    model = HolypolyModel(
        tiny_v5_basecaller(), hidden_features=16, length_head_mode=mode
    )
    checkpoint = work_root / "checkpoint.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "random_seed": 7,
            "model_config": {"standardisation": {"mean": 0, "stdev": 1}},
            "manifest_schema_version": "train-command-test-v1",
            "epoch": 3,
            "global_step": 9,
            "git_commit": "synthetic",
            "bonito_commit": "synthetic",
            "loss_config": {},
            **({"head_config": model.head_config} if mode != "mean" else {}),
        },
        checkpoint,
    )
    before = checkpoint.read_bytes()
    manifest = work_root / "manifest.tsv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES, delimiter="\t")
        writer.writeheader()
        writer.writerows(
            [
                _row(
                    read_id,
                    data_root / "synthetic.pod5",
                    split=split,
                    pod5_file_id="synthetic",
                    poly_type="polyA",
                    length_nt=10,
                )
                for read_id, split in (
                    ("training", "train"),
                    ("validation", "validation"),
                    ("test-ok", "test"),
                    ("test-bad", "test"),
                )
            ]
        )

    accessed = []

    def reader(path: Path, read_id: str) -> torch.Tensor:
        accessed.append(read_id)
        if read_id == "test-bad":
            raise ValueError("synthetic unreadable signal")
        return torch.linspace(-2, 2, 240)

    monkeypatch.setattr(evaluate, "_evaluation_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        evaluate, "load_bonito_v5_model", lambda *args, **kwargs: tiny_v5_basecaller()
    )
    monkeypatch.setattr(evaluate, "Pod5SignalReader", lambda **kwargs: reader)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda index: "synthetic CPU fixture"
    )
    assert (
        main(
            [
                "--data-root",
                str(data_root),
                "--work-root",
                str(work_root),
                "--manifest",
                str(manifest),
                "--model-dir",
                str(model_dir),
                "--checkpoint",
                str(checkpoint),
                "--output-dir",
                "results/test",
                "--batch-size",
                "2",
            ]
        )
        == 0
    )
    output = work_root / "results" / "test"
    with (output / "predictions.tsv").open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    summary = json.loads((output / "summary.json").read_text())
    assert accessed == ["test-ok", "test-bad"]
    assert len(rows) == 2 and rows[1]["prediction_status"] == "input_failure"
    assert rows[1]["regression_length_nt"] == ""
    assert summary["epoch"] == 3 and summary["metrics"]["evaluable_reads"] == 1
    assert summary["metrics"]["prediction_failures"] == 1
    assert summary["head_config"]["length_head_mode"] == mode
    assert (output / "metrics_by_group.tsv").is_file()
    assert checkpoint.read_bytes() == before
