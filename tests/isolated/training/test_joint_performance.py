"""Synthetic regressions for Poly-only forward, monitoring and progress."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import asdict

import pytest
import torch

from holypoly.models.joint_model import JointHolypolyModel, create_joint_optimizer
from holypoly.training import engine, joint_engine, progress
from holypoly.training.data import collate_poly_training_examples
from holypoly.training.losses import PolyAuxiliaryLoss, PolyLossWeights
from tests.isolated.training.test_reference_joint import (
    joint_basecaller,
    synthetic_sequence,
    weak_status_examples,
)


@pytest.mark.parametrize("head_type", ["legacy", "tcn"])
@pytest.mark.parametrize("unfreeze", [0, 1])
def test_poly_forward_preserves_predictions_gradients_and_full_crf(
    head_type: str,
    unfreeze: int,
) -> None:
    """Omitting CRF scores changes neither auxiliary values nor their gradients."""
    torch.manual_seed(17)
    model = JointHolypolyModel(
        joint_basecaller(),
        head_type=head_type,
        length_head_mode="sum",
        unfreeze_last_layers=unfreeze,
    ).train()
    batch = collate_poly_training_examples(
        weak_status_examples(), supervise_status=True
    )
    criterion = PolyAuxiliaryLoss(
        PolyLossWeights(lambda_location=0, length_loss="huber")
    )
    calls = []
    hook = model.basecaller.encoder.crf.register_forward_hook(
        lambda *_: calls.append("crf")
    )
    gradients = []
    outputs = []
    before = set(model.state_dict())
    for forward in (model, model.forward_poly):
        torch.manual_seed(23)  # Match TCN dropout in both forward paths.
        model.zero_grad(set_to_none=True)
        output = forward(batch.signal, batch.raw_padding_mask)
        criterion(
            output, batch.location_targets, batch.true_length_nt, batch.status_targets
        )["total_loss"].backward()
        gradients.append(
            {
                name: parameter.grad.clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
        )
        outputs.append(output)
    hook.remove()
    assert calls == ["crf"] and set(model.state_dict()) == before
    assert outputs[0].basecall_scores.shape[:2] == (22, 2)
    assert not hasattr(outputs[1], "basecall_scores")
    for field in ("length_nt", "status_logits"):
        torch.testing.assert_close(
            getattr(outputs[0].length, field),
            getattr(outputs[1].length, field),
            rtol=0,
            atol=0,
        )
    torch.testing.assert_close(
        outputs[0].location.status_logits,
        outputs[1].location.status_logits,
        rtol=0,
        atol=0,
    )
    assert gradients[0].keys() == gradients[1].keys()
    for name in gradients[0]:
        torch.testing.assert_close(
            gradients[0][name], gradients[1][name], rtol=0, atol=0
        )


class ForbiddenSequences:
    """Fail if an optimization even creates an iterator for the skipped stream."""

    def __iter__(self):
        raise AssertionError("frozen monitoring must not read sequence signals")


@pytest.mark.parametrize("prefetch", [0, 2])
def test_frozen_training_skips_sequence_io_without_changing_head_updates(
    monkeypatch,
    prefetch: int,
) -> None:
    """The removed monitoring computation has no influence on Poly gradients."""
    monkeypatch.setattr(joint_engine, "require_single_cuda_device", lambda _: None)
    torch.manual_seed(11)
    initial = JointHolypolyModel(
        joint_basecaller(),
        length_head_mode="sum",
        unfreeze_last_layers=0,
    )
    batch = collate_poly_training_examples(
        weak_status_examples(), supervise_status=True
    )
    results = []
    states = []
    for monitoring in (True, False):
        model = deepcopy(initial)
        monkeypatch.setattr(
            model.basecaller, "loss", lambda scores, *_: scores.square().mean()
        )
        optimizer = create_joint_optimizer(
            model,
            head_learning_rate=0.001,
            encoder_learning_rate=1e-5,
            weight_decay=0,
        )
        metrics, sequence = joint_engine.train_joint_epoch(
            model,
            [batch] * 2,
            [[synthetic_sequence()]] if monitoring else ForbiddenSequences(),
            PolyAuxiliaryLoss(PolyLossWeights(lambda_location=0, length_loss="huber")),
            optimizer,
            lambda_basecall=0.1,
            device=torch.device("cpu"),
            gradient_clip_norm=2,
            global_step=0,
            prefetch_batches=prefetch,
            monitor_frozen_sequences=monitoring,
        )
        results.append(asdict(metrics))
        states.append(deepcopy(model.state_dict()))
        if not monitoring:
            assert sequence == {
                "sequence_examples": 0,
                "sequence_unique_reads": 0,
                "sequence_metrics_source": "disabled_frozen",
            }
            row = joint_engine.joint_log_record(
                metrics,
                sequence,
                epoch=1,
                split="train",
                lambda_basecall=0.1,
                loss_config={},
                joint_config={},
            )
            assert "basecalling_loss" not in row["losses"]
            assert row["losses"]["total_loss"] == metrics.losses["total_loss"]
    assert results[0] == results[1]
    for name in states[0]:
        torch.testing.assert_close(states[0][name], states[1][name], rtol=0, atol=0)


def test_validation_progress_and_frozen_baseline_reuse(monkeypatch, capsys) -> None:
    """Show both phases and ETA totals; cached monitoring never opens signals."""
    monkeypatch.setattr(joint_engine, "require_single_cuda_device", lambda _: None)
    monkeypatch.setattr(engine, "require_single_cuda_device", lambda _: None)
    model = JointHolypolyModel(joint_basecaller(), unfreeze_last_layers=0)
    monkeypatch.setattr(
        model.basecaller, "loss", lambda scores, *_: scores.square().mean()
    )
    monkeypatch.setattr(model.basecaller, "decode_batch", lambda _: ["ACGTACGT"])
    batch = collate_poly_training_examples(
        weak_status_examples(), supervise_status=True
    )
    criterion = PolyAuxiliaryLoss(
        PolyLossWeights(lambda_location=0, length_loss="huber")
    )
    timings = {}
    metrics, baseline = joint_engine.validate_joint_epoch(
        model,
        [batch],
        [[synthetic_sequence()]],
        criterion,
        device=torch.device("cpu"),
        global_step=0,
        progress_desc="Epoch 1 val",
        sequence_examples=1,
        timings=timings,
    )
    displayed = capsys.readouterr()
    assert "Epoch 1 val length/status" in displayed.err
    assert "Epoch 1 val head sequence" in displayed.err
    assert "1/1" in displayed.err and "finished in" in displayed.err
    assert len(timings) == 2
    cached_metrics, cached = joint_engine.validate_joint_epoch(
        model,
        [batch],
        ForbiddenSequences(),
        criterion,
        device=torch.device("cpu"),
        global_step=0,
        cached_sequence=baseline,
    )
    assert asdict(metrics) == asdict(cached_metrics)
    assert cached == {**baseline, "sequence_metrics_source": "frozen_baseline"}
    # Never reuse a frozen baseline for an encoder that can update.
    model.basecaller.encoder.transformer_encoder[-1].requires_grad_(True)
    with pytest.raises(ValueError, match="fully frozen"):
        joint_engine.validate_joint_epoch(
            model,
            [batch],
            ForbiddenSequences(),
            criterion,
            device=torch.device("cpu"),
            global_step=0,
            cached_sequence=baseline,
        )


def test_bounded_profiling_and_disabled_path(monkeypatch) -> None:
    """No synchronization in normal batches, warmup or after the timing window."""
    calls = []
    monkeypatch.setattr(progress, "_synchronize", lambda _: calls.append("sync"))
    timer = progress.BatchProfiler(torch.device("cpu"), batches=2, warmup_batches=1)
    for _ in range(5):
        timer.start_batch()
        with timer.measure("encoder"):
            pass
    assert len(calls) == 4
    assert timer.report()["measured_batches"] == 2
    calls.clear()
    disabled = progress.BatchProfiler(torch.device("cpu"))
    for _ in range(7):
        disabled.start_batch()
        with disabled.measure("encoder"):
            pass
    assert not calls and not disabled.seconds


@pytest.mark.parametrize("batches,measured", [(4, 0), (6, 1), (8, 2)])
def test_profile_reports_actual_batches_including_short_epochs(
    monkeypatch,
    batches: int,
    measured: int,
) -> None:
    """Exhausting a loader does not add a spurious profiled batch."""
    monkeypatch.setattr(joint_engine, "require_single_cuda_device", lambda _: None)
    model = JointHolypolyModel(joint_basecaller(), unfreeze_last_layers=0)
    batch = collate_poly_training_examples(
        weak_status_examples(), supervise_status=True
    )
    optimizer = create_joint_optimizer(
        model,
        head_learning_rate=0.001,
        encoder_learning_rate=1e-5,
        weight_decay=0,
    )
    report = {}
    joint_engine.train_joint_epoch(
        model,
        [batch] * batches,
        ForbiddenSequences(),
        PolyAuxiliaryLoss(PolyLossWeights(lambda_location=0, length_loss="huber")),
        optimizer,
        lambda_basecall=0.1,
        device=torch.device("cpu"),
        gradient_clip_norm=2,
        global_step=0,
        monitor_frozen_sequences=False,
        profile_batches=2,
        profile_result=report,
    )
    assert report["measured_batches"] == measured
    expected = {
        "length_data_wait",
        "transfer",
        "encoder",
        "poly_heads",
        "auxiliary_loss_backward",
        "optimizer",
        "metrics",
    }
    assert set(report["seconds_per_batch"]) == (expected if measured else set())


def test_frozen_cli_runs_epochs_and_resumes_with_cached_monitoring(
    tmp_path,
    monkeypatch,
) -> None:
    """Exercise real orchestration/checkpoints with CPU surrogates and no POD5 IO."""
    from holypoly.datasets import prepare_sequence
    from holypoly.training import joint_command
    from holypoly.training.data import PolyTrainingDataset
    from tests.isolated.datasets.test_prepare_sequence import sequence_setup

    args, _, _, work, manifest = sequence_setup(tmp_path)
    prepare_sequence.prepare(args)
    model_dir = work / "model"
    model_dir.mkdir()
    (model_dir / "config.toml").write_text(
        "[model.seqdist]\nstate_len=1\n"
        "[model.encoder.transformer_encoder]\ndepth=2\n"
        "[standardisation]\nmean=0\nstdev=1\n"
    )

    # Redirect only this module's device selection. Global torch remains intact.
    class CpuTorch:
        def __getattr__(self, name):
            return getattr(torch, name)

        @staticmethod
        def device(_):
            return torch.device("cpu")

    monkeypatch.setattr(joint_command, "torch", CpuTorch())
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    for name in (
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "CUDA_CACHE_PATH",
    ):
        monkeypatch.setenv(name, str(work / "cache" / name.lower()))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "synthetic")
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: [])
    monkeypatch.setattr(torch.cuda, "set_rng_state_all", lambda _: None)
    for module in (joint_command, joint_engine, engine):
        monkeypatch.setattr(module, "require_single_cuda_device", lambda _: None)
    batch = collate_poly_training_examples(
        weak_status_examples(), supervise_status=True
    )
    monkeypatch.setattr(
        joint_command,
        "DataLoader",
        lambda dataset, **_: (
            [batch]
            if isinstance(dataset, PolyTrainingDataset)
            else [[synthetic_sequence()]] * len(dataset)
        ),
    )

    def basecaller(*_, **__):
        base = joint_basecaller()
        monkeypatch.setattr(base, "loss", lambda scores, *_: scores.square().mean())
        monkeypatch.setattr(base, "decode_batch", lambda _: ["ACGTACGT"])
        return base

    monkeypatch.setattr(joint_command, "load_bonito_v5_model", basecaller)
    arguments = [
        "--config",
        str(args.config),
        "--manifest",
        str(manifest),
        "--sequence-manifest",
        str(work / args.output_dir / "sequence_manifest.tsv"),
        "--model-dir",
        str(model_dir),
        "--run-id",
        "synthetic-performance",
        "--epochs",
        "2",
        "--unfreeze-last-layers",
        "0",
        "--chunk-size-samples",
        "252",
    ]
    result = joint_command.run(joint_command.build_parser().parse_args(arguments))
    log = work / "logs/synthetic-performance/train.jsonl"
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 4
    assert all(row["sequence_examples"] == 0 for row in rows if row["split"] == "train")
    assert all(
        row["sequence_metrics_source"] == "frozen_baseline"
        for row in rows
        if row["split"] == "validation"
    )
    assert any("save checkpoint" in stage for stage in result["stage_seconds"])
    before = log.read_text()
    result = joint_command.run(
        joint_command.build_parser().parse_args(
            [*arguments, "--epochs", "3", "--resume"]
        )
    )
    assert log.read_text().startswith(before)
    assert result["input_pipeline"]["frozen_sequence_monitor"] == "baseline"
    assert any("Epoch 1/2" in stage for stage in result["stage_seconds"])
    assert any("Epoch 3/3" in stage for stage in result["stage_seconds"])
