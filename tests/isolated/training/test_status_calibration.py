"""Frozen status calibration on synthetic signals; never accesses real POD5."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path

import pytest
import torch

from holypoly.datasets.prepare import FIELDS, write_table
from holypoly.models.model import HolypolyModel
from holypoly.training import engine, status_command, status_training
from holypoly.training.data import collate_poly_training_examples
from holypoly.training.losses import PolyLossWeights
from holypoly.training.status_training import (
    StatusFeatures,
    balanced_status_loss,
    balanced_status_weights,
    extract_status_features,
    freeze_for_status,
    status_logits,
    status_metrics,
    status_selection_key,
    train_status_epoch,
)
from tests.isolated.datasets.test_prepare_sequence import sequence_setup
from tests.isolated.helpers import tiny_v5_basecaller
from tests.isolated.training.test_reference_joint import weak_status_examples


def test_balancing_uses_train_counts_and_per_read_normalization() -> None:
    weights = torch.tensor(balanced_status_weights([0] * 6 + [1]))
    torch.testing.assert_close(weights, torch.tensor([7 / 12, 7 / 2, 0]))
    assert weights[1] / weights[0] == pytest.approx(6)
    with pytest.raises(ValueError):
        balanced_status_weights([0, 0])
    with pytest.raises(ValueError):
        balanced_status_weights([0, 1, 2])
    logits = torch.randn(7, 3)
    targets = torch.tensor([0] * 6 + [1])
    loss = balanced_status_loss(logits, targets, weights)
    parts = [
        balanced_status_loss(logits[:3], targets[:3], weights),
        balanced_status_loss(logits[3:], targets[3:], weights),
    ]
    torch.testing.assert_close(loss, (3 * parts[0] + 4 * parts[1]) / 7)


def status_fixture(device: torch.device, monkeypatch: pytest.MonkeyPatch | None = None):
    """Match cached features to actual classifier inputs, including padded signals."""
    if monkeypatch is not None:
        monkeypatch.setattr(
            status_training, "require_single_cuda_device", lambda _: None
        )
    torch.manual_seed(17)
    model = HolypolyModel(
        tiny_v5_basecaller(), length_head_mode="sum", hidden_features=6
    ).to(device)
    parameters = freeze_for_status(model)
    batch = collate_poly_training_examples(
        weak_status_examples(), supervise_status=True
    )
    features = extract_status_features(model, [batch], [0, 150], device=device)
    with torch.no_grad():
        original = model(batch.signal.to(device), batch.raw_padding_mask.to(device))
        expected = (original.location.status_logits + original.length.status_logits) / 2
    torch.testing.assert_close(
        status_logits(model, features.location.to(device), features.length.to(device)),
        expected,
    )
    torch.testing.assert_close(
        features.raw_lengths_nt.to(device), original.length.length_nt
    )
    assert len(parameters) == 4
    assert all(
        "status_classifier" in n for n, p in model.named_parameters() if p.requires_grad
    )
    return model, batch, features, original


def assert_frozen_step(
    device: torch.device, monkeypatch: pytest.MonkeyPatch | None = None
) -> None:
    model, batch, features, original = status_fixture(device, monkeypatch)
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0
    )
    count = train_status_epoch(
        model,
        features,
        optimizer,
        torch.ones(3, device=device),
        device=device,
        batch_size=2,
        seed=17,
        gradient_clip_norm=2,
    )
    assert count == 1
    trainable = {name for name, p in model.named_parameters() if p.requires_grad}
    after = model.state_dict()
    assert any(not torch.equal(before[name], after[name]) for name in trainable)
    for name in before:
        if name not in trainable:
            torch.testing.assert_close(before[name], after[name], rtol=0, atol=0)
    assert all(p.grad is None for p in model.parameters() if not p.requires_grad)
    with torch.no_grad():
        output = model(batch.signal.to(device), batch.raw_padding_mask.to(device))
    for left, right in (
        (output.length.length_nt, original.length.length_nt),
        (output.basecall_scores, original.basecall_scores),
        (output.location.location_logits, original.location.location_logits),
    ):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_cpu_classifier_step_preserves_all_frozen_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert_frozen_step(torch.device("cpu"), monkeypatch)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="status training requires selected CUDA"
)
def test_status_step_on_single_selected_gpu() -> None:
    assert torch.cuda.device_count() == 1
    assert_frozen_step(torch.device("cuda:0"))


def test_selection_rejects_all_present_and_all_zero_collapse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = HolypolyModel(tiny_v5_basecaller(), hidden_features=6)
    features = StatusFeatures(
        torch.zeros(4, 6),
        torch.zeros(4, 6),
        torch.tensor([1, 1, 0, 0]),
        torch.tensor([0.0, 0.0, 20.0, 20.0]),
        torch.ones(4) * 40,
    )
    weights = torch.ones(3)

    def evaluate(predictions):
        logits = torch.full((4, 3), -10.0)
        logits[torch.arange(4), predictions] = 10
        monkeypatch.setattr(status_training, "status_logits", lambda *_: logits)
        return status_metrics(model, features, weights, device=torch.device("cpu"))

    present = evaluate([0, 0, 0, 0])
    zero = evaluate([1, 1, 1, 1])
    perfect = evaluate([1, 1, 0, 0])
    assert present["selection_score"] == zero["selection_score"] == 0
    assert perfect["selection_score"] == perfect["zero_nt_recall"] == 1
    assert zero["twenty_nt_false_zero_fraction"] == 1
    assert perfect["twenty_nt_false_zero_fraction"] == 0
    assert status_selection_key(perfect) > status_selection_key(present)
    assert perfect["length_mae_nt"] == present["length_mae_nt"] == 30


def command_fixture(tmp_path: Path):
    """An external synthetic v4 manifest with 0/20 nt in every split."""
    args, _, _, work, manifest = sequence_setup(tmp_path)
    with manifest.open() as stream:
        positives = list(csv.DictReader(stream, delimiter="\t"))
    rows = list(positives)
    for row in positives:
        zero = {
            **row,
            "read_id": row["read_id"] + "-zero",
            "poly_type": "none",
            "training_length_nt": "0",
            "true_length_nt": "0",
            "nominal_length_nt": "0",
            "label_status": "zero_nt",
        }
        rows.append(zero)
    write_table(manifest, rows, FIELDS)
    model_dir = work / "model"
    model_dir.mkdir()
    (model_dir / "config.toml").write_text("[standardisation]\nmean=0\nstdev=1\n")
    source = work / "source.pt"
    model = HolypolyModel(
        tiny_v5_basecaller(), length_head_mode="sum", hidden_features=6
    )
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "head_config": model.head_config,
            "model_config": {"standardisation": {"mean": 0, "stdev": 1}},
            "manifest_schema_version": "construct-nominal-head-poly-v4",
            "loss_config": asdict(PolyLossWeights(length_loss="huber")),
        },
        source,
    )
    argv = [
        "--config",
        str(args.config),
        "--manifest",
        str(manifest),
        "--initialize-from",
        str(source),
        "--model-dir",
        str(model_dir),
        "--run-id",
        "synthetic-status",
        "--epochs",
        "2",
        "--batch-size",
        "2",
    ]
    return work, manifest, source, argv


def test_status_command_dry_run_reads_no_signals_weights_or_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, manifest, _, argv = command_fixture(tmp_path)
    original = manifest.read_bytes()

    def forbidden(*_args, **_kwargs):
        raise AssertionError("dry run cannot load signals, weights or GPU")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(status_command, "training_device", forbidden)
    monkeypatch.setattr(status_command, "load_bonito_v5_model", forbidden)
    assert status_command.main([*argv, "--dry-run"]) == 0
    assert not (work / "models").exists()
    assert original == manifest.read_bytes()
    assert status_command.main([*argv, "--epochs", "0", "--dry-run"]) == 2


def test_status_command_checkpoints_resume_and_never_reads_test(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    work, manifest, source, argv = command_fixture(tmp_path)
    reads = []
    for name in (
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "CUDA_CACHE_PATH",
    ):
        monkeypatch.setenv(name, str(work / "cache" / name.lower()))
    monkeypatch.setattr(status_command, "training_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(status_training, "require_single_cuda_device", lambda _: None)
    monkeypatch.setattr(engine, "require_single_cuda_device", lambda _: None)
    for name, function in (
        ("current_device", lambda: 0),
        ("get_device_name", lambda _: "synthetic"),
        ("get_rng_state_all", lambda: []),
        ("set_rng_state_all", lambda _: None),
    ):
        monkeypatch.setattr(torch.cuda, name, function)
    monkeypatch.setattr(
        status_command,
        "load_bonito_v5_model",
        lambda *_args, **_kwargs: tiny_v5_basecaller(),
    )

    def reader(_path, read_id):
        assert "test" not in read_id
        reads.append(read_id)
        return torch.linspace(-1, 1, 2410)

    monkeypatch.setattr(status_command, "Pod5SignalReader", lambda **_: reader)
    original_manifest = manifest.read_bytes()
    assert status_command.main(argv) == 0
    assert len(reads) == 4  # once per train/validation read, across all epochs
    assert status_command.main(argv) == 2  # refuses overwriting an old experiment
    current = work / "results" / "training_evaluation" / "synthetic-status"
    assert (current / "completion.json").is_file()
    assert status_command.main([*argv, "--epochs", "3", "--resume"]) == 0
    folder = work / "results" / "training_evaluation" / "synthetic-status"
    completion = json.loads((folder / "completion.json").read_text())
    assert completion["completed_epochs"] == 3
    assert completion["frozen_state_unchanged"] and not completion["test_used"]
    assert original_manifest == manifest.read_bytes()
    assert (work / "models" / "synthetic-status" / "status.epoch_0003.pt").is_file()
    latest = torch.load(
        work / "models" / "synthetic-status" / "heads.pt", weights_only=False
    )
    initial = torch.load(source, weights_only=False)
    for key, value in latest["model_state_dict"].items():
        if "status_classifier" not in key:
            torch.testing.assert_close(
                value, initial["model_state_dict"][key], rtol=0, atol=0
            )
    assert (
        status_command.main(
            [*argv, "--epochs", "4", "--learning-rate", "0.02", "--resume"]
        )
        == 2
    )
    # Simulate a pre-layout experiment and resume without migrating its results.
    legacy = work / "results" / "synthetic-status"
    current.rename(legacy)
    assert status_command.main([*argv, "--epochs", "4", "--resume"]) == 0
    assert not current.exists()
    assert json.loads((legacy / "completion.json").read_text())["completed_epochs"] == 4
