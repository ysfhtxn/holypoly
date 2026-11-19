"""TCN selection and resume guards with synthetic models and metadata only."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from holypoly.datasets import prepare_sequence
from holypoly.models.model import HolypolyModel
from holypoly.training import engine, joint_command
from holypoly.training.engine import CheckpointContext, load_checkpoint, save_checkpoint
from holypoly.training.losses import PolyLossWeights
from tests.isolated.datasets.test_prepare_sequence import sequence_setup
from tests.isolated.helpers import tiny_v5_basecaller


@pytest.mark.parametrize("unfreeze_layers", [0, 1])
@pytest.mark.parametrize("sum_scale", [0.002, 1.0])
@pytest.mark.parametrize("length_features", ["sum", "sum_mean_duration"])
def test_tcn_dry_run_preserves_existing_manifests(
    tmp_path,
    monkeypatch,
    unfreeze_layers: int,
    sum_scale: float,
    length_features: str,
) -> None:
    """Record architecture selection before loading weights, signals or CUDA."""
    args, _, _, work, manifest = sequence_setup(tmp_path)
    prepare_sequence.prepare(args)
    model_dir = work / "model"
    model_dir.mkdir()
    (model_dir / "config.toml").write_text(
        "[model.seqdist]\nstate_len=1\n"
        "[model.encoder.transformer_encoder]\ndepth=2\n"
        "[standardisation]\nmean=0\nstdev=1\n"
    )
    sequence = work / args.output_dir / "sequence_manifest.tsv"
    before = (manifest.read_bytes(), sequence.read_bytes())

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not load signals, model weights or CUDA")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    monkeypatch.setattr(joint_command, "load_bonito_v5_model", forbidden)
    arguments = joint_command.build_parser().parse_args(
        [
            "--config",
            str(args.config),
            "--manifest",
            str(manifest),
            "--sequence-manifest",
            str(sequence),
            "--model-dir",
            str(model_dir),
            "--chunk-size-samples",
            "252",
            "--head-type",
            "tcn",
            "--tcn-length-sum-scale",
            str(sum_scale),
            "--tcn-length-features",
            length_features,
            "--unfreeze-last-layers",
            str(unfreeze_layers),
            "--run-id",
            "new-tcn",
            "--dry-run",
        ]
    )
    result = joint_command.run(arguments)
    assert result["joint_config"]["head_type"] == "tcn"
    assert result["joint_config"].get("tcn_length_sum_scale", 1.0) == sum_scale
    assert result["joint_config"].get("tcn_length_features", "sum") == length_features
    assert result["joint_config"]["unfreeze_last_layers"] == unfreeze_layers
    assert result["joint_config"]["location_supervision"] is False
    assert result["input_pipeline"]["frozen_sequence_monitor"] == "baseline"
    assert result["input_pipeline"]["poly_only_forward"] is True
    assert (manifest.read_bytes(), sequence.read_bytes()) == before
    assert not (work / "models/new-tcn").exists()
    arguments.use_timing_reference = True
    with pytest.raises(ValueError, match="no timing reference"):
        joint_command.run(arguments)


def test_tcn_checkpoint_optimizer_restore_and_cross_architecture_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Restore TCN state exactly; reject a legacy resume before loading weights."""
    for name, function in (
        ("current_device", lambda: 0),
        ("get_device_name", lambda _: "synthetic"),
        ("get_rng_state_all", lambda: []),
        ("set_rng_state_all", lambda _: None),
    ):
        monkeypatch.setattr(torch.cuda, name, function)
    monkeypatch.setattr(engine, "require_single_cuda_device", lambda _: None)
    model = HolypolyModel(tiny_v5_basecaller(), length_head_mode="sum", head_type="tcn")
    optimizer = torch.optim.AdamW(model.trainable_parameters(), lr=0.001)
    # Populate optimizer buffers using only synthetic encoder features.
    features = torch.randn(2, 9, 8)
    output = model.length_head(features)
    objective = output.status_logits.square().mean() + output.length_nt.square().mean()
    objective.backward()
    optimizer.step()
    weights = PolyLossWeights(length_loss="huber")
    context = CheckpointContext(17, "synthetic", {}, "synthetic", weights, "synthetic")
    path = tmp_path / "synthetic-tcn.pt"
    save_checkpoint(
        path, model=model, optimizer=optimizer, epoch=1, global_step=2, context=context
    )
    restored = HolypolyModel(
        tiny_v5_basecaller(), length_head_mode="sum", head_type="tcn"
    )
    new_optimizer = torch.optim.AdamW(restored.trainable_parameters(), lr=0.01)
    state = load_checkpoint(
        path,
        model=restored,
        optimizer=new_optimizer,
        device=torch.device("cpu"),
        loss_weights=weights,
    )
    assert state.epoch == 1 and state.global_step == 2
    assert new_optimizer.param_groups[0]["lr"] == 0.001
    assert len(new_optimizer.state) == len(optimizer.state) > 0
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)
    mismatched = HolypolyModel(
        tiny_v5_basecaller(),
        length_head_mode="sum",
        head_type="tcn",
        tcn_length_sum_scale=0.02,
    )
    before = {name: value.clone() for name, value in mismatched.state_dict().items()}
    with pytest.raises(ValueError, match="tcn_length_sum_scale"):
        load_checkpoint(
            path,
            model=mismatched,
            optimizer=torch.optim.AdamW(mismatched.trainable_parameters()),
            device=torch.device("cpu"),
            loss_weights=weights,
        )
    for name, value in before.items():
        torch.testing.assert_close(mismatched.state_dict()[name], value, rtol=0, atol=0)
    old_inputs = HolypolyModel(
        tiny_v5_basecaller(),
        length_head_mode="sum",
        head_type="tcn",
        tcn_length_features="sum",
    )
    before = {name: value.clone() for name, value in old_inputs.state_dict().items()}
    with pytest.raises(ValueError, match="tcn_length_features"):
        load_checkpoint(
            path,
            model=old_inputs,
            optimizer=torch.optim.AdamW(old_inputs.trainable_parameters()),
            device=torch.device("cpu"),
            loss_weights=weights,
        )
    for name, value in before.items():
        torch.testing.assert_close(old_inputs.state_dict()[name], value, rtol=0, atol=0)
    legacy = HolypolyModel(tiny_v5_basecaller(), length_head_mode="sum")
    legacy_before = {k: v.clone() for k, v in legacy.state_dict().items()}
    with pytest.raises(ValueError, match="head architecture"):
        load_checkpoint(
            path,
            model=legacy,
            optimizer=torch.optim.AdamW(legacy.trainable_parameters()),
            device=torch.device("cpu"),
            loss_weights=weights,
        )
    for name, value in legacy_before.items():
        torch.testing.assert_close(legacy.state_dict()[name], value, rtol=0, atol=0)
    # Simulate a historical checkpoint with the original 128-input length MLP.
    old_model = HolypolyModel(
        tiny_v5_basecaller(),
        length_head_mode="sum",
        head_type="tcn",
        tcn_length_sum_scale=1.0,
        tcn_length_features="sum",
    )
    save_checkpoint(
        path,
        model=old_model,
        optimizer=torch.optim.AdamW(old_model.trainable_parameters()),
        epoch=1,
        global_step=2,
        context=context,
    )
    payload = torch.load(path, map_location="cpu", weights_only=False)
    del payload["head_config"]["tcn_length_sum_scale"]
    del payload["head_config"]["tcn_length_features"]
    torch.save(payload, path)
    old_model = HolypolyModel(
        tiny_v5_basecaller(),
        length_head_mode="sum",
        head_type="tcn",
        tcn_length_sum_scale=1.0,
        tcn_length_features="sum",
    )
    state = load_checkpoint(
        path,
        model=old_model,
        optimizer=torch.optim.AdamW(old_model.trainable_parameters()),
        device=torch.device("cpu"),
        loss_weights=weights,
    )
    assert state.epoch == 1 and old_model.length_head.tcn_length_sum_scale == 1.0
    with pytest.raises(ValueError, match="tcn_length_sum_scale"):
        load_checkpoint(
            path,
            model=restored,
            optimizer=new_optimizer,
            device=torch.device("cpu"),
            loss_weights=weights,
        )


@pytest.mark.parametrize("scale", ["0", "-1", "nan", "inf"])
def test_invalid_tcn_sum_scale_fails_before_data_access(scale: str) -> None:
    """Validate the CLI multiplier before attempting to open a manifest."""
    args = joint_command.build_parser().parse_args(
        [
            "--manifest",
            "unused",
            "--sequence-manifest",
            "unused",
            "--head-type",
            "tcn",
            "--tcn-length-sum-scale",
            scale,
        ]
    )
    with pytest.raises(ValueError, match="tcn_length_sum_scale"):
        joint_command.run(args)


def test_tcn_joint_optimizer_keeps_encoder_policy() -> None:
    """The same selected encoder tail and both TCN branches enter the optimizer."""
    from holypoly.models.joint_model import JointHolypolyModel, create_joint_optimizer
    from tests.isolated.training.test_reference_joint import joint_basecaller

    model = JointHolypolyModel(
        joint_basecaller(), head_type="tcn", length_head_mode="sum"
    )
    optimizer = create_joint_optimizer(
        model, head_learning_rate=0.001, encoder_learning_rate=1e-5, weight_decay=0
    )
    heads, encoder = optimizer.param_groups
    assert {id(p) for p in heads["params"]} == {
        id(p)
        for branch in (model.length_head, model.location_head)
        for p in branch.parameters()
    }
    stack = model.basecaller.encoder.transformer_encoder
    assert {id(p) for p in encoder["params"]} == {id(p) for p in stack[-1].parameters()}
    assert all(not p.requires_grad for p in stack[0].parameters())
    assert all(not p.requires_grad for p in model.basecaller.encoder.crf.parameters())
    model.train()
    assert model.length_head.blocks[0].dropout1.training
    assert not model.basecaller.training


@pytest.mark.parametrize("head_type", ["legacy", "tcn"])
def test_frozen_encoder_updates_only_heads_and_monitors_sequence_loss(
    monkeypatch, head_type: str
) -> None:
    """Zero unfrozen blocks preserve every backbone parameter and BN buffer."""
    import holypoly.training.joint_engine as joint_engine
    from holypoly.models.joint_model import JointHolypolyModel, create_joint_optimizer
    from holypoly.training.data import collate_poly_training_examples
    from holypoly.training.losses import PolyAuxiliaryLoss
    from tests.isolated.training.test_reference_joint import (
        joint_basecaller,
        synthetic_sequence,
        weak_status_examples,
    )

    monkeypatch.setattr(joint_engine, "require_single_cuda_device", lambda _: None)
    torch.manual_seed(19)
    model = JointHolypolyModel(
        joint_basecaller(),
        head_type=head_type,
        length_head_mode="sum",
        unfreeze_last_layers=0,
    )
    monkeypatch.setattr(
        model.basecaller, "loss", lambda scores, *_: scores.square().mean()
    )
    model.train()
    assert not model.basecaller.training
    assert not any(p.requires_grad for p in model.basecaller.parameters())
    before = {k: v.clone() for k, v in model.state_dict().items()}
    optimizer = create_joint_optimizer(
        model, head_learning_rate=0.001, encoder_learning_rate=1e-5, weight_decay=0
    )
    assert optimizer.param_groups[1]["params"] == []
    result, sequence = joint_engine.train_joint_epoch(
        model,
        [collate_poly_training_examples(weak_status_examples(), supervise_status=True)],
        [[synthetic_sequence()]],
        PolyAuxiliaryLoss(PolyLossWeights(lambda_location=0, length_loss="huber")),
        optimizer,
        lambda_basecall=0.1,
        device=torch.device("cpu"),
        gradient_clip_norm=2,
        global_step=0,
    )
    assert result.global_step == 1 and sequence["basecalling_loss"] > 0
    assert all(p.grad is None for p in model.basecaller.parameters())
    for name, tensor in model.basecaller.state_dict().items():
        torch.testing.assert_close(
            tensor, before["multitask.basecaller." + name], rtol=0, atol=0
        )
    for prefix in ("multitask.length_head.", "multitask.location_head."):
        assert any(
            not torch.equal(before[name], value)
            for name, value in model.state_dict().items()
            if name.startswith(prefix)
        )
    with pytest.raises(ValueError, match="must be 0"):
        JointHolypolyModel(joint_basecaller(), unfreeze_last_layers=-1)
