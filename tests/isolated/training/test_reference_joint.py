"""Synthetic joint supervision checks; real native CRF backward requires CUDA."""

from __future__ import annotations

import csv
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from holypoly.datasets import prepare_sequence
from holypoly.models.joint_model import JointHolypolyModel, create_joint_optimizer
from holypoly.training import engine, joint_command
from holypoly.training.data import (
    ManifestLabelStatus,
    ManifestPolyType,
    ManifestSplit,
    ManifestSupervisionMode,
    PolyTrainingDataset,
    PolyTrainingRecord,
    collate_poly_training_examples,
    load_training_manifest,
)
from holypoly.training.engine import CheckpointContext, load_checkpoint, save_checkpoint
from holypoly.training.joint_engine import (
    edit_distance,
    reference_crf_loss,
    sequence_scores,
    train_joint_epoch,
    validate_sequences,
)
from holypoly.training.losses import PolyAuxiliaryLoss, PolyLossWeights
from holypoly.training.sequence_data import (
    SequenceDataset,
    SequenceExample,
    SequenceRecord,
    load_sequence_manifest,
)
from tests.isolated.datasets.test_prepare_sequence import sequence_setup
from tests.isolated.helpers import tiny_v5_basecaller


def joint_basecaller():
    """Two small parameterized synthetic encoder blocks, with original native CRF."""
    base = tiny_v5_basecaller()
    base.encoder.transformer_encoder = nn.Sequential(
        nn.Sequential(nn.Linear(8, 8), nn.GELU()),
        nn.Sequential(nn.Linear(8, 8), nn.GELU()),
    )
    return base


def synthetic_sequence(samples: int = 125) -> SequenceExample:
    """An in-memory, acquisition-order head label with a partial final stride."""
    record = SequenceRecord(
        "synthetic",
        Path("synthetic.pod5"),
        ManifestSplit.TRAIN,
        samples,
        0,
        samples,
        "ACGTACGT",
        "a" * 64,
        "b" * 64,
    )
    return SequenceExample(
        torch.linspace(-1, 1, samples), torch.tensor([1, 2, 3, 4] * 2), record
    )


def weak_status_examples():
    """Two synthetic v4 candidates with unknown boundaries and suffix padding."""
    records = [
        PolyTrainingRecord(
            read_id=f"synthetic-{length}",
            pod5_path=Path("synthetic.pod5"),
            pod5_file_id="synthetic",
            split=ManifestSplit.TRAIN,
            orientation="forward",
            raw_signal_length=150,
            chunk_start_sample=12,
            chunk_end_sample=137,
            poly_start_sample=None,
            poly_end_sample=None,
            poly_type=ManifestPolyType.NONE if length == 0 else ManifestPolyType.POLY_A,
            true_length_nt=length,
            nominal_length_nt=length,
            label_status=(
                ManifestLabelStatus.ZERO_NT
                if length == 0
                else ManifestLabelStatus.POLY_PRESENT
            ),
            poly_is_complete=None,
            failure_reason="",
            label_source="construct_nominal_weak_supervision",
            manifest_version="construct-nominal-head-poly-v4",
            supervision_mode=ManifestSupervisionMode.LENGTH_ONLY,
            model_input_samples=132,
            head_excluded=True,
            head_boundary_sample=12,
        )
        for length in (0, 150)
    ]
    dataset = PolyTrainingDataset(
        records, lambda _path, _read: torch.linspace(-1, 1, 150)
    )
    return [dataset[0], dataset[1]]


def test_status_opt_in_preserves_lengths_locations_padding_and_manifest() -> None:
    examples = weak_status_examples()
    original_records = [example.record for example in examples]
    default = collate_poly_training_examples(examples)
    enabled = collate_poly_training_examples(examples, supervise_status=True)
    assert default.status_targets.tolist() == [-100, -100]
    assert enabled.status_targets.tolist() == [1, 0]
    assert enabled.true_length_nt.tolist() == [0, 150]
    assert torch.all(enabled.location_targets == -100)
    assert enabled.raw_padding_mask[:, 125:].all()
    assert torch.equal(default.location_targets, enabled.location_targets)
    assert torch.equal(default.signal, enabled.signal)
    assert original_records == [example.record for example in examples]
    failed = replace(
        examples[0],
        record=replace(
            examples[0].record,
            label_status=ManifestLabelStatus.FAILURE,
            true_length_nt=float("nan"),
            failure_reason="synthetic_failure",
        ),
    )
    batch = collate_poly_training_examples([failed], supervise_status=True)
    assert batch.status_targets.tolist() == [2]
    model = JointHolypolyModel(
        joint_basecaller(), hidden_features=6, length_head_mode="sum"
    )
    losses = PolyAuxiliaryLoss(PolyLossWeights(length_loss="huber"))(
        model(batch.signal, batch.raw_padding_mask),
        batch.location_targets,
        batch.true_length_nt,
        batch.status_targets,
    )
    assert losses["length_examples"] == 0 and losses["length_loss"] == 0
    assert torch.isfinite(losses["status_loss"]) and losses["status_examples"] == 1


@pytest.mark.parametrize("with_timing", [False, True])
def test_three_loss_joint_step_updates_status_but_not_location_or_crf(
    monkeypatch: pytest.MonkeyPatch,
    with_timing: bool,
) -> None:
    """CPU uses surrogate CRF; the native CUDA test below covers actual CRF."""
    import holypoly.training.joint_engine as joint_engine

    monkeypatch.setattr(joint_engine, "require_single_cuda_device", lambda _: None)
    torch.manual_seed(17)
    model = JointHolypolyModel(
        joint_basecaller(),
        hidden_features=6,
        length_head_mode="sum",
        use_timing_reference=with_timing,
    )
    monkeypatch.setattr(
        model.basecaller, "loss", lambda scores, *_: scores.square().mean()
    )
    before = {key: value.clone() for key, value in model.state_dict().items()}
    optimizer = create_joint_optimizer(
        model,
        head_learning_rate=0.001,
        encoder_learning_rate=1e-5,
        weight_decay=0,
    )
    batch = collate_poly_training_examples(
        weak_status_examples(),
        supervise_status=True,
        timing_references=(
            {"synthetic-0": (0, 0, 0), "synthetic-150": (2.5, 0.1, 1)}
            if with_timing
            else None
        ),
    )
    weights = PolyLossWeights(lambda_location=0, length_loss="huber")
    criterion = PolyAuxiliaryLoss(weights)
    output = model(batch.signal, batch.raw_padding_mask)
    losses = criterion(
        output, batch.location_targets, batch.true_length_nt, batch.status_targets
    )
    assert losses["location_loss"] == 0 and losses["status_loss"] > 0
    assert losses["length_examples"] == losses["status_examples"] == 2
    torch.testing.assert_close(
        losses["total_loss"],
        losses["length_loss"] + losses["status_loss"],
    )
    shared_logits = (output.location.status_logits + output.length.status_logits) / 2
    torch.testing.assert_close(
        losses["status_loss"],
        torch.nn.functional.cross_entropy(shared_logits, batch.status_targets),
    )
    result, sequence = train_joint_epoch(
        model,
        [batch],
        [[synthetic_sequence()]],
        criterion,
        optimizer,
        lambda_basecall=0.1,
        device=torch.device("cpu"),
        gradient_clip_norm=2,
        global_step=0,
    )
    assert result.global_step == 1 and result.losses["status_loss"] > 0
    assert sequence["basecalling_loss"] > 0
    if with_timing:
        assert model.length_head.timing_residual[-1].weight.grad.abs().sum() > 0
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in model.basecaller.encoder.transformer_encoder[
                -1
            ].parameters()
        )
    for head in (model.location_head, model.length_head):
        assert head.status_classifier.weight.grad.abs().sum() > 0
    after = model.state_dict()
    status_keys = [key for key in before if key.endswith("status_classifier.weight")]
    assert len(status_keys) == 2
    assert all(not torch.equal(before[key], after[key]) for key in status_keys)
    for key in before:
        if "location_classifier" in key or key.startswith("basecaller.encoder.crf"):
            torch.testing.assert_close(before[key], after[key], rtol=0, atol=0)


def test_joint_gradients_reach_only_selected_encoder_and_poly_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CPU unit backward uses a differentiable surrogate, not native CUDA CRF."""
    torch.manual_seed(17)
    basecaller = joint_basecaller().eval()
    model = JointHolypolyModel(
        basecaller, hidden_features=6, length_head_mode="sum"
    ).eval()
    example = synthetic_sequence()
    calls = []

    def surrogate(scores, targets, lengths):
        calls.append((scores.shape, targets.detach().clone(), lengths.detach().clone()))
        return scores.square().mean()

    monkeypatch.setattr(basecaller, "loss", surrogate)
    scores = sequence_scores(model, example, torch.device("cpu"))
    assert scores.shape[0] == 21  # ceil(125 / 6), not padded 132 / 6.
    crf = reference_crf_loss(model, scores, example, torch.device("cpu"))
    assert calls[0][1].tolist() == [[1, 2, 3, 4] * 2]
    assert calls[0][2].tolist() == [8]
    crf.backward()
    layers = model.basecaller.encoder.transformer_encoder
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in layers[-1].parameters()
    )
    assert all(p.grad is None and not p.requires_grad for p in layers[0].parameters())
    assert all(
        p.grad is None and not p.requires_grad
        for p in model.basecaller.encoder.crf.parameters()
    )
    assert all(p.grad is None for p in model.length_head.parameters())
    model.zero_grad(set_to_none=True)
    output = model(torch.randn(2, 1, 120))
    losses = PolyAuxiliaryLoss(PolyLossWeights(length_loss="huber"))(
        output,
        torch.full((2, 20), -100),
        torch.tensor([0.0, 150.0]),
        torch.full((2,), -100),
    )
    losses["total_loss"].backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.length_head.parameters()
    )
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in layers[-1].parameters()
    )
    optimizer = create_joint_optimizer(
        model, head_learning_rate=0.001, encoder_learning_rate=1e-5, weight_decay=0
    )
    assert [g["lr"] for g in optimizer.param_groups] == [0.001, 1e-5]
    model.train()
    assert not model.basecaller.training
    assert all(
        not module.training for module in model.basecaller.encoder.conv.modules()
    )


@pytest.mark.parametrize("mode", ["mean", "sum"])
def test_joint_initial_forward_preserves_basecalling_and_rejects_invalid_depth(
    mode: str,
) -> None:
    base = joint_basecaller().eval()
    original = {key: value.clone() for key, value in base.state_dict().items()}
    signal = torch.randn(2, 1, 120)
    model = JointHolypolyModel(base, length_head_mode=mode).eval()
    for key, value in base.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    with torch.no_grad():
        # Compare both routes under the same joint-mode gradient flags; freezing
        # can change PyTorch's linear kernel without changing parameters.
        expected = base(signal)
        actual = model(signal).basecall_scores
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    with pytest.raises(ValueError, match="final Transformer"):
        JointHolypolyModel(joint_basecaller(), unfreeze_last_layers=3)
    with pytest.raises(ValueError, match="only unfreezes"):
        JointHolypolyModel(joint_basecaller(), freeze_basecalling_head=False)


@pytest.mark.parametrize(
    "mutation",
    [
        "split",
        "source_manifest_sha256",
        "sequence_target",
        "chunk_end_sample",
        "read_id",
    ],
)
def test_sequence_loader_rejects_leakage_and_invalid_labels(
    tmp_path: Path, mutation: str
) -> None:
    args, _, data, work, manifest = sequence_setup(tmp_path)
    prepare_sequence.prepare(args)
    path = work / args.output_dir / "sequence_manifest.tsv"
    with path.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    rows[0][mutation] = {
        "split": "validation",
        "source_manifest_sha256": "c" * 64,
        "sequence_target": "ACNT",
        "chunk_end_sample": "2409",
        "read_id": "not-in-source",
    }[mutation]
    prepare_sequence.write_table(path, rows, prepare_sequence.FIELDS)
    with pytest.raises(ValueError):
        load_sequence_manifest(
            path,
            load_training_manifest(manifest),
            source_manifest_sha256=prepare_sequence.sha256(manifest),
            data_root=data,
        )


def test_sequence_dataset_slices_actual_head_and_keeps_order(tmp_path: Path) -> None:
    args, _, data, work, manifest = sequence_setup(tmp_path, reverse=True)
    prepare_sequence.prepare(args)
    records = load_sequence_manifest(
        work / args.output_dir / "sequence_manifest.tsv",
        load_training_manifest(manifest),
        source_manifest_sha256=prepare_sequence.sha256(manifest),
        data_root=data,
    )
    dataset = SequenceDataset(
        records,
        lambda path, read_id: torch.arange(2410).float(),
        split=ManifestSplit.TRAIN,
    )
    example = dataset[0]
    assert example.signal[0] == example.record.chunk_start_sample
    assert example.signal[-1] == example.record.chunk_end_sample - 1
    assert (
        "".join("NACGT"[index] for index in example.targets)
        == example.record.sequence_target
    )


def test_joint_command_dry_run_loads_no_weights_signals_or_gpu(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args, _, _, work, manifest = sequence_setup(tmp_path)
    prepare_sequence.prepare(args)
    model_dir = work / "model"
    model_dir.mkdir()
    (model_dir / "config.toml").write_text(
        "[model.seqdist]\nstate_len=1\n[model.encoder.transformer_encoder]\ndepth=2\n[standardisation]\nmean=0\nstdev=1\n"
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("dry run must not read model weights, POD5 signals or GPU")

    monkeypatch.setattr(torch, "load", forbidden)
    monkeypatch.setattr(torch.cuda, "device_count", forbidden)
    monkeypatch.setattr(joint_command, "load_bonito_v5_model", forbidden)
    arguments = [
        "--config",
        str(args.config),
        "--manifest",
        str(manifest),
        "--sequence-manifest",
        str(work / args.output_dir / "sequence_manifest.tsv"),
        "--model-dir",
        str(model_dir),
        "--chunk-size-samples",
        "252",
        "--dry-run",
        "--run-id",
        "synthetic-joint",
    ]
    assert joint_command.main(arguments) == 0
    source_bytes = manifest.read_bytes()
    default_summary = joint_command.run(
        joint_command.build_parser().parse_args(arguments)
    )
    assert default_summary["status_supervision_enabled"] is True
    assert Path(default_summary["result_dir"]) == (
        work / "results" / "training_evaluation" / "synthetic-joint"
    )
    assert default_summary["joint_config"]["status_train_counts"] == {"poly_present": 1}
    assert default_summary["loss_config"]["lambda_location"] == 0
    assert default_summary["loss_config"]["lambda_status"] > 0
    legacy_summary = joint_command.run(
        joint_command.build_parser().parse_args([*arguments, "--no-supervise-status"])
    )
    assert legacy_summary["status_supervision_enabled"] is False
    assert "status_supervision" not in legacy_summary["joint_config"]
    enabled = joint_command.build_parser().parse_args(
        [
            *arguments,
            "--supervise-status",
            "--lambda-status",
            "2",
            "--lambda-length",
            "0.5",
        ]
    )
    summary = joint_command.run(enabled)
    assert (
        summary["joint_config"]["status_supervision"]
        == "construct_nominal_weak_supervision"
    )
    assert summary["joint_config"]["status_train_counts"] == {"poly_present": 1}
    assert summary["loss_config"]["lambda_location"] == 0
    assert summary["loss_config"]["lambda_status"] == 2
    assert summary["loss_config"]["lambda_length"] == 0.5
    assert manifest.read_bytes() == source_bytes
    assert (
        joint_command.main([*arguments, "--supervise-status", "--lambda-status", "0"])
        == 2
    )
    assert joint_command.main([*arguments, "--lambda-length", "nan"]) == 2
    assert not (work / "models").exists() and not (work / "logs").exists()
    assert joint_command.main([*arguments, "--unfreeze-last-layers", "3"]) == 2


@pytest.mark.parametrize("status_setting", [None, "0", "1"])
def test_joint_script_dry_run_defaults_to_three_losses_and_allows_legacy_mode(
    tmp_path: Path, status_setting: str | None
) -> None:
    """Exercise actual shell-to-CLI defaults with synthetic metadata, no training."""
    args, _, _, work, manifest = sequence_setup(tmp_path)
    with manifest.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    for row in rows:
        row["model_input_samples"] = "12000"
    prepare_sequence.write_table(manifest, rows, list(rows[0]))
    prepare_sequence.prepare(args)
    model_dir = work / "model"
    model_dir.mkdir()
    (model_dir / "config.toml").write_text(
        "[model.seqdist]\nstate_len=1\n[model.encoder.transformer_encoder]\ndepth=2\n"
    )
    args.config.write_text(
        args.config.read_text() + f"\ntraining:\n  bonito_model: {model_dir}\n"
    )
    ignored = {
        "STATUS_SUPERVISION",
        "RESUME",
        "INITIALIZE_FROM",
        "EXPERIMENT_ID",
        "GPU_ID",
        "EPOCHS",
        "BATCH_SIZE",
        "SEQUENCE_BATCH_SIZE",
        "PREFETCH_BATCHES",
        "POD5_READER_CACHE_SIZE",
        "UNFREEZE_LAST_LAYERS",
        "HEAD_LEARNING_RATE",
        "ENCODER_LEARNING_RATE",
        "HUBER_DELTA_NT",
        "SEED",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ignored and not key.startswith(("HOLYPOLY_", "LAMBDA_"))
    }
    environment.update(
        EXPERIMENT_ID="synthetic-default-joint", PYTHONDONTWRITEBYTECODE="1"
    )
    if status_setting is not None:
        environment["STATUS_SUPERVISION"] = status_setting
    project = Path(__file__).resolve().parents[3]
    before = set(work.rglob("*"))
    completed = subprocess.run(
        [
            "bash",
            str(project / "scripts/run_reference_joint.sh"),
            str(manifest),
            str(work / args.output_dir / "sequence_manifest.tsv"),
            str(args.config),
            "--dry-run",
        ],
        cwd=project,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    summary = json.loads(completed.stdout)
    assert summary["status_supervision_enabled"] is (status_setting != "0")
    assert summary["joint_config"]["length_batch_size"] == 32
    assert summary["loss_config"]["length_loss"] == "huber"
    assert summary["joint_config"]["lambda_basecall"] == 0.1
    assert summary["input_pipeline"]["prefetch_batches"] == 2
    assert summary["input_pipeline"]["pod5_reader_cache_size"] == 8
    assert set(work.rglob("*")) == before


def test_joint_checkpoint_rejects_frozen_resume_and_changed_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name, function in (
        ("current_device", lambda: 0),
        ("get_device_name", lambda _: "synthetic"),
        ("get_rng_state_all", lambda: []),
        ("set_rng_state_all", lambda _: None),
    ):
        monkeypatch.setattr(torch.cuda, name, function)
    monkeypatch.setattr(engine, "require_single_cuda_device", lambda _: None)
    model = JointHolypolyModel(
        joint_basecaller(), hidden_features=6, length_head_mode="sum"
    )
    optimizer = create_joint_optimizer(
        model, head_learning_rate=0.001, encoder_learning_rate=1e-5, weight_decay=0
    )
    config = {"lambda_basecall": 0.1, "unfreeze_last_layers": 1}
    weights = PolyLossWeights(length_loss="huber")
    context = CheckpointContext(
        17, "synthetic", {}, "synthetic", weights, "synthetic", joint_config=config
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(
        path, model=model, optimizer=optimizer, epoch=1, global_step=2, context=context
    )
    for expected in (
        None,
        {**config, "lambda_basecall": 0.2},
        {**config, "status_supervision": "construct_nominal_weak_supervision"},
    ):
        with pytest.raises(ValueError, match="joint training configuration"):
            load_checkpoint(
                path,
                model=model,
                optimizer=optimizer,
                device=torch.device("cpu"),
                loss_weights=weights,
                joint_config=expected,
            )
    restored = load_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        loss_weights=weights,
        joint_config=config,
    )
    assert restored.epoch == 1 and restored.global_step == 2


def test_edit_distance_handles_empty_and_indel_sequences() -> None:
    assert edit_distance("ACGT", "ACGT") == 0
    assert edit_distance("ACGT", "") == 4
    assert edit_distance("ACGT", "AGT") == 1
    assert edit_distance("ACGT", "ACGTT") == 1


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native CTC-CRF backward requires CUDA"
)
def test_native_crf_and_joint_epoch_on_single_selected_gpu() -> None:
    """Actual native CRF on synthetic signal; never reads real data or model files."""
    assert torch.cuda.device_count() == 1, "select one GPU with CUDA_VISIBLE_DEVICES"
    from bonito.crf.model import CTC_CRF
    from bonito.nn import LinearCRFEncoder, TransformerEncoderLayer

    base = tiny_v5_basecaller()
    base.encoder.transformer_encoder = nn.Sequential(
        *[
            TransformerEncoderLayer(8, 1, 16, 1.0, 1.0, attn_window=[4, 4])
            for _ in range(2)
        ]
    )
    base.encoder.crf = LinearCRFEncoder(
        insize=8,
        n_base=4,
        state_len=5,
        bias=False,
        blank_score=2.0,
        permute=[1, 0, 2],
    )
    base.seqdist = CTC_CRF(5, ["N", "A", "C", "G", "T"])
    model = JointHolypolyModel(base, hidden_features=6, length_head_mode="sum").cuda()
    optimizer = create_joint_optimizer(
        model, head_learning_rate=0.001, encoder_learning_rate=1e-5, weight_decay=0
    )
    result, sequence = train_joint_epoch(
        model,
        [collate_poly_training_examples(weak_status_examples(), supervise_status=True)],
        [[synthetic_sequence()]],
        PolyAuxiliaryLoss(PolyLossWeights(length_loss="huber")),
        optimizer,
        lambda_basecall=0.1,
        device=torch.device("cuda:0"),
        gradient_clip_norm=2,
        global_step=0,
    )
    assert result.global_step == 1 and sequence["sequence_examples"] == 1
    assert result.losses["status_loss"] > 0 and result.losses["length_loss"] > 0
    assert result.losses["location_loss"] == 0
    for head in (model.location_head, model.length_head):
        assert head.status_classifier.weight.grad.abs().sum() > 0
    assert torch.isfinite(torch.tensor(sequence["basecalling_loss"]))
    assert all(p.grad is None for p in model.basecaller.encoder.crf.parameters())
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.basecaller.encoder.transformer_encoder[-1].parameters()
    )
    metrics = validate_sequences(
        model, [[synthetic_sequence()]], device=torch.device("cuda:0")
    )
    assert metrics["sequence_examples"] == 1
    assert torch.isfinite(torch.tensor(metrics["reference_edit_rate"]))
