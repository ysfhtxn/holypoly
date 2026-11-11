"""Freeze a checkpoint and calibrate its existing status classifiers on one GPU."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import toml
import torch
from torch.utils.data import DataLoader

from holypoly.core.config import load_config
from holypoly.core.result_paths import training_result_dir
from holypoly.datasets.prepare_sequence import sha256
from holypoly.models.model import (
    HolypolyModel,
    head_config_from_checkpoint,
    load_bonito_v5_model,
)
from holypoly.models.poly_heads import ReadStatus
from holypoly.training.data import (
    ManifestSplit,
    Pod5SignalReader,
    PolyTrainingDataset,
    collate_poly_training_examples,
    load_training_manifest,
)
from holypoly.training.engine import (
    CheckpointContext,
    load_checkpoint,
    require_single_cuda_device,
    save_checkpoint,
    set_random_seed,
)
from holypoly.training.losses import PolyLossWeights
from holypoly.training.run_validation import (
    git_commit,
    manifest_summary,
    single_manifest_version,
    validate_manifest_sources,
    validate_training_paths,
)
from holypoly.training.status_training import (
    balanced_status_weights,
    extract_status_features,
    freeze_for_status,
    status_metrics,
    status_selection_key,
    train_status_epoch,
)


def build_parser() -> argparse.ArgumentParser:
    """Expose a separate experiment so previous joint/frozen commands stay intact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--initialize-from", required=True, type=Path)
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument(
        "--run-id",
        default="v4-status-calibration-" + datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--gradient-clip-norm", type=float, default=2.0)
    parser.add_argument("--random-seed", type=int, default=20260923)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Atomically replace a report within an already validated experiment folder."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def training_device() -> torch.device:
    """Require exactly one explicitly selected physical GPU for real training."""
    device = torch.device("cuda:0")
    if not os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.device_count() != 1:
        raise ValueError("select exactly one GPU with CUDA_VISIBLE_DEVICES")
    require_single_cuda_device(device)
    return device


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Validate metadata, extract frozen pools once, then calibrate classifiers."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.run_id):
        raise ValueError("run-id must be one directory name")
    if args.epochs < 1 or args.batch_size < 1 or args.random_seed < 0:
        raise ValueError("epochs/batch size must be positive; seed non-negative")
    for key in ("learning_rate", "gradient_clip_norm"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise ValueError(f"{key} must be finite and positive")
    config = load_config(args.config)
    if config.paths.data_root is None or config.paths.work_root is None:
        raise ValueError("configure data_root/work_root")
    data, work = (
        p.expanduser().resolve()
        for p in (config.paths.data_root, config.paths.work_root)
    )
    project = Path(__file__).resolve().parents[3]
    if work.is_relative_to(data) or work.is_relative_to(project):
        raise ValueError("work_root must be external and outside data_root")
    manifest = (work / args.manifest).resolve()
    source = (work / args.initialize_from).resolve()
    model_value = args.model_dir or config.training.bonito_model
    if model_value is None:
        raise ValueError("configure a Bonito model or use --model-dir")
    model_dir = (work / model_value).resolve()
    checkpoint = (work / "models" / args.run_id / "heads.pt").resolve()
    best_path = checkpoint.with_name("heads.best.pt")
    log = (work / "logs" / args.run_id / "train.jsonl").resolve()
    result_dir = training_result_dir(work, args.run_id, resume=args.resume)
    for path in (manifest, source):
        if (
            not path.is_relative_to(work)
            or path.is_relative_to(data)
            or not path.is_file()
        ):
            raise ValueError("manifest and initialization must exist inside work_root")
    for path in (checkpoint.parent, log.parent, result_dir):
        if (
            not path.is_relative_to(work)
            or path == work
            or path.is_relative_to(data)
            or path.is_relative_to(project)
            or manifest.is_relative_to(path)
            or source.is_relative_to(path)
        ):
            raise ValueError("unsafe calibration output")
        if path.exists() and not args.resume:
            raise FileExistsError(f"choose a new run-id; output exists: {path}")
    validate_training_paths(
        data_root=data,
        work_root=work,
        manifest_path=manifest,
        model_directory=model_dir,
        checkpoint_path=checkpoint,
        loss_log_path=log,
    )
    records = load_training_manifest(manifest)
    validate_manifest_sources(records, data_root=data)
    if len({r.read_id for r in records}) != len(records):
        raise ValueError("calibration requires unique read IDs across splits")
    if any(r.manifest_version != "construct-nominal-head-poly-v4" for r in records):
        raise ValueError(
            "calibration currently requires original v4 construct weak labels"
        )
    selected = {
        split: [r for r in records if r.split == split]
        for split in (ManifestSplit.TRAIN, ManifestSplit.VALIDATION)
    }
    for split, group in selected.items():
        if not any(r.nominal_length_nt == 0 for r in group) or not any(
            r.nominal_length_nt == 20 for r in group
        ):
            raise ValueError(f"{split.value} needs 0-nt and 20-nt examples")
    targets = [
        int(ReadStatus.ZERO_NT)
        if r.true_length_nt == 0
        else int(ReadStatus.POLY_PRESENT)
        for r in selected[ManifestSplit.TRAIN]
    ]
    class_weights = balanced_status_weights(targets)
    summary = {
        **manifest_summary(records, epochs=args.epochs),
        "run_id": args.run_id,
        "result_dir": str(result_dir),
        "dry_run": args.dry_run,
        "initialize_from": str(source),
        "checkpoint_bytes": source.stat().st_size,
        "class_weights": class_weights,
        "selection_rule": (
            "max harmonic mean(zero recall, 20-nt present recall); "
            "tie all-positive recall, lower balanced CE, earlier epoch"
        ),
        "test_used": False,
        "limitations": (
            "Construct weak status labels, no independent read truth or "
            "demonstrated failure detection; raw length cannot improve when frozen."
        ),
    }
    print(json.dumps(summary), flush=True)
    if args.dry_run:
        return summary
    device = training_device()
    for name in (
        "TORCHINDUCTOR_CACHE_DIR",
        "TRITON_CACHE_DIR",
        "TORCH_EXTENSIONS_DIR",
        "CUDA_CACHE_PATH",
    ):
        cache = Path(
            os.environ.get(name, str(work / "cache" / args.run_id / name.lower()))
        ).resolve()
        if not cache.is_relative_to(work) or cache.is_relative_to(data):
            raise ValueError(f"{name} must remain inside work_root")
        os.environ[name] = str(cache)
    set_random_seed(args.random_seed)
    payload = torch.load(source, map_location="cpu", weights_only=False)
    model_config = toml.load(model_dir / "config.toml")
    if payload["model_config"] != model_config:
        raise ValueError("source checkpoint and Bonito model configs differ")
    if payload["manifest_schema_version"] != single_manifest_version(records):
        raise ValueError("source checkpoint schema differs from manifest")
    source_joint = payload.get("joint_config") or {}
    if source_joint.get("length_manifest_sha256", sha256(manifest)) != sha256(manifest):
        raise ValueError("source checkpoint used a different length manifest")
    model = HolypolyModel(
        load_bonito_v5_model(model_dir, device=str(device)),
        **head_config_from_checkpoint(payload),
    ).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    parameters = freeze_for_status(model)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=0)
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    calibration = {
        "schema": "frozen-status-calibration-v1",
        "status_supervision": "construct_nominal_weak_supervision",
        "status_class_balance": "train_inverse_frequency_mean_per_read",
        "status_class_weights": class_weights,
        "train_status_counts": dict(
            Counter(r.label_status.value for r in selected[ManifestSplit.TRAIN])
        ),
        "length_manifest_sha256": sha256(manifest),
        "initialize_from": str(source),
        "initialize_from_sha256": sha256(source),
        "source_joint_config": source_joint,
        "model_dir": str(model_dir),
        "trainable_parameters": trainable_names,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_clip_norm": args.gradient_clip_norm,
        "random_seed": args.random_seed,
        "selection_rule": summary["selection_rule"],
        "location_supervision": False,
        "length_supervision": False,
        "basecalling_supervision": False,
        "decision_rule": "shared_logits_argmax; no length gating",
    }
    weights = PolyLossWeights(
        lambda_location=0,
        lambda_length=0,
        lambda_status=1,
        length_loss="huber",
        huber_delta_nt=float(payload["loss_config"].get("huber_delta_nt", 10)),
    )
    context = CheckpointContext(
        args.random_seed,
        git_commit(project / "src" / "bonito"),
        model_config,
        single_manifest_version(records),
        weights,
        git_commit(project),
        args.random_seed,
        calibration,
    )
    start_epoch, step, history = 0, 0, []
    if args.resume:
        state = load_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            device=device,
            data_order_seed=args.random_seed,
            loss_weights=weights,
            joint_config=calibration,
        )
        start_epoch, step = state.epoch, state.global_step
        history = [json.loads(line) for line in log.read_text().splitlines()]
        expected = [
            (epoch, split)
            for epoch in range(1, start_epoch + 1)
            for split in ("train", "validation")
        ]
        if (
            [(r["epoch"], r["split"]) for r in history] != expected
            or history[-1]["global_step"] != step
            or any(r["joint_config"] != calibration for r in history)
            or start_epoch >= args.epochs
        ):
            raise ValueError("resume log/settings/target epoch mismatch")
        if (
            not best_path.is_file()
            or not (result_dir / "validation_baseline.json").is_file()
        ):
            raise ValueError("resume requires baseline and best checkpoint")
    # Keep source tensors to verify every frozen parameter and buffer before export.
    reference_state = payload["model_state_dict"]
    summary.update(
        joint_config=calibration,
        loss_config=asdict(weights),
        started_utc=datetime.now(UTC).isoformat(),
        parameter_counts=model.parameter_counts(),
        command=[
            sys.executable,
            "-m",
            "holypoly.training.status_command",
            *sys.argv[1:],
        ],
    )
    for folder in (checkpoint.parent, log.parent, result_dir):
        folder.mkdir(parents=True, exist_ok=args.resume)
    if not args.resume:
        write_json(result_dir / "run_summary.json", summary)
    standard = model_config["standardisation"]
    reader = Pod5SignalReader(
        signal_shift=float(standard["mean"]), signal_scale=float(standard["stdev"])
    )
    features = {}
    for split, group in selected.items():
        loader = DataLoader(
            PolyTrainingDataset(group, reader),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=partial(collate_poly_training_examples, supervise_status=True),
        )
        features[split] = extract_status_features(
            model,
            loader,
            [r.nominal_length_nt for r in group],
            device=device,
            description=f"Frozen pools: {split.value}",
        )
    tensor_weights = torch.tensor(class_weights, device=device)
    if not args.resume:
        baseline = status_metrics(
            model, features[ManifestSplit.VALIDATION], tensor_weights, device=device
        )
        write_json(result_dir / "validation_baseline.json", baseline)
    best_rows = [row for row in history if row["split"] == "validation"]
    best = max(best_rows, key=status_selection_key) if best_rows else None
    for epoch in range(start_epoch + 1, args.epochs + 1):
        step += train_status_epoch(
            model,
            features[ManifestSplit.TRAIN],
            optimizer,
            tensor_weights,
            device=device,
            batch_size=args.batch_size,
            seed=args.random_seed + epoch,
            gradient_clip_norm=args.gradient_clip_norm,
        )
        rows = []
        for split in (ManifestSplit.TRAIN, ManifestSplit.VALIDATION):
            metrics = status_metrics(
                model, features[split], tensor_weights, device=device
            )
            rows.append(
                {
                    "epoch": epoch,
                    "split": split.value,
                    "global_step": step,
                    "loss_config": asdict(weights),
                    "joint_config": calibration,
                    **metrics,
                }
            )
        # Generic full checkpoints remain consumable by the existing evaluator.
        save_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=step,
            context=context,
            data_root=data,
            work_root=work,
        )
        # Per-epoch snapshots contain only the two classifiers and optimizer.
        adapter = checkpoint.with_name(f"status.epoch_{epoch:04d}.pt")
        if adapter.exists() or adapter.is_symlink():
            raise FileExistsError(f"status snapshot already exists: {adapter}")
        torch.save(
            {
                "status_state_dict": {
                    name: p.detach().cpu().clone()
                    for name, p in model.named_parameters()
                    if p.requires_grad
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "global_step": step,
                "random_seed": args.random_seed,
                "joint_config": calibration,
                "loss_config": asdict(weights),
                "manifest_schema_version": context.manifest_schema_version,
                "git_commit": context.git_commit,
                "bonito_commit": context.bonito_commit,
                "pytorch_version": torch.__version__,
            },
            adapter,
        )
        validation = rows[1]
        if best is None or status_selection_key(validation) > status_selection_key(
            best
        ):
            best = validation
            temporary = best_path.with_suffix(".pt.tmp")
            shutil.copyfile(checkpoint, temporary)
            temporary.replace(best_path)
            write_json(
                result_dir / "best_checkpoint.json",
                {"checkpoint": str(best_path), **best},
            )
        with log.open("a") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "split": "validation",
                    **{
                        key: validation[key]
                        for key in (
                            "zero_nt_recall",
                            "twenty_nt_false_zero_fraction",
                            "twenty_nt_present_recall",
                            "selection_score",
                        )
                    },
                }
            ),
            flush=True,
        )
    frozen_unchanged = all(
        torch.equal(tensor.detach().cpu(), reference_state[name])
        for name, tensor in model.state_dict().items()
        if name not in trainable_names
    )
    if not frozen_unchanged:
        raise RuntimeError("a frozen parameter or buffer changed")
    summary.update(
        finished_utc=datetime.now(UTC).isoformat(),
        frozen_state_unchanged=True,
        completed_epochs=args.epochs,
        global_step=step,
        best_validation=best,
        test_used=False,
    )
    write_json(result_dir / "completion.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """Fail explicitly rather than dropping unreadable labeled candidates."""
    try:
        run(build_parser().parse_args(argv))
    except (OSError, ValueError, KeyError, RuntimeError, TypeError) as error:
        print(f"Status calibration failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
