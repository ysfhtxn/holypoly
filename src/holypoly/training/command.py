"""Legacy frozen-encoder implementation of ``holypoly train``."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import toml
import torch
from torch.utils.data import DataLoader

from holypoly.models.length_head_config import LENGTH_HEAD_MODES
from holypoly.models.model import HolypolyModel, load_bonito_v5_model
from holypoly.training.data import (
    ManifestSplit,
    Pod5SignalReader,
    PolyTrainingDataset,
    collate_poly_training_examples,
    load_training_manifest,
)
from holypoly.training.engine import (
    CheckpointContext,
    append_loss_log,
    create_optimizer,
    load_checkpoint,
    require_single_cuda_device,
    save_checkpoint,
    set_random_seed,
    train_one_epoch,
    validate_one_epoch,
)
from holypoly.training.loss_config import LENGTH_LOSS_MODES
from holypoly.training.losses import PolyAuxiliaryLoss, PolyLossWeights
from holypoly.training.run_validation import (
    git_commit,
    manifest_summary,
    single_manifest_version,
    validate_epoch_checkpoint_paths,
    validate_manifest_sources,
    validate_output_state,
    validate_training_paths,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the isolated training-process argument parser."""
    parser = argparse.ArgumentParser(prog="holypoly train")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--loss-log", type=Path, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--chunk-size-samples", type=int, required=True)
    parser.add_argument("--hidden-features", type=int, required=True)
    parser.add_argument("--length-head-mode", choices=LENGTH_HEAD_MODES, default="mean")
    parser.add_argument("--data-order-seed", type=int)
    parser.add_argument("--dropout", type=float, required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--weight-decay", type=float, required=True)
    parser.add_argument("--gradient-clip-norm", type=float, required=True)
    parser.add_argument("--random-seed", type=int, required=True)
    parser.add_argument("--num-workers", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--lambda-location", type=float, required=True)
    parser.add_argument("--lambda-length", type=float, required=True)
    parser.add_argument("--lambda-status", type=float, required=True)
    parser.add_argument(
        "--length-loss", choices=LENGTH_LOSS_MODES, default="log_smooth_l1"
    )
    parser.add_argument("--huber-delta-nt", type=float, default=10.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the manifest, then train both Poly heads on one GPU."""
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 2


def run(args: argparse.Namespace) -> int:
    """Execute one complete frozen-encoder training run."""
    _validate_arguments(args)
    weights = PolyLossWeights(
        lambda_location=args.lambda_location,
        lambda_length=args.lambda_length,
        lambda_status=args.lambda_status,
        length_loss=args.length_loss,
        huber_delta_nt=args.huber_delta_nt,
    )
    data_root = args.data_root.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    model_directory = args.model_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    loss_log_path = args.loss_log.expanduser().resolve()
    validate_training_paths(
        data_root=data_root,
        work_root=work_root,
        manifest_path=manifest_path,
        model_directory=model_directory,
        checkpoint_path=checkpoint_path,
        loss_log_path=loss_log_path,
    )

    records = load_training_manifest(manifest_path)
    validate_manifest_sources(records, data_root=data_root)
    summary = manifest_summary(records, epochs=args.epochs)
    summary["length_head_mode"] = args.length_head_mode
    summary["data_order_seed"] = args.data_order_seed
    summary["loss_config"] = asdict(weights)
    print(json.dumps(summary, sort_keys=True))
    if summary["positive_length_values"] == 1:
        print(
            "Warning: all poly_present records have one training length; this is "
            "a pipeline smoke test, not meaningful length-regression training.",
            file=sys.stderr,
        )
    if args.dry_run:
        print(
            "Dry run complete; no POD5 signal, model weights, or outputs were written."
        )
        return 0

    validate_output_state(
        checkpoint_path=checkpoint_path,
        loss_log_path=loss_log_path,
        resume=args.resume,
        overwrite=args.overwrite,
    )
    if args.overwrite and not args.resume and loss_log_path.exists():
        loss_log_path.write_text("", encoding="utf-8")

    device = torch.device(args.device)
    require_single_cuda_device(device)
    set_random_seed(args.random_seed)
    model_config = toml.load(model_directory / "config.toml")
    standardisation = model_config.get("standardisation")
    if not isinstance(standardisation, dict):
        raise ValueError("Bonito config must define [standardisation]")
    reader = Pod5SignalReader(
        signal_shift=float(standardisation["mean"]),
        signal_scale=float(standardisation["stdev"]),
    )
    train_dataset = PolyTrainingDataset(
        records,
        reader,
        split=ManifestSplit.TRAIN,
        chunk_size_samples=args.chunk_size_samples,
    )
    validation_dataset = PolyTrainingDataset(
        records,
        reader,
        split=ManifestSplit.VALIDATION,
        chunk_size_samples=args.chunk_size_samples,
    )
    shuffle_generator = torch.Generator() if args.data_order_seed is not None else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_generator,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_poly_training_examples,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_poly_training_examples,
    )

    basecaller = load_bonito_v5_model(model_directory, device=str(device))
    model = HolypolyModel(
        basecaller,
        hidden_features=args.hidden_features,
        length_head_mode=args.length_head_mode,
        dropout=args.dropout,
    ).to(device)
    criterion = PolyAuxiliaryLoss(weights)
    optimizer = create_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    start_epoch = 0
    global_step = 0
    if args.resume:
        resume_state = load_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            device=device,
            data_order_seed=args.data_order_seed,
            loss_weights=weights,
        )
        start_epoch = resume_state.epoch
        global_step = resume_state.global_step
        if start_epoch >= args.epochs:
            raise ValueError(
                f"checkpoint epoch {start_epoch} already reaches target "
                f"epochs {args.epochs}"
            )

    validate_epoch_checkpoint_paths(
        checkpoint_path,
        start_epoch=start_epoch,
        epochs=args.epochs,
        data_root=data_root,
        work_root=work_root,
        protected_paths=(manifest_path, loss_log_path),
    )

    project_root = Path(__file__).resolve().parents[3]
    context = CheckpointContext(
        random_seed=args.random_seed,
        bonito_commit=git_commit(project_root / "src" / "bonito"),
        model_config=model_config,
        manifest_schema_version=single_manifest_version(records),
        loss_weights=weights,
        git_commit=git_commit(project_root),
        data_order_seed=args.data_order_seed,
    )
    for epoch in range(start_epoch + 1, args.epochs + 1):
        if shuffle_generator is not None:
            shuffle_generator.manual_seed(args.data_order_seed + epoch)
        train_result = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            gradient_clip_norm=args.gradient_clip_norm,
            global_step=global_step,
            progress_desc=f"Epoch {epoch}/{args.epochs} train",
        )
        global_step = train_result.global_step
        validation_result = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device=device,
            global_step=global_step,
            progress_desc=f"Epoch {epoch}/{args.epochs} validation",
        )
        append_loss_log(
            loss_log_path,
            train_result,
            epoch=epoch,
            split="train",
            loss_weights=weights,
        )
        append_loss_log(
            loss_log_path,
            validation_result,
            epoch=epoch,
            split="validation",
            loss_weights=weights,
        )
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            context=context,
            data_root=data_root,
            work_root=work_root,
            archive_epoch=True,
        )
        print(
            json.dumps(
                {
                    "epoch": epoch,
                    "train": train_result.losses,
                    "validation": validation_result.losses,
                    "validation_length_mae_nt": validation_result.length_mae_nt,
                },
                sort_keys=True,
            )
        )
    return 0


def _validate_arguments(args: argparse.Namespace) -> None:
    if args.data_order_seed is not None and args.data_order_seed < 0:
        raise ValueError("data_order_seed must be non-negative")
    if args.device != "cuda:0":
        raise ValueError(
            "device must be cuda:0; select the physical GPU with CUDA_VISIBLE_DEVICES"
        )
    for name in ("epochs", "batch_size", "chunk_size_samples", "hidden_features"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be greater than zero")
    if args.chunk_size_samples % 12 != 0:
        raise ValueError("chunk_size_samples must be divisible by 12")
    if not 0 <= args.dropout < 1:
        raise ValueError("dropout must satisfy 0 <= dropout < 1")
    for name in ("learning_rate", "gradient_clip_norm"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be greater than zero")
    for name in (
        "weight_decay",
        "random_seed",
        "num_workers",
        "lambda_location",
        "lambda_length",
        "lambda_status",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name} must be non-negative")
    if args.resume and args.overwrite:
        raise ValueError("resume and overwrite cannot be used together")


if __name__ == "__main__":
    raise SystemExit(main())
