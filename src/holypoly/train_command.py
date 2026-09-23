"""Bonito-environment implementation of ``holypoly train``."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path

import toml
import torch
from torch.utils.data import DataLoader

from holypoly.losses import PolyAuxiliaryLoss, PolyLossWeights
from holypoly.model import HolypolyModel, load_bonito_v5_model
from holypoly.training import (
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
from holypoly.training_data import (
    ManifestSplit,
    Pod5SignalReader,
    PolyTrainingDataset,
    PolyTrainingRecord,
    collate_poly_training_examples,
    load_training_manifest,
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Validate the manifest, then train both Poly heads on one GPU."""
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (FileNotFoundError, RuntimeError, TypeError, ValueError) as error:
        print(f"Training failed: {error}", file=sys.stderr)
        return 2


def run(args: argparse.Namespace) -> int:
    """Execute one complete frozen-encoder training run."""
    _validate_arguments(args)
    data_root = args.data_root.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    manifest_path = args.manifest.expanduser().resolve()
    model_directory = args.model_dir.expanduser().resolve()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    loss_log_path = args.loss_log.expanduser().resolve()
    _validate_paths(
        data_root=data_root,
        work_root=work_root,
        manifest_path=manifest_path,
        model_directory=model_directory,
        checkpoint_path=checkpoint_path,
        loss_log_path=loss_log_path,
    )

    records = load_training_manifest(manifest_path)
    _validate_manifest_sources(records, data_root=data_root)
    summary = _manifest_summary(records, epochs=args.epochs)
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

    _validate_output_state(
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
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
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
        dropout=args.dropout,
    ).to(device)
    weights = PolyLossWeights(
        lambda_location=args.lambda_location,
        lambda_length=args.lambda_length,
        lambda_status=args.lambda_status,
    )
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
        )
        start_epoch = resume_state.epoch
        global_step = resume_state.global_step
        if start_epoch >= args.epochs:
            raise ValueError(
                f"checkpoint epoch {start_epoch} already reaches target "
                f"epochs {args.epochs}"
            )

    project_root = Path(__file__).resolve().parents[2]
    context = CheckpointContext(
        random_seed=args.random_seed,
        bonito_commit=_git_commit(project_root / "src" / "bonito"),
        model_config=model_config,
        manifest_schema_version=_single_manifest_version(records),
        loss_weights=weights,
        git_commit=_git_commit(project_root),
    )
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_result = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            gradient_clip_norm=args.gradient_clip_norm,
            global_step=global_step,
        )
        global_step = train_result.global_step
        validation_result = validate_one_epoch(
            model,
            validation_loader,
            criterion,
            device=device,
            global_step=global_step,
        )
        append_loss_log(loss_log_path, train_result, epoch=epoch, split="train")
        append_loss_log(
            loss_log_path,
            validation_result,
            epoch=epoch,
            split="validation",
        )
        save_checkpoint(
            checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            global_step=global_step,
            context=context,
            data_root=data_root,
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


def _validate_paths(
    *,
    data_root: Path,
    work_root: Path,
    manifest_path: Path,
    model_directory: Path,
    checkpoint_path: Path,
    loss_log_path: Path,
) -> None:
    if not data_root.is_dir():
        raise FileNotFoundError(f"data_root does not exist: {data_root}")
    if not work_root.is_dir():
        raise FileNotFoundError(f"work_root does not exist: {work_root}")
    if data_root == work_root or work_root.is_relative_to(data_root):
        raise ValueError("work_root must be outside data_root")
    if not manifest_path.is_file():
        raise FileNotFoundError(f"training manifest does not exist: {manifest_path}")
    if not model_directory.is_dir():
        raise FileNotFoundError(
            f"Bonito model directory does not exist: {model_directory}"
        )
    if not (model_directory / "config.toml").is_file():
        raise FileNotFoundError(
            f"Bonito model config does not exist: {model_directory / 'config.toml'}"
        )
    for output_path, label in (
        (checkpoint_path, "checkpoint"),
        (loss_log_path, "loss log"),
    ):
        if output_path == data_root or output_path.is_relative_to(data_root):
            raise ValueError(f"{label} must not be inside data_root: {output_path}")
        if output_path != work_root and not output_path.is_relative_to(work_root):
            raise ValueError(f"{label} must be inside work_root: {output_path}")


def _validate_manifest_sources(
    records: Sequence[PolyTrainingRecord], *, data_root: Path
) -> None:
    split_read_ids: dict[ManifestSplit, set[str]] = {
        split: set() for split in ManifestSplit
    }
    for record in records:
        pod5_path = record.pod5_path.expanduser().resolve()
        if pod5_path != data_root and not pod5_path.is_relative_to(data_root):
            raise ValueError(
                f"manifest POD5 path is outside data_root for read "
                f"{record.read_id!r}: {pod5_path}"
            )
        if not pod5_path.is_file():
            raise FileNotFoundError(
                f"manifest POD5 file does not exist for read {record.read_id!r}: "
                f"{pod5_path}"
            )
        split_read_ids[record.split].add(record.read_id)
    overlap = (
        split_read_ids[ManifestSplit.TRAIN] & split_read_ids[ManifestSplit.VALIDATION]
    )
    if overlap:
        raise ValueError(
            "train and validation splits share read_id values: "
            + ", ".join(sorted(overlap))
        )
    if not split_read_ids[ManifestSplit.TRAIN]:
        raise ValueError("manifest contains no training reads")
    if not split_read_ids[ManifestSplit.VALIDATION]:
        raise ValueError("manifest contains no validation reads")


def _manifest_summary(
    records: Sequence[PolyTrainingRecord], *, epochs: int
) -> dict[str, object]:
    split_counts = Counter(record.split.value for record in records)
    status_counts = Counter(record.label_status.value for record in records)
    poly_counts = Counter(record.poly_type.value for record in records)
    active_records = [
        record
        for record in records
        if record.split in {ManifestSplit.TRAIN, ManifestSplit.VALIDATION}
    ]
    raw_samples_per_epoch = sum(record.raw_signal_length for record in active_records)
    positive_lengths = {
        record.training_length_nt
        for record in active_records
        if math.isfinite(record.training_length_nt) and record.training_length_nt > 0
    }
    return {
        "records": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "poly_type_counts": dict(sorted(poly_counts.items())),
        "pod5_files": len({record.pod5_path for record in active_records}),
        "raw_samples_per_epoch": raw_samples_per_epoch,
        "approx_float32_gib_per_epoch": raw_samples_per_epoch * 4 / 1024**3,
        "epochs": epochs,
        "positive_length_values": len(positive_lengths),
    }


def _validate_output_state(
    *,
    checkpoint_path: Path,
    loss_log_path: Path,
    resume: bool,
    overwrite: bool,
) -> None:
    if resume:
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"cannot resume because checkpoint does not exist: {checkpoint_path}"
            )
        return
    existing = [path for path in (checkpoint_path, loss_log_path) if path.exists()]
    if existing and not overwrite:
        raise ValueError(
            "training output already exists; use --resume or --overwrite: "
            + ", ".join(str(path) for path in existing)
        )


def _single_manifest_version(records: Sequence[PolyTrainingRecord]) -> str:
    versions = {record.manifest_version for record in records}
    if len(versions) != 1:
        raise ValueError(
            "training manifest must contain exactly one manifest_version, found: "
            + ", ".join(sorted(versions))
        )
    return next(iter(versions))


def _git_commit(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"cannot determine Git commit for {repository}: {completed.stderr.strip()}"
        )
    return completed.stdout.strip()


if __name__ == "__main__":
    raise SystemExit(main())
