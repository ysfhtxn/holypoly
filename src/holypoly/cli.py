"""Command-line interface for holypoly."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from holypoly.core.config import ConfigError, load_config
from holypoly.core.doctor import render_report, run_doctor
from holypoly.training.launcher import (
    TrainingLaunchError,
    apply_training_overrides,
    launch_training,
    render_launch_error,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the holypoly argument parser."""
    parser = argparse.ArgumentParser(
        prog="holypoly",
        description="Nanopore poly(A/T) data preparation utilities.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    doctor_parser = subparsers.add_parser(
        "doctor", help="Check the configured external data environment."
    )
    doctor_parser.add_argument(
        "--config",
        type=Path,
        help="Local YAML configuration layered over configs/default.yaml.",
    )
    doctor_parser.add_argument(
        "--data-root", type=Path, help="Override the read-only data root."
    )
    doctor_parser.add_argument(
        "--work-root", type=Path, help="Override the writable work root."
    )
    train_parser = subparsers.add_parser(
        "train", help="Train frozen-encoder Poly location and length heads."
    )
    train_parser.add_argument(
        "--config",
        type=Path,
        help="Local YAML configuration layered over configs/default.yaml.",
    )
    train_parser.add_argument(
        "--data-root", type=Path, help="Override the read-only data root."
    )
    train_parser.add_argument(
        "--work-root", type=Path, help="Override the writable work root."
    )
    train_parser.add_argument("--manifest", type=Path)
    train_parser.add_argument("--model-dir", type=Path)
    train_parser.add_argument("--checkpoint", type=Path)
    train_parser.add_argument("--loss-log", type=Path)
    train_parser.add_argument("--epochs", type=int)
    train_parser.add_argument("--batch-size", type=int)
    train_parser.add_argument("--chunk-size-samples", type=int)
    train_parser.add_argument("--hidden-features", type=int)
    train_parser.add_argument("--dropout", type=float)
    train_parser.add_argument("--learning-rate", type=float)
    train_parser.add_argument("--weight-decay", type=float)
    train_parser.add_argument("--gradient-clip-norm", type=float)
    train_parser.add_argument("--random-seed", type=int)
    train_parser.add_argument("--num-workers", type=int)
    train_parser.add_argument("--device", choices=("cuda:0",))
    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate paths and manifest without reading POD5 signal or model weights."
        ),
    )
    train_parser.add_argument(
        "--resume", action="store_true", help="Resume from the configured checkpoint."
    )
    train_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow a fresh run to replace existing training outputs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the holypoly command-line interface."""
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        try:
            config = load_config(
                args.config,
                cli_data_root=args.data_root,
                cli_work_root=args.work_root,
            )
        except ConfigError as error:
            print(f"Configuration error: {error}", file=sys.stderr)
            return 2
        report = run_doctor(config)
        print(render_report(report))
        return 0 if report.ok else 1
    if args.command == "train":
        try:
            config = load_config(
                args.config,
                cli_data_root=args.data_root,
                cli_work_root=args.work_root,
            )
            training = apply_training_overrides(
                config.training,
                manifest_path=args.manifest,
                bonito_model=args.model_dir,
                checkpoint_path=args.checkpoint,
                loss_log_path=args.loss_log,
                epochs=args.epochs,
                batch_size=args.batch_size,
                chunk_size_samples=args.chunk_size_samples,
                hidden_features=args.hidden_features,
                dropout=args.dropout,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                gradient_clip_norm=args.gradient_clip_norm,
                random_seed=args.random_seed,
                num_workers=args.num_workers,
                device=args.device,
            )
            from dataclasses import replace

            config = replace(config, training=training)
            project_root = Path(__file__).resolve().parents[2]
            return launch_training(
                config,
                project_root=project_root,
                dry_run=args.dry_run,
                resume=args.resume,
                overwrite=args.overwrite,
            )
        except (ConfigError, TrainingLaunchError) as error:
            render_launch_error(error)
            return 2
    return 2
