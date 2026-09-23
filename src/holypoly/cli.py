"""Command-line interface for holypoly."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from holypoly.config import ConfigError, load_config
from holypoly.doctor import render_report, run_doctor


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
    return 2
