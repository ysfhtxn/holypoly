from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from holypoly.cli import build_parser


def test_package_initializers_and_cli_keep_dependencies_isolated() -> None:
    project_root = Path(__file__).resolve().parents[3]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import holypoly.cli; import holypoly.datasets; "
            "import holypoly.models; import holypoly.training; "
            "assert not {'torch', 'pod5', 'pysam', 'bonito'}.intersection(sys.modules)",
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_cli_preserves_doctor_and_train_commands() -> None:
    parser = build_parser()
    assert parser.parse_args(["doctor"]).command == "doctor"
    assert parser.parse_args(["train", "--dry-run"]).command == "train"
