from __future__ import annotations

import os
import subprocess
from pathlib import Path


def test_poly_heads_in_isolated_bonito_environment() -> None:
    """Run Torch-dependent unit tests in Bonito's isolated environment."""
    project_root = Path(__file__).resolve().parents[1]
    python_path = project_root / "src" / "bonito" / ".venv" / "bin" / "python"
    assert python_path.is_file(), (
        "Bonito environment is missing; create src/bonito/.venv and install "
        "Bonito dependencies before running the project tests"
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(project_root / "src")
    completed = subprocess.run(
        [
            str(python_path),
            "-m",
            "pytest",
            "-q",
            "tests/torch/test_poly_heads.py",
        ],
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
