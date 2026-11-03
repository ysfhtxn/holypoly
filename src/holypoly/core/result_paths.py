"""Result directory conventions with explicit support for legacy resumes."""

from __future__ import annotations

import re
from pathlib import Path


def training_result_dir(work_root: Path, run_id: str, *, resume: bool = False) -> Path:
    """Select a new evaluation directory or the existing legacy resume directory.

    This function only resolves paths; it never creates or migrates artifacts.
    Callers must validate the returned destination against their data/work roots.
    """
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise ValueError("run-id must be one directory name")
    current = work_root / "results" / "training_evaluation" / run_id
    legacy = work_root / "results" / run_id
    if resume:
        if current.exists() and legacy.exists():
            raise ValueError(
                "ambiguous training results: both new and legacy paths exist"
            )
        if legacy.exists():
            return legacy.resolve()
    return current.resolve()
