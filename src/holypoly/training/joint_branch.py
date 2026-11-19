"""Plan and copy an isolated joint-training branch without editing its parent."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from holypoly.core.result_paths import training_result_dir
from holypoly.training.engine import epoch_checkpoint_path


@dataclass(frozen=True)
class JointBranchPlan:
    """Validated inputs and fresh destinations for a historical epoch branch."""

    checkpoint: Path
    epoch: int
    parent_run_id: str
    parent_log: Path
    parent_result: Path
    history_text: str
    copies: tuple[tuple[Path, Path], ...]
    output_checkpoint: Path
    output_log: Path
    output_result: Path
    copy_bytes: int

    def provenance(self, global_step: int) -> dict[str, Any]:
        """Describe the fork point independently of later learning-rate changes."""
        return {
            "checkpoint": str(self.checkpoint),
            "parent_run_id": self.parent_run_id,
            "epoch": self.epoch,
            "global_step": global_step,
            "inherited_epochs": self.epoch,
            "copy_bytes": self.copy_bytes,
        }


def plan_joint_branch(
    source_checkpoint: Path,
    *,
    output_checkpoint: Path,
    output_log: Path,
    output_result: Path,
    work_root: Path,
    data_root: Path,
    project_root: Path,
) -> JointBranchPlan:
    """Check archive layout, containment and copy size without creating files."""

    def safe(path: Path) -> Path:
        resolved = path.resolve()
        if (
            not resolved.is_relative_to(work_root)
            or resolved.is_relative_to(data_root)
            or resolved.is_relative_to(project_root)
        ):
            raise ValueError(f"unsafe joint branch path: {path}")
        return resolved

    source = safe(work_root / source_checkpoint)
    match = re.fullmatch(r"heads\.epoch_([0-9]{4,})\.pt", source.name)
    if (
        match is None
        or source.parent.parent != (work_root / "models").resolve()
        or not source.is_file()
    ):
        raise ValueError(
            "resume-from requires an archived models/<run>/heads.epoch_NNNN.pt"
        )
    epoch = int(match[1])
    if epoch < 1:
        raise ValueError("resume-from epoch must be positive")
    parent_id = source.parent.name
    parent_log = safe(work_root / "logs" / parent_id / "train.jsonl")
    parent_result = safe(training_result_dir(work_root, parent_id, resume=True))
    for destination in (output_checkpoint.parent, output_log.parent, output_result):
        safe(destination)
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"choose a new run-id; output exists: {destination}")
    copies = [
        (
            safe(epoch_checkpoint_path(source.parent / "heads.pt", index)),
            epoch_checkpoint_path(output_checkpoint, index),
        )
        for index in range(1, epoch + 1)
    ]
    copies.extend(
        [
            (source, output_checkpoint),
            (
                safe(parent_result / "basecalling_validation_baseline.json"),
                output_result / "basecalling_validation_baseline.json",
            ),
        ]
    )
    for path in (parent_log, parent_result / "run_summary.json"):
        if not safe(path).is_file():
            raise FileNotFoundError(f"joint branch source does not exist: {path}")
    for origin, destination in copies:
        if not origin.is_file():
            raise FileNotFoundError(f"joint branch source does not exist: {origin}")
        safe(destination)
    history_text = "".join(
        parent_log.read_text().splitlines(keepends=True)[: 2 * epoch]
    )
    if history_text and not history_text.endswith("\n"):
        history_text += "\n"
    return JointBranchPlan(
        source,
        epoch,
        parent_id,
        parent_log,
        parent_result,
        history_text,
        tuple(copies),
        output_checkpoint,
        output_log,
        output_result,
        sum(origin.stat().st_size for origin, _ in copies),
    )


def materialize_joint_branch(plan: JointBranchPlan, summary: dict[str, Any]) -> None:
    """Copy only the selected history into a new run, preserving source bytes."""
    # Recheck destinations immediately before mutation; never merge into a run.
    for path in (
        plan.output_checkpoint.parent,
        plan.output_log.parent,
        plan.output_result,
    ):
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"choose a new run-id; output exists: {path}")
    for path in (
        plan.output_checkpoint.parent,
        plan.output_log.parent,
        plan.output_result,
    ):
        path.mkdir(parents=True, exist_ok=False)
    for source, destination in plan.copies:
        with source.open("rb") as reader, destination.open("xb") as writer:
            shutil.copyfileobj(reader, writer)
    with plan.output_log.open("x") as stream:
        stream.write(plan.history_text)
    with (plan.output_result / "run_summary.json").open("x") as stream:
        stream.write(json.dumps(summary, indent=2) + "\n")
