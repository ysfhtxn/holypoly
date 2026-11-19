"""Synthetic historical branches preserve parent bytes and omit later epochs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from holypoly.training import joint_command
from holypoly.training.joint_branch import (
    materialize_joint_branch,
    plan_joint_branch,
)
from tests.isolated.training.test_joint_resume import _config, _history_row


@pytest.fixture
def branch_setup(tmp_path: Path):
    """Provide a parent with three completed epochs and a branch at epoch two."""
    work = tmp_path / "work"
    data = tmp_path / "raw"
    project = tmp_path / "project"
    data.mkdir()
    project.mkdir()
    parent_models = work / "models/parent"
    parent_models.mkdir(parents=True)
    for epoch in (1, 2, 3):
        (parent_models / f"heads.epoch_{epoch:04d}.pt").write_bytes(
            f"immutable synthetic epoch {epoch}".encode()
        )
    (parent_models / "heads.pt").write_bytes(b"latest epoch three")
    log = work / "logs/parent/train.jsonl"
    log.parent.mkdir(parents=True)
    log.write_text(
        "".join(
            json.dumps(_history_row(epoch, split, _config())) + "\n"
            for epoch in (1, 2, 3)
            for split in ("train", "validation")
        )
    )
    result = work / "results/training_evaluation/parent"
    result.mkdir(parents=True)
    (result / "run_summary.json").write_text(json.dumps({"joint_config": _config()}))
    (result / "basecalling_validation_baseline.json").write_text("{}\n")
    arguments = dict(
        output_checkpoint=work / "models/child/heads.pt",
        output_log=work / "logs/child/train.jsonl",
        output_result=work / "results/training_evaluation/child",
        work_root=work,
        data_root=data,
        project_root=project,
    )
    return work, data, parent_models / "heads.epoch_0002.pt", arguments


def test_branch_copies_only_ancestor_history_and_is_independent(branch_setup) -> None:
    """All inherited snapshots remain available for best-validation selection."""
    work, _, source, arguments = branch_setup
    before = {path: path.read_bytes() for path in work.rglob("*") if path.is_file()}
    plan = plan_joint_branch(source, **arguments)
    requested = {**_config(), "head_learning_rate": 3e-4, "encoder_learning_rate": 3e-6}
    payload = {"epoch": 2, "global_step": 6, "joint_config": _config()}
    with pytest.raises(ValueError, match="log and resume checkpoint"):
        joint_command._validate_resume_metadata(
            payload,
            log=plan.parent_log,
            result_dir=plan.parent_result,
            joint_config=requested,
            target_epochs=5,
            allow_batch_size_change=True,
            recorded_utc="synthetic",
        )
    assert (
        joint_command._validate_resume_metadata(
            payload,
            log=plan.parent_log,
            result_dir=plan.parent_result,
            joint_config=requested,
            target_epochs=5,
            allow_batch_size_change=True,
            recorded_utc="synthetic",
            historical_branch=True,
        )
        == []
    )
    assert {
        path: path.read_bytes() for path in work.rglob("*") if path.is_file()
    } == before
    assert plan.copy_bytes == sum(path.stat().st_size for path, _ in plan.copies)
    summary = {"joint_config": requested, "branch_origin": plan.provenance(6)}
    materialize_joint_branch(plan, summary)
    assert all(path.read_bytes() == contents for path, contents in before.items())
    assert plan.output_checkpoint.read_bytes() == source.read_bytes()
    assert plan.output_checkpoint.stat().st_ino != source.stat().st_ino
    assert sorted(path.name for path in plan.output_checkpoint.parent.iterdir()) == [
        "heads.epoch_0001.pt",
        "heads.epoch_0002.pt",
        "heads.pt",
    ]
    rows = [json.loads(line) for line in plan.output_log.read_text().splitlines()]
    assert [row["epoch"] for row in rows] == [1, 1, 2, 2]
    # A later ordinary resume accepts the branch's lower-rate summary while the
    # starting checkpoint and inherited history still carry the parent's rates.
    assert (
        joint_command._validate_resume_metadata(
            payload,
            log=plan.output_log,
            result_dir=plan.output_result,
            joint_config=requested,
            target_epochs=5,
            allow_batch_size_change=True,
            recorded_utc="synthetic",
        )
        == []
    )
    before_repeat = {
        path: path.read_bytes() for path in work.rglob("*") if path.is_file()
    }
    with pytest.raises(FileExistsError, match="new run-id"):
        materialize_joint_branch(plan, summary)
    assert {
        path: path.read_bytes() for path in work.rglob("*") if path.is_file()
    } == before_repeat


@pytest.mark.parametrize("missing", ["snapshot", "log", "baseline", "summary"])
def test_missing_branch_inputs_fail_without_outputs(branch_setup, missing) -> None:
    work, _, source, arguments = branch_setup
    paths = {
        "snapshot": work / "models/parent/heads.epoch_0001.pt",
        "log": work / "logs/parent/train.jsonl",
        "baseline": work
        / "results/training_evaluation/parent/basecalling_validation_baseline.json",
        "summary": work / "results/training_evaluation/parent/run_summary.json",
    }
    paths[missing].unlink()
    with pytest.raises(FileNotFoundError):
        plan_joint_branch(source, **arguments)
    assert not arguments["output_checkpoint"].parent.exists()


@pytest.mark.parametrize("kind", ["latest", "outside", "raw", "existing", "symlink"])
def test_branch_rejects_unsafe_or_existing_destinations(branch_setup, kind) -> None:
    work, data, source, arguments = branch_setup
    if kind == "latest":
        source = source.parent / "heads.pt"
    elif kind == "outside":
        source = work.parent / "heads.epoch_0002.pt"
    elif kind == "raw":
        arguments["output_checkpoint"] = data / "child/heads.pt"
    elif kind == "existing":
        arguments["output_log"].parent.mkdir()
    else:
        arguments["output_checkpoint"].parent.symlink_to(data, target_is_directory=True)
    with pytest.raises((ValueError, FileExistsError)):
        plan_joint_branch(source, **arguments)


def test_branch_rejects_malformed_inherited_epoch_pair(branch_setup) -> None:
    _, _, source, arguments = branch_setup
    plan = plan_joint_branch(source, **arguments)
    rows = [json.loads(line) for line in plan.parent_log.read_text().splitlines()]
    rows[0]["joint_config"]["lambda_basecall"] = 0.5
    plan.parent_log.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="configurations differ"):
        joint_command._validate_resume_metadata(
            {"epoch": 2, "global_step": 6, "joint_config": _config()},
            log=plan.parent_log,
            result_dir=plan.parent_result,
            joint_config=_config(),
            target_epochs=5,
            allow_batch_size_change=True,
            recorded_utc="synthetic",
            historical_branch=True,
        )
