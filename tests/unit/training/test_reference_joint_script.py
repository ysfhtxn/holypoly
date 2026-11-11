"""Exercise shell orchestration with fake Python commands, no signals or GPU."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def pipeline(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    """Copy the real shell script into a synthetic project with command spies."""
    project = tmp_path / "project"
    script = project / "scripts/run_reference_joint.sh"
    script.parent.mkdir(parents=True)
    source = Path(__file__).resolve().parents[3] / "scripts/run_reference_joint.sh"
    shutil.copyfile(source, script)
    work = tmp_path / "work"
    result = work / "results/training_evaluation/existing"
    result.mkdir(parents=True)
    (result / "best_checkpoint.json").write_text(
        json.dumps({"checkpoint": "models/existing/heads.epoch_0002.pt"})
    )
    for name in ("test-best-validation/summary.json", "training-plots/old.svg"):
        path = result / name
        path.parent.mkdir(exist_ok=True)
        path.write_text("preserve old result")
    old_log = work / "logs/existing-pipeline/pipeline.log"
    old_log.parent.mkdir(parents=True)
    old_log.write_text("preserve old pipeline log")
    spy = project / "spy.py"
    spy.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "work = Path(os.environ['SYNTHETIC_WORK_ROOT'])\n"
        "with (work / 'calls.jsonl').open('a') as stream:\n"
        "    stream.write(json.dumps(args) + '\\n')\n"
        "if args[0] == '-':\n"
        "    if len(args) == 2:\n"
        "        print(work)\n"
        "    else:\n"
        "        result = work / 'results/training_evaluation' / args[2]\n"
        "        print(json.loads((result / 'best_checkpoint.json').read_text())"
        "['checkpoint'])\n"
        "elif args[:2] == ['-m', 'holypoly.training.joint_command']:\n"
        "    if '--dry-run' in args and os.environ.get('FAIL_STAGE') == 'preflight':\n"
        "        sys.exit(7)\n"
        "    print('{}')\n"
        "elif args[:2] == ['-m', 'holypoly.training.evaluate']:\n"
        "    if os.environ.get('FAIL_STAGE') == 'evaluate':\n"
        "        sys.exit(7)\n"
        "    output = work / args[args.index('--output-dir') + 1]\n"
        "    output.mkdir(parents=True)\n"
        "    (output / 'predictions.tsv').write_text('synthetic')\n"
        "elif args[0] == 'scripts/plot_training.py':\n"
        "    output = work / args[args.index('--output') + 1]\n"
        "    output.mkdir(parents=True)\n"
        "    (output / 'plot.svg').write_text('synthetic')\n"
        "else:\n"
        "    raise SystemExit('unexpected command')\n"
    )
    spy.chmod(0o755)
    for relative in (".venv/bin/python", "src/bonito/.venv/bin/python"):
        python = project / relative
        python.parent.mkdir(parents=True)
        python.symlink_to(spy)
    ignored = {
        "RESUME",
        "EXPERIMENT_ID",
        "INITIALIZE_FROM",
        "GPU_ID",
        "STATUS_SUPERVISION",
        "EPOCHS",
        "BATCH_SIZE",
        "EVAL_BATCH_SIZE",
        "USE_TIMING_REFERENCE",
        "TIMING_REFERENCE_MANIFEST",
    }
    environment = {key: val for key, val in os.environ.items() if key not in ignored}
    environment.update(SYNTHETIC_WORK_ROOT=str(work), EXPERIMENT_ID="existing")
    return script, work, environment


def run_script(
    pipeline: tuple[Path, Path, dict[str, str]], *options: str
) -> subprocess.CompletedProcess[str]:
    """Invoke the unchanged script body with synthetic command implementations."""
    script, _, environment = pipeline
    return subprocess.run(
        ["bash", str(script), "length.tsv", "sequence.tsv", *options],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def calls(work: Path) -> list[list[str]]:
    """Return recorded subprocess arguments, if preflight was reached."""
    path = work / "calls.jsonl"
    return (
        [json.loads(line) for line in path.read_text().splitlines()]
        if path.exists()
        else []
    )


@pytest.mark.parametrize("use_environment", [False, True])
def test_resume_runs_training_test_and_plots_without_overwriting(
    pipeline: tuple[Path, Path, dict[str, str]], use_environment: bool
) -> None:
    """Both resume spellings preserve old artifacts and use the selected epoch."""
    _, work, environment = pipeline
    before = {path: path.read_bytes() for path in work.rglob("*") if path.is_file()}
    environment.update(EPOCHS="10", BATCH_SIZE="64")
    if use_environment:
        environment["RESUME"] = "1"
    completed = run_script(pipeline, *([] if use_environment else ["--resume"]))
    assert completed.returncode == 0, completed.stdout + completed.stderr
    recorded = calls(work)
    training = [
        row for row in recorded if row[:2] == ["-m", "holypoly.training.joint_command"]
    ]
    assert len(training) == 2
    assert "--dry-run" in training[0] and "--dry-run" not in training[1]
    for arguments in training:
        assert "--resume" in arguments
        assert arguments[arguments.index("--epochs") + 1] == "10"
        assert arguments[arguments.index("--batch-size") + 1] == "64"
    evaluation = next(
        row for row in recorded if row[:2] == ["-m", "holypoly.training.evaluate"]
    )
    assert evaluation[evaluation.index("--checkpoint") + 1].endswith("epoch_0002.pt")
    output = work / evaluation[evaluation.index("--output-dir") + 1]
    assert output.parent.name.startswith("resume-")
    assert (output / "plots/plot.svg").is_file()
    assert (output.parent / "training-plots/plot.svg").is_file()
    assert (
        len(list((work / "logs").glob("existing-pipeline-resume-*/pipeline.log"))) == 1
    )
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize(
    "options", [("--resume", "--dry-run"), ("--dry-run", "--resume")]
)
def test_resume_dry_run_stops_after_preflight(
    pipeline: tuple[Path, Path, dict[str, str]], options: tuple[str, str]
) -> None:
    """Either flag order performs preflight only, with no output directories."""
    _, work, _ = pipeline
    before = set(work.rglob("*"))
    completed = run_script(pipeline, *options)
    assert completed.returncode == 0, completed.stderr
    assert len(calls(work)) == 1
    assert "--resume" in calls(work)[0] and "--dry-run" in calls(work)[0]
    assert set(work.rglob("*")) - before == {work / "calls.jsonl"}


@pytest.mark.parametrize("failure", ["missing-id", "initialize", "invalid-resume"])
def test_resume_configuration_errors_fail_before_python(
    pipeline: tuple[Path, Path, dict[str, str]], failure: str
) -> None:
    """Never silently create a new run or discard optimizer state on resume."""
    _, work, environment = pipeline
    environment["RESUME"] = "1"
    if failure == "missing-id":
        environment.pop("EXPERIMENT_ID")
    elif failure == "initialize":
        environment["INITIALIZE_FROM"] = "old.pt"
    else:
        environment["RESUME"] = "invalid"
    completed = run_script(pipeline)
    assert completed.returncode == 2
    assert not calls(work)


@pytest.mark.parametrize("stage", ["preflight", "evaluate"])
def test_pipeline_stops_on_failure(
    pipeline: tuple[Path, Path, dict[str, str]], stage: str
) -> None:
    """A failed preflight or test never proceeds to plot stale predictions."""
    _, work, environment = pipeline
    environment.update(RESUME="1", FAIL_STAGE=stage)
    completed = run_script(pipeline)
    assert completed.returncode == 7
    recorded = calls(work)
    assert not any(row[0] == "scripts/plot_training.py" for row in recorded)
    if stage == "preflight":
        assert len(recorded) == 1
        assert not list((work / "logs").glob("existing-pipeline-resume-*"))


def test_fresh_pipeline_retains_existing_output_layout(
    pipeline: tuple[Path, Path, dict[str, str]],
) -> None:
    """New runs retain their original paths and do not receive --resume."""
    _, work, environment = pipeline
    environment["EXPERIMENT_ID"] = "new-run"
    result = work / "results/training_evaluation/new-run"
    result.mkdir()
    (result / "best_checkpoint.json").write_text(json.dumps({"checkpoint": "best.pt"}))
    completed = run_script(pipeline)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert all("--resume" not in row for row in calls(work))
    assert (result / "test-best-validation/plots/plot.svg").is_file()
    assert (result / "training-plots/plot.svg").is_file()


@pytest.mark.parametrize(
    ("overrides", "train_batch", "eval_batch"),
    [
        ({}, "32", "4"),
        ({"BATCH_SIZE": "64"}, "64", "64"),
        ({"BATCH_SIZE": "64", "EVAL_BATCH_SIZE": "8"}, "64", "8"),
    ],
)
def test_evaluation_batch_can_change_without_changing_training(
    pipeline: tuple[Path, Path, dict[str, str]],
    overrides: dict[str, str],
    train_batch: str,
    eval_batch: str,
) -> None:
    """Preserve old defaults while allowing a smaller test-inference batch."""
    _, work, environment = pipeline
    environment.update(RESUME="1", **overrides)
    completed = run_script(pipeline)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    recorded = calls(work)
    training = next(
        row for row in recorded if row[:2] == ["-m", "holypoly.training.joint_command"]
    )
    evaluation = next(
        row for row in recorded if row[:2] == ["-m", "holypoly.training.evaluate"]
    )
    assert training[training.index("--batch-size") + 1] == train_batch
    assert evaluation[evaluation.index("--batch-size") + 1] == eval_batch


def test_invalid_evaluation_batch_fails_before_training(
    pipeline: tuple[Path, Path, dict[str, str]],
) -> None:
    """Reject invalid test settings before any training or output creation."""
    _, work, environment = pipeline
    environment["EVAL_BATCH_SIZE"] = "0"
    completed = run_script(pipeline)
    assert completed.returncode == 2
    assert not calls(work)


@pytest.mark.parametrize("enabled", [False, True])
def test_timing_option_reaches_training_and_matching_evaluation(
    pipeline: tuple[Path, Path, dict[str, str]],
    enabled: bool,
) -> None:
    """Train and test use the same opt-in table; default commands omit both."""
    _, work, environment = pipeline
    environment["RESUME"] = "1"
    if enabled:
        environment.update(
            USE_TIMING_REFERENCE="1", TIMING_REFERENCE_MANIFEST="timing.tsv"
        )
    completed = run_script(pipeline)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    for arguments in calls(work):
        if arguments[:2] not in (
            ["-m", "holypoly.training.joint_command"],
            ["-m", "holypoly.training.evaluate"],
        ):
            continue
        assert ("--timing-reference-manifest" in arguments) == enabled
        if enabled:
            assert (
                arguments[arguments.index("--timing-reference-manifest") + 1]
                == "timing.tsv"
            )
        if arguments[1].endswith("joint_command"):
            assert ("--use-timing-reference" in arguments) == enabled


@pytest.mark.parametrize("setting", ["invalid", "missing-table", "stray-table"])
def test_invalid_timing_configuration_stops_before_training(
    pipeline: tuple[Path, Path, dict[str, str]],
    setting: str,
) -> None:
    """Do not read inputs or launch training for contradictory timing settings."""
    _, work, environment = pipeline
    if setting == "invalid":
        environment["USE_TIMING_REFERENCE"] = "yes"
    elif setting == "missing-table":
        environment["USE_TIMING_REFERENCE"] = "1"
    else:
        environment["TIMING_REFERENCE_MANIFEST"] = "unused.tsv"
    assert run_script(pipeline).returncode != 0
    assert not calls(work)
