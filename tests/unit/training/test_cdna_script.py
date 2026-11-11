"""Check the actual shell preflight using synthetic paths and empty artifacts."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest


def test_invalid_timing_flag_fails_before_basecalling() -> None:
    """An invalid optional mode must abort before data or GPU work."""
    script = Path(__file__).resolve().parents[3] / "scripts/holypoly_predict_cdna.sh"
    result = subprocess.run(
        ["bash", str(script), "unused"],
        env={**os.environ, "CHECKPOINT": "unused.pt", "USE_TIMING_REFERENCE": "yes"},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "USE_TIMING_REFERENCE must be 0 or 1" in result.stderr


@pytest.mark.parametrize("cache_size", ["-1", "1.5", "invalid"])
def test_reader_cache_size_rejected_before_reading_inputs(cache_size: str) -> None:
    """Invalid reader cache settings fail before data/model preflight."""
    project = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["bash", str(project / "scripts/holypoly_predict_cdna.sh"), "unused"],
        env={
            **os.environ,
            "CHECKPOINT": "unused.pt",
            "POD5_READER_CACHE_SIZE": cache_size,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "POD5_READER_CACHE_SIZE" in result.stderr


@pytest.mark.parametrize("batch_size", ["0", "-1", "1.5", "invalid"])
def test_inference_batch_size_rejected_before_reading_inputs(batch_size: str) -> None:
    """Invalid Poly batch settings must fail before data/model preflight."""
    project = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        ["bash", str(project / "scripts/holypoly_predict_cdna.sh"), "unused"],
        env={
            **os.environ,
            "CHECKPOINT": "unused.pt",
            "GPU_ID": "0",
            "BATCH_SIZE": "8",
            "MAX_READS": "1",
            "INFERENCE_BATCH_SIZE": batch_size,
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "INFERENCE_BATCH_SIZE" in result.stderr


def test_script_environment_loads_real_bonito_cli_without_gpu(tmp_path: Path) -> None:
    """Catch outer-src namespace shadowing using the script's actual path export."""
    project = Path(__file__).resolve().parents[3]
    script = project / "scripts" / "holypoly_predict_cdna.sh"
    export = next(
        line
        for line in script.read_text().splitlines()
        if line.startswith("export PYTHONPATH=")
    )
    python = project / "src" / "bonito" / ".venv" / "bin" / "python"
    assert python.is_file(), "Bonito environment is missing"
    env = {
        **os.environ,
        "PYTHONPATH": str(project / "src"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
        "MPLCONFIGDIR": str(tmp_path / "matplotlib"),
    }
    result = subprocess.run(
        [
            "bash",
            "-c",
            'PROJECT_ROOT="$1"\n' + export + '\nexec "$2" -m bonito basecaller --help',
            "check-bonito-import",
            str(project),
            str(python),
        ],
        cwd=project,
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--max-reads" in result.stdout and "--no-trim" in result.stdout


@pytest.mark.parametrize(
    ("dataset", "experiment", "expected"),
    [
        ("fixed-len", "", "holypoly-fixed-len-cdna"),
        ("hek293t", "", "holypoly-hek293t-cdna"),
        ("", "", "holypoly-cdna"),
        ("hek293t", "custom-experiment", "custom-experiment"),
        ("../escape", "", None),
        ("two words", "", None),
    ],
)
def test_dataset_name_generates_task_name_without_reading_data(
    dataset: str, experiment: str, expected: str | None
) -> None:
    """Exercise the script's naming section without running its data pipeline."""
    script = Path(__file__).resolve().parents[3] / "scripts/holypoly_predict_cdna.sh"
    source = script.read_text()
    naming = source.split('DATASET_NAME="${DATASET_NAME:-}"', 1)[1].split(
        "# Plan validation", 1
    )[0]
    naming = 'DATASET_NAME="${DATASET_NAME:-}"' + naming
    env = {**os.environ, "DATASET_NAME": dataset, "EXPERIMENT_ID": experiment}
    result = subprocess.run(
        ["bash", "-c", naming + '\nprintf "%s\\n" "$EXPERIMENT_ID"'],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    if expected is None:
        assert result.returncode == 2 and "DATASET_NAME" in result.stderr
    else:
        assert result.returncode == 0, result.stderr
        if experiment:
            assert result.stdout.strip() == expected
        else:
            assert re.fullmatch(rf"{expected}-\d{{8}}-\d{{6}}-\d+\n", result.stdout)


@pytest.mark.parametrize("category", ["cdna_prediction", "training_evaluation"])
def test_cdna_plan_separates_results_and_refuses_overwrites(
    tmp_path: Path, category: str
) -> None:
    project = Path(__file__).resolve().parents[3]
    script = project / "scripts" / "holypoly_predict_cdna.sh"
    blocks = re.findall(r"<<'PY'\n(.*?)\nPY", script.read_text(), re.DOTALL)
    assert len(blocks) == 1
    data, work = tmp_path / "raw", tmp_path / "work"
    data.mkdir()
    work.mkdir()
    checkpoint = work / "synthetic.pt"
    checkpoint.touch()
    model = work / "synthetic-model"
    model.mkdir()
    (model / "config.toml").touch()
    config = tmp_path / "config.yaml"
    config.write_text(f"paths:\n  data_root: {data}\n  work_root: {work}\n")
    env = {
        key: val for key, val in os.environ.items() if not key.startswith("HOLYPOLY_")
    }
    env.update(PYTHONPATH=str(project / "src"), PYTHONDONTWRITEBYTECODE="1")
    command = [
        sys.executable,
        "-c",
        blocks[0],
        str(config),
        str(data),
        str(checkpoint),
        str(model),
        "holypoly-predict-cdna-synthetic",
        "",
        str(project),
        "",
        category,
    ]
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    output = work / "results" / category / "holypoly-predict-cdna-synthetic"
    assert Path(result.stdout.splitlines()[4]) == output
    assert not output.parent.exists()
    assert not (work / "results").exists()
    identifiers = work / "test_read_ids.txt"
    identifiers.write_text("00000000-0000-0000-0000-000000000001\n")
    command[-2] = str(identifiers)
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.splitlines()[7]) == identifiers
    identifiers.write_text("read_id\n")
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode != 0
    assert not output.parent.exists()
    identifiers.write_text("00000000-0000-0000-0000-000000000001\n" * 2)
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode != 0 and "unique" in result.stderr
    identifiers.write_text("00000000-0000-0000-0000-000000000001\n")
    command[8] = str(work / "reused.bam")
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode != 0 and "cannot combine" in result.stderr
    command[8] = ""
    output.mkdir(parents=True)
    marker = output / "pipeline.log"
    marker.write_text("existing result\n")
    result = subprocess.run(
        command, env=env, capture_output=True, text=True, check=False
    )
    assert result.returncode != 0 and "existing output" in result.stderr
    assert marker.read_text() == "existing result\n"
