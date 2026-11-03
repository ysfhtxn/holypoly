"""Check result layout and legacy resume selection without real data."""

from pathlib import Path

import pytest

from holypoly.core.result_paths import training_result_dir


def test_new_results_are_grouped_without_creating_directories(tmp_path: Path) -> None:
    expected = tmp_path / "results" / "training_evaluation" / "synthetic-run"
    assert training_result_dir(tmp_path, "synthetic-run") == expected
    assert training_result_dir(tmp_path, "synthetic-run", resume=True) == expected
    assert not expected.parent.exists()


def test_resume_uses_legacy_directory_and_rejects_ambiguous_layout(
    tmp_path: Path,
) -> None:
    legacy = tmp_path / "results" / "synthetic-run"
    legacy.mkdir(parents=True)
    marker = legacy / "best_checkpoint.json"
    marker.write_text('{"epoch": 1}\n')
    current = training_result_dir(tmp_path, "synthetic-run")
    assert current.parent.name == "training_evaluation"
    assert training_result_dir(tmp_path, "synthetic-run", resume=True) == legacy
    assert marker.read_text() == '{"epoch": 1}\n'
    current.mkdir(parents=True)
    with pytest.raises(ValueError, match="ambiguous"):
        training_result_dir(tmp_path, "synthetic-run", resume=True)


@pytest.mark.parametrize("run_id", ["", ".", "..", "../escape", "a/b", "/absolute"])
def test_result_directory_rejects_invalid_run_ids(tmp_path: Path, run_id: str) -> None:
    with pytest.raises(ValueError, match="run-id"):
        training_result_dir(tmp_path, run_id)
