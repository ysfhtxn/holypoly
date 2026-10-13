import gzip
from collections.abc import Iterator
from pathlib import Path

import pytest

from holypoly.core.config import AppConfig, DoctorConfig, PathsConfig
from holypoly.core.doctor import CheckStatus, render_report, run_doctor


def _config(data_root: Path, work_root: Path, sample_size: int = 2) -> AppConfig:
    return AppConfig(
        paths=PathsConfig(
            data_root=data_root,
            work_root=work_root,
            pod5_dir=Path("pod5"),
            basecalls_dir=Path("basecalls"),
            reference_path=Path("reference.fa"),
            processed_dir=Path("processed"),
            metadata_dir=Path("metadata"),
            results_dir=Path("results"),
            models_dir=Path("models"),
            logs_dir=Path("logs"),
            cache_dir=Path("cache"),
        ),
        doctor=DoctorConfig(sample_size=sample_size),
    )


def test_doctor_passes_for_readable_mock_files(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    work_root = tmp_path / "work"
    pod5_dir = data_root / "pod5"
    basecalls_dir = data_root / "basecalls"
    pod5_dir.mkdir(parents=True)
    basecalls_dir.mkdir()
    work_root.mkdir()
    (pod5_dir / "read.pod5").write_bytes(b"POD5")
    (basecalls_dir / "reads.bam").write_bytes(b"BAM")
    with gzip.open(basecalls_dir / "reads.fastq.gz", "wb") as handle:
        handle.write(b"@read\nACGT\n+\n!!!!\n")
    (data_root / "reference.fa").write_text(">ref\nACGT\n", encoding="utf-8")

    report = run_doctor(_config(data_root, work_root))

    assert report.ok
    assert all(check.status is CheckStatus.PASS for check in report.checks)
    rendered = render_report(report)
    assert "1 file(s)" in rendered
    assert "2 file(s)" in rendered
    assert "all checks passed" in rendered


def test_doctor_reports_missing_inputs_without_creating_them(tmp_path: Path) -> None:
    data_root = tmp_path / "missing-data"
    work_root = tmp_path / "missing-work"

    report = run_doctor(_config(data_root, work_root))

    assert not report.ok
    assert not data_root.exists()
    assert not work_root.exists()
    rendered = render_report(report)
    assert "[FAIL] data_root" in rendered
    assert "[FAIL] work_root" in rendered
    assert "0 matching files" not in rendered
    assert "one or more checks failed" in rendered


def test_doctor_samples_only_configured_number_of_files(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    work_root = tmp_path / "work"
    pod5_dir = data_root / "pod5"
    basecalls_dir = data_root / "basecalls"
    pod5_dir.mkdir(parents=True)
    basecalls_dir.mkdir()
    work_root.mkdir()
    for index in range(4):
        (pod5_dir / f"read-{index}.pod5").write_bytes(b"x")
        (basecalls_dir / f"read-{index}.fastq").write_bytes(b"x")
    (data_root / "reference.fa").write_bytes(b">ref\nA\n")

    report = run_doctor(_config(data_root, work_root, sample_size=2))

    sample_checks = [
        check for check in report.checks if "sample readability" in check.name
    ]
    assert {check.detail for check in sample_checks} == {"read 2 file header(s)"}


def test_doctor_reports_directory_scan_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_root = tmp_path / "data"
    work_root = tmp_path / "work"
    (data_root / "pod5").mkdir(parents=True)
    (data_root / "basecalls").mkdir()
    (data_root / "reference.fa").write_bytes(b">ref\nA\n")
    work_root.mkdir()

    def fail_scan(self: Path, pattern: str) -> Iterator[Path]:
        del pattern
        raise PermissionError(f"permission denied: {self}")
        yield self

    monkeypatch.setattr(Path, "rglob", fail_scan)

    report = run_doctor(_config(data_root, work_root))

    assert not report.ok
    rendered = render_report(report)
    assert f"cannot scan {data_root / 'pod5'}" in rendered
    assert f"permission denied: {data_root / 'pod5'}" in rendered
