"""Lightweight, read-only checks for the external holypoly data environment."""

from __future__ import annotations

import gzip
import os
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from holypoly.core.config import AppConfig


class CheckStatus(StrEnum):
    """Outcome of one doctor check."""

    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True)
class CheckResult:
    """One named environment check and its human-readable detail."""

    name: str
    status: CheckStatus
    detail: str


@dataclass(frozen=True)
class DoctorReport:
    """Complete ordered result of a doctor run."""

    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        """Return whether every required check passed."""
        return all(check.status is CheckStatus.PASS for check in self.checks)


@dataclass(frozen=True)
class FileScan:
    """Files found in one directory, or the error that prevented scanning."""

    files: tuple[Path, ...]
    error: str | None = None


def run_doctor(config: AppConfig) -> DoctorReport:
    """Inspect configured paths without modifying or fully reading external data."""
    checks: list[CheckResult] = []
    paths = config.paths

    checks.append(_check_data_root(paths.data_root))
    checks.append(_check_work_root(paths.work_root))

    pod5_dir = paths.data_path(paths.pod5_dir)
    pod5_scan = _find_files(pod5_dir, _is_pod5)
    checks.append(_check_collection("POD5 files", pod5_dir, pod5_scan))
    checks.append(
        _check_samples(
            "POD5 sample readability",
            pod5_dir,
            pod5_scan,
            config.doctor.sample_size,
        )
    )

    basecalls_dir = paths.data_path(paths.basecalls_dir)
    basecall_scan = _find_files(basecalls_dir, _is_basecall)
    checks.append(_check_collection("BAM/FASTQ files", basecalls_dir, basecall_scan))
    checks.append(
        _check_samples(
            "BAM/FASTQ sample readability",
            basecalls_dir,
            basecall_scan,
            config.doctor.sample_size,
        )
    )

    reference_path = paths.data_path(paths.reference_path)
    checks.append(_check_reference(reference_path))
    return DoctorReport(checks=tuple(checks))


def render_report(report: DoctorReport) -> str:
    """Render a doctor report for terminal output."""
    lines = ["holypoly doctor", "================"]
    for check in report.checks:
        lines.append(f"[{check.status}] {check.name}: {check.detail}")
    summary = "all checks passed" if report.ok else "one or more checks failed"
    lines.extend(["", f"Summary: {summary}"])
    return "\n".join(lines)


def _check_data_root(path: Path | None) -> CheckResult:
    if path is None:
        return CheckResult("data_root", CheckStatus.FAIL, "not configured")
    if not path.is_dir():
        return CheckResult("data_root", CheckStatus.FAIL, f"not a directory: {path}")
    if not os.access(path, os.R_OK | os.X_OK):
        return CheckResult("data_root", CheckStatus.FAIL, f"not readable: {path}")
    return CheckResult("data_root", CheckStatus.PASS, f"readable directory: {path}")


def _check_work_root(path: Path | None) -> CheckResult:
    if path is None:
        return CheckResult("work_root", CheckStatus.FAIL, "not configured")
    if not path.is_dir():
        return CheckResult("work_root", CheckStatus.FAIL, f"not a directory: {path}")
    if not os.access(path, os.W_OK | os.X_OK):
        return CheckResult("work_root", CheckStatus.FAIL, f"not writable: {path}")
    return CheckResult("work_root", CheckStatus.PASS, f"writable directory: {path}")


def _find_files(directory: Path | None, predicate: Callable[[Path], bool]) -> FileScan:
    if directory is None or not directory.is_dir():
        return FileScan(files=())
    try:
        files = tuple(
            sorted(
                path
                for path in directory.rglob("*")
                if path.is_file() and predicate(path)
            )
        )
    except OSError as error:
        return FileScan(files=(), error=f"cannot scan {directory}: {error}")
    return FileScan(files=files)


def _is_pod5(path: Path) -> bool:
    return path.suffix.lower() == ".pod5"


def _is_basecall(path: Path) -> bool:
    lower_name = path.name.lower()
    return lower_name.endswith((".bam", ".fastq", ".fq", ".fastq.gz", ".fq.gz"))


def _check_collection(name: str, directory: Path | None, scan: FileScan) -> CheckResult:
    if directory is None:
        return CheckResult(name, CheckStatus.FAIL, "data_root is not configured")
    if not directory.is_dir():
        return CheckResult(name, CheckStatus.FAIL, f"directory not found: {directory}")
    if scan.error is not None:
        return CheckResult(name, CheckStatus.FAIL, scan.error)
    if not scan.files:
        return CheckResult(name, CheckStatus.FAIL, f"0 matching files in {directory}")
    return CheckResult(
        name, CheckStatus.PASS, f"{len(scan.files)} file(s) in {directory}"
    )


def _check_samples(
    name: str, directory: Path | None, scan: FileScan, limit: int
) -> CheckResult:
    if scan.error is not None:
        return CheckResult(name, CheckStatus.FAIL, scan.error)
    if not scan.files:
        location = (
            "unconfigured data directory" if directory is None else str(directory)
        )
        return CheckResult(
            name, CheckStatus.FAIL, f"no matching files available in {location}"
        )
    sampled = scan.files[:limit]
    failures: list[str] = []
    for path in sampled:
        error = _probe_readable(path)
        if error is not None:
            failures.append(f"{path}: {error}")
    if failures:
        return CheckResult(name, CheckStatus.FAIL, "; ".join(failures))
    return CheckResult(name, CheckStatus.PASS, f"read {len(sampled)} file header(s)")


def _check_reference(path: Path | None) -> CheckResult:
    if path is None:
        return CheckResult("reference sequence", CheckStatus.FAIL, "not configured")
    if not path.is_file():
        return CheckResult("reference sequence", CheckStatus.FAIL, f"not found: {path}")
    error = _probe_readable(path)
    if error is not None:
        return CheckResult("reference sequence", CheckStatus.FAIL, f"{path}: {error}")
    return CheckResult("reference sequence", CheckStatus.PASS, f"readable file: {path}")


def _probe_readable(path: Path) -> str | None:
    """Read at most one decompressed byte and return an error message on failure."""
    try:
        if path.name.lower().endswith(".gz"):
            with gzip.open(path, "rb") as handle:
                handle.read(1)
        else:
            with path.open("rb") as handle:
                handle.read(1)
    except OSError as error:
        return str(error)
    return None
