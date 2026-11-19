"""Resolve and verify completed dataset directories for the joint launcher."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from holypoly.core.config import load_config
from holypoly.datasets.prepare_sequence import sha256


@dataclass(frozen=True)
class _DatasetLayout:
    """Summary and size fields used by one dataset builder."""

    workflow: str
    summary_name: str
    size_key: str
    raw_summary_name: str = ""
    final_subdirectory: str = ""


_LAYOUTS = (
    _DatasetLayout("head-poly-dataset", "dataset.json", "chunk_size_samples"),
    _DatasetLayout(
        "strict-poly-candidate-v1",
        "strict_dataset.json",
        "model_input_samples",
        "raw_strict_dataset.json",
        "strict",
    ),
    _DatasetLayout(
        "head-free-platform-qc-v1",
        "head_free_qc_dataset.json",
        "model_input_samples",
        "head_free_qc_raw_dataset.json",
        "head_free_qc",
    ),
)


def _has_entry(path: Path) -> bool:
    """Recognize broken links so an invalid summary cannot be bypassed."""
    return path.exists() or path.is_symlink()


def _dataset_file(directory: Path, relative: str, data_root: Path) -> Path:
    """Locate a file under its dataset directory before opening it."""
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory) or path.is_relative_to(data_root):
        raise ValueError(f"unsafe dataset file path: {directory / relative}")
    if not path.is_file():
        raise FileNotFoundError(f"missing dataset file: {path}")
    return path


def _read_summary(
    directory: Path, filename: str, workflow: str, data_root: Path
) -> dict[str, Any]:
    """Read the expected summary and require a completed build."""
    path = _dataset_file(directory, filename, data_root)
    summary = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(summary, dict) or summary.get("workflow") != workflow:
        raise ValueError(f"expected workflow {workflow} in {path}")
    if summary.get("status") != "complete":
        raise ValueError(f"dataset is incomplete ({summary.get('status')}); see {path}")
    return summary


def resolve_dataset_bundle(
    directory: Path,
    *,
    work_root: Path,
    data_root: Path,
    expected_workflow: str = "",
) -> tuple[str, str, int]:
    """Return the workflow, relative training subdirectory, and padded input size."""
    work = work_root.expanduser().resolve()
    data = data_root.expanduser().resolve()
    project = Path(__file__).resolve().parents[3]
    if not work.is_dir() or not data.is_dir():
        raise ValueError("configure existing data_root and work_root directories")
    if work.is_relative_to(data) or work.is_relative_to(project):
        raise ValueError("work_root must be external and outside raw data")
    dataset = (work / directory).resolve()
    if (
        dataset == work
        or not dataset.is_relative_to(work)
        or dataset.is_relative_to(data)
    ):
        raise ValueError("dataset must be inside work_root and outside raw data")
    if not dataset.is_dir():
        raise FileNotFoundError(f"dataset directory does not exist: {dataset}")
    if dataset.name == "metadata_pool" and any(
        layout.raw_summary_name and _has_entry(dataset.parent / layout.raw_summary_name)
        for layout in _LAYOUTS
    ):
        raise ValueError(
            "use the completed QC dataset directory instead of metadata_pool"
        )

    matches = [
        (layout, name)
        for layout in _LAYOUTS
        for name in (layout.summary_name, layout.raw_summary_name)
        if name and _has_entry(dataset / name)
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one recognized dataset summary in {dataset}")
    layout, entry_name = matches[0]
    if expected_workflow and layout.workflow != expected_workflow:
        raise ValueError(f"expected {expected_workflow}, found {layout.workflow}")

    final = dataset
    subdirectory = "."
    raw_directory = None
    if entry_name == layout.raw_summary_name:
        subdirectory = layout.final_subdirectory
        final = (dataset / subdirectory).resolve()
        raw_directory = dataset
    elif (
        layout.raw_summary_name
        and dataset.name == layout.final_subdirectory
        and _has_entry(dataset.parent / layout.raw_summary_name)
    ):
        raw_directory = dataset.parent

    raw_summary = None
    if raw_directory is not None:
        raw_summary = _read_summary(
            raw_directory, layout.raw_summary_name, f"{layout.workflow}-from-raw", data
        )
        if (
            final == raw_directory
            or not final.is_relative_to(raw_directory)
            or final.is_relative_to(data)
            or raw_summary.get("training_dataset_dir") != str(final)
        ):
            raise ValueError("unsafe or mismatched final QC dataset path")

    summary = _read_summary(final, layout.summary_name, layout.workflow, data)
    if layout.workflow == "head-free-platform-qc-v1":
        policy = summary.get("method_policy")
        if (
            not isinstance(policy, dict)
            or policy.get("full_head_free_input") is not True
            or policy.get("metadata_only_zero") is not True
        ):
            raise ValueError("expected full head-free inputs and metadata-only zeros")

    artifacts = summary.get("artifacts")
    raw_artifacts = raw_summary.get("artifacts") if raw_summary else None
    for name, relative in (
        ("length_manifest", "length/manifest.tsv"),
        ("sequence_manifest", "sequence/sequence_manifest.tsv"),
    ):
        path = _dataset_file(final, relative, data)
        artifact = artifacts.get(name) if isinstance(artifacts, dict) else None
        if (
            not isinstance(artifact, dict)
            or artifact.get("path") != str(path)
            or artifact.get("sha256") != sha256(path)
            or (
                raw_summary is not None
                and (
                    not isinstance(raw_artifacts, dict)
                    or raw_artifacts.get(name) != artifact
                )
            )
        ):
            raise ValueError(f"dataset manifest path or hash mismatch: {path}")

    settings = summary.get("settings")
    chunk_size = settings.get(layout.size_key) if isinstance(settings, dict) else None
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError(f"invalid {layout.size_key} in {final / layout.summary_name}")
    return layout.workflow, subdirectory, chunk_size


def main(argv: Sequence[str] | None = None) -> int:
    """Emit the shell launcher's three path-independent metadata fields."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("--expect-workflow", default="")
    args = parser.parse_args(argv)
    try:
        paths = load_config(args.config).paths
        if paths.work_root is None or paths.data_root is None:
            raise ValueError("configure data_root and work_root")
        fields = resolve_dataset_bundle(
            args.dataset_dir,
            work_root=paths.work_root,
            data_root=paths.data_root,
            expected_workflow=args.expect_workflow,
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Training dataset preflight failed: {error}", file=sys.stderr)
        return 2
    print(*fields, sep="\t")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
