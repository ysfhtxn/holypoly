"""Join existing tool outputs without conflating missing estimates and zero tails.

The BAM adapter runs in the Bonito environment. This module never loads models
or raw signals; plotting is delegated to the parent environment.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from holypoly.core.config import load_config

TOOLS = ("Holypoly", "Dorado", "PolyAnalysis")


def read_unique(path: Path) -> dict[str, dict[str, str]]:
    """Read one row per read, rejecting duplicate or empty identifiers."""
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if "read_id" not in (reader.fieldnames or []):
            raise ValueError(f"missing read_id column: {path}")
        rows: dict[str, dict[str, str]] = {}
        for row in reader:
            identifier = row["read_id"]
            if not identifier or identifier in rows:
                raise ValueError(f"empty or duplicate read_id in {path}")
            rows[identifier] = row
    if not rows:
        raise ValueError(f"empty table: {path}")
    return rows


def normalize_table(row: dict[str, str], tool: str) -> dict[str, Any]:
    """Retain native status and numeric outputs; never fill missing with zero."""
    native = row["status"]
    success = {
        "Holypoly": "candidate_predicted",
        "PolyAnalysis": "polyA_length_proxy_available",
    }[tool]
    value = None
    if native == success:
        value = float(row["predicted_length_nt"])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"invalid candidate length from {tool}")
        if tool == "Holypoly" and row.get("poly_type") != "polyA":
            raise ValueError("successful Holypoly row is not polyA")
        if tool == "PolyAnalysis" and row.get("detector_region_type") != "polyA":
            raise ValueError("successful PolyAnalysis row is not polyA")
    category = (
        "positive_length"
        if value is not None and value > 0
        else "zero_output"
        if value == 0
        else "no_candidate"
        if native == "no_candidate"
        else "other_or_ambiguous"
        if native in {"type_not_selected", "ambiguous_candidate", "other_region_polyT"}
        else "error_or_missing"
    )
    return {
        "native_status": native,
        "status_category": category,
        "reported_length_nt": value,
        "positive_length_nt": value if category == "positive_length" else None,
        "native_pa_has_interval": "",
        "failure_reason": row.get("failure_reason", ""),
    }


def normalize_dorado(pt: int | None, pa: list[int] | None) -> dict[str, Any]:
    """Keep pt=0 and pt<0 separate; pt=0 alone is not a verified negative."""
    interval = bool(pa and len(pa) >= 3 and 0 <= pa[1] < pa[2])
    native = (
        "missing_pt"
        if pt is None
        else "pt_positive"
        if pt > 0
        else "pt_negative"
        if pt < 0
        else "pt_zero_with_interval"
        if interval
        else "pt_zero_without_interval"
    )
    category = (
        "positive_length"
        if pt is not None and pt > 0
        else "zero_output"
        if pt == 0
        else "error_or_missing"
    )
    return {
        "native_status": native,
        "status_category": category,
        "reported_length_nt": pt,
        "positive_length_nt": pt if category == "positive_length" else None,
        "native_pa_has_interval": interval,
        "failure_reason": "" if category == "positive_length" else native,
    }


def read_dorado(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read primary BAM identifiers and Poly tags only, not sequences/signals."""
    import pysam

    rows: dict[str, dict[str, Any]] = {}
    ignored = 0
    with pysam.AlignmentFile(str(path), "rb", check_sq=False) as bam:
        programs = bam.header.to_dict().get("PG", [])
        for record in bam.fetch(until_eof=True):
            if record.is_secondary or record.is_supplementary:
                ignored += 1
                continue
            if not record.query_name or record.query_name in rows:
                raise ValueError("empty/duplicate primary Dorado read_id")
            pt = record.get_tag("pt") if record.has_tag("pt") else None
            if pt is not None and not isinstance(pt, int):
                raise ValueError("Dorado pt tag must be an integer")
            pa = list(record.get_tag("pa")) if record.has_tag("pa") else None
            rows[record.query_name] = normalize_dorado(pt, pa)
    return rows, {"programs": programs, "ignored_secondary_supplementary": ignored}


def join_outputs(
    labels: dict[str, dict[str, str]],
    outputs: dict[str, dict[str, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Use the full label cohort as denominator; reserve missing tool rows."""
    if set(outputs) != set(TOOLS):
        raise ValueError("require all three tools")
    for tool, rows in outputs.items():
        if set(rows) - set(labels):
            raise ValueError(f"{tool} contains reads outside the comparison cohort")
    joined = []
    for identifier, label in labels.items():
        nominal = int(label["nominal_length_nt"])
        if nominal < 0 or not label.get("label_source"):
            raise ValueError("labels require nonnegative nominal lengths and source")
        if label.get("poly_type") != ("none" if nominal == 0 else "polyA"):
            raise ValueError("comparison supports polyA and separate zero_nt only")
        for tool in TOOLS:
            result = outputs[tool].get(
                identifier,
                {
                    "native_status": "missing_read",
                    "status_category": "error_or_missing",
                    "reported_length_nt": None,
                    "positive_length_nt": None,
                    "native_pa_has_interval": "",
                    "failure_reason": "absent_from_tool_output",
                },
            )
            joined.append(
                {
                    "read_id": identifier,
                    "nominal_length_nt": nominal,
                    "label_source": label["label_source"],
                    "tool": tool,
                    **result,
                }
            )
    return joined


def quantile(values: list[float], fraction: float) -> float | None:
    """Return linearly interpolated descriptive quantiles, without inference."""
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def summarize(joined: list[dict[str, Any]]) -> dict[str, Any]:
    """Describe native outputs and paired deviations from construct weak labels."""
    ids = {row["read_id"] for row in joined}
    positive_sets = {
        tool: {
            row["read_id"]
            for row in joined
            if row["tool"] == tool and row["positive_length_nt"] is not None
        }
        for tool in TOOLS
    }
    shared = set.intersection(*positive_sets.values())
    # Keep nominal-zero negatives in coverage/distributions, not positive-tail errors.
    shared = {
        row["read_id"]
        for row in joined
        if row["read_id"] in shared and row["nominal_length_nt"] > 0
    }
    lengths = sorted({row["nominal_length_nt"] for row in joined})
    groups: list[dict[str, Any]] = []
    overall: list[dict[str, Any]] = []
    for scope in ("tool_positive_outputs", "shared_positive_nominal_gt0"):
        for tool in TOOLS:
            for length in [None, *lengths]:
                cohort = [
                    row
                    for row in joined
                    if row["tool"] == tool
                    and (length is None or row["nominal_length_nt"] == length)
                ]
                eligible = [
                    row
                    for row in cohort
                    if row["positive_length_nt"] is not None
                    and (scope == "tool_positive_outputs" or row["read_id"] in shared)
                ]
                values = [float(row["positive_length_nt"]) for row in eligible]
                deviations = [
                    float(row["positive_length_nt"]) - row["nominal_length_nt"]
                    for row in eligible
                ]
                result = {
                    "scope": scope,
                    "tool": tool,
                    "nominal_length_nt": length,
                    "total_group_reads": len(cohort),
                    "n": len(values),
                    "fraction_of_group": len(values) / len(cohort) if cohort else None,
                    "median_nt": quantile(values, 0.5),
                    "q25_nt": quantile(values, 0.25),
                    "q75_nt": quantile(values, 0.75),
                    "mean_nt": statistics.mean(values) if values else None,
                    "mean_nominal_bias_nt": statistics.mean(deviations)
                    if values
                    else None,
                    "mean_absolute_nominal_deviation_nt": statistics.mean(
                        map(abs, deviations)
                    )
                    if values
                    else None,
                    "median_absolute_nominal_deviation_nt": quantile(
                        list(map(abs, deviations)), 0.5
                    ),
                }
                (overall if length is None else groups).append(result)
    return {
        "total_reads": len(ids),
        "nominal_lengths_nt": lengths,
        "counts_by_nominal_length": dict(
            Counter(
                row["nominal_length_nt"] for row in joined if row["tool"] == TOOLS[0]
            )
        ),
        "shared_positive_nominal_gt0_reads": len(shared),
        "shared_counts_by_nominal_length": dict(
            Counter(
                row["nominal_length_nt"]
                for row in joined
                if row["tool"] == TOOLS[0] and row["read_id"] in shared
            )
        ),
        "native_status_counts": {
            tool: dict(
                Counter(row["native_status"] for row in joined if row["tool"] == tool)
            )
            for tool in TOOLS
        },
        "status_category_counts": {
            tool: dict(
                Counter(row["status_category"] for row in joined if row["tool"] == tool)
            )
            for tool in TOOLS
        },
        "overall": overall,
        "by_nominal_length": groups,
    }


def write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write auditable source data, preserving empty values as missing."""
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    """Require explicit results, cohort and a new external output directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    for name in ("labels", "holypoly", "dorado-bam", "polyanalysis"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New work-root output directory: results/tool_comparison/<comparison-id>",
    )
    return parser


def resolve_paths(args: argparse.Namespace) -> tuple[list[Path], Path]:
    """Validate read-only result inputs and a new output outside raw data/repo."""
    config = load_config(args.config)
    if config.paths.work_root is None or config.paths.data_root is None:
        raise ValueError("configure data_root/work_root")
    work, data = config.paths.work_root.resolve(), config.paths.data_root.resolve()
    project = Path(__file__).resolve().parents[3]
    inputs = [
        (work / path).resolve()
        for path in (args.labels, args.holypoly, args.dorado_bam, args.polyanalysis)
    ]
    output = (work / args.output).absolute()
    if (
        output.is_symlink()
        or output.exists()
        or output.resolve() == work
        or not output.resolve().is_relative_to(work)
        or output.resolve().is_relative_to(data)
        or output.resolve().is_relative_to(project)
        or any(p.is_relative_to(output.resolve()) for p in inputs)
    ):
        raise ValueError(
            "output must be a new work-root child outside raw data/repo/inputs"
        )
    if any(not p.is_file() or not p.is_relative_to(work) for p in inputs):
        raise ValueError("all comparison inputs must exist inside work_root")
    return inputs, output.resolve()


def fingerprint(path: Path) -> dict[str, Any]:
    """Record content fingerprints for small pre-existing comparison inputs."""
    with path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": digest}


def extract(args: argparse.Namespace) -> None:
    """Extract only existing results into a reproducible comparison bundle."""
    inputs, output = resolve_paths(args)
    before = [fingerprint(p) for p in inputs]
    labels = read_unique(inputs[0])
    holo_native, poly_native = read_unique(inputs[1]), read_unique(inputs[3])
    for identifier, row in poly_native.items():
        if identifier in labels and int(row["nominal_length_nt"]) != int(
            labels[identifier]["nominal_length_nt"]
        ):
            raise ValueError("PolyAnalysis nominal labels differ from selected cohort")
    dorado, bam_metadata = read_dorado(inputs[2])
    outputs = {
        "Holypoly": {
            identifier: normalize_table(row, "Holypoly")
            for identifier, row in holo_native.items()
        },
        "PolyAnalysis": {
            identifier: normalize_table(row, "PolyAnalysis")
            for identifier, row in poly_native.items()
        },
        "Dorado": dorado,
    }
    joined = join_outputs(labels, outputs)
    summary = summarize(joined)
    summary.update(
        created_utc=datetime.now(UTC).isoformat(),
        inputs=before,
        dorado_bam_metadata=bam_metadata,
        command=[
            "scripts/compare_fixed_len_tools.py",
            *map(str, sys.argv[1:]),
        ],
        python_version=platform.python_version(),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        limitations=[
            "Construct nominal weak labels, not independently verified per-read "
            "truth; same acquisition as development.",
            "Main length distributions use positive native outputs only. "
            "Missing estimates and Dorado pt=0 are never relabeled as verified "
            "zero tails.",
            "All native numeric outputs, including Dorado zero/negative tags, "
            "are retained in source data. Zero-tag boundary validity is "
            "separately recorded.",
            "PolyAnalysis uses native max(1, round(interval_samples/30)); "
            "this proxy is uncalibrated for these cDNA reads.",
            "Dorado run used custom poly-A configuration and reference "
            "alignment; tool settings differ. No parameters or lengths were "
            "recalibrated on this final cohort.",
            "Shared-positive comparisons condition on all tools producing a "
            "positive value and nominal length >0; the intersection is not "
            "length-balanced. Group counts must be reported.",
            "No-candidate/zero tags are not biological negative calls; output "
            "coverage is not recall. Deviations from nominal are descriptive, "
            "not formal accuracy.",
        ],
    )
    if before != [fingerprint(p) for p in inputs]:
        raise ValueError("comparison inputs changed while reading")
    summary["inputs_unchanged"] = True
    output.mkdir(parents=True)
    write_table(output / "per_read_comparison.tsv", joined)
    write_table(output / "group_summary.tsv", summary["by_nominal_length"])
    write_table(output / "overall_summary.tsv", summary["overall"])
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "reads": len(labels),
                "shared_positive_reads": summary["shared_positive_nominal_gt0_reads"],
            }
        )
    )


if __name__ == "__main__":
    extract(build_parser().parse_args())
