"""Extract three unlabeled cDNA tool outputs without models or raw signals."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from holypoly.core.config import load_config
from holypoly.evaluation.cdna import CATEGORIES, read_holypoly
from holypoly.evaluation.fixed_len import fingerprint, normalize_table, read_dorado

TOOLS = ("Holypoly", "Dorado", "PolyAnalysis")
FIELDS = (
    "native_status",
    "status_category",
    "reported_length_nt",
    "positive_length_nt",
    "native_pa_has_interval",
    "failure_reason",
)


def read_polyanalysis(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Read complete persisted records; audit a terminal fragment without editing it.

    Only an unterminated final physical line may be discarded. Malformed complete
    rows, duplicate IDs and inconsistent native interval/30 proxies are errors.
    This does not infer biological zero lengths from missing candidates.
    """
    audit: dict[str, Any] = {"discarded_terminal_fragments": 0}
    rows: dict[str, dict[str, Any]] = {}

    def complete_lines(stream: Any) -> Any:
        """Exclude only the physically incomplete terminal line."""
        for line in stream:
            if not line.endswith("\n"):
                audit["discarded_terminal_fragments"] += 1
                continue
            yield line

    with path.open(newline="") as stream:
        reader = csv.DictReader(complete_lines(stream), delimiter="\t", strict=True)
        required = {"read_id", "status", "length_proxy_nt", "detector_region_type"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("PolyAnalysis table lacks required columns")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("malformed complete PolyAnalysis record")
            identifier = row["read_id"]
            if not identifier or identifier in rows:
                raise ValueError("empty or duplicate PolyAnalysis read_id")
            adapted = dict(row)
            adapted["predicted_length_nt"] = row["length_proxy_nt"]
            if row["status"] == "polyA_proxy_available":
                adapted["status"] = "polyA_length_proxy_available"
                start, end, size = (
                    int(row[key])
                    for key in ("start_sample", "end_sample", "raw_signal_length")
                )
                span = int(row["interval_samples"])
                if not (0 <= start < end <= size and end - start == span):
                    raise ValueError("invalid PolyAnalysis raw-sample interval")
                if float(row["length_proxy_nt"]) != max(1, round(span / 30)):
                    raise ValueError(
                        "PolyAnalysis proxy differs from native interval/30"
                    )
            elif row["length_proxy_nt"]:
                raise ValueError("unsuccessful PolyAnalysis row has a length")
            if row["status"] not in {
                "polyA_proxy_available",
                "other_region_polyT",
                "no_candidate",
                "failure",
            }:
                raise ValueError("unrecognized PolyAnalysis native status")
            normalized = normalize_table(adapted, "PolyAnalysis")
            normalized["native_status"] = row["status"]
            rows[identifier] = normalized
    if not rows:
        raise ValueError("no complete PolyAnalysis records")
    audit["retained_complete_records"] = len(rows)
    return rows, audit


def cohort_summary(outputs: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    """Separate all-available, shared-ID and shared-positive populations."""
    if set(outputs) != set(TOOLS) or any(not rows for rows in outputs.values()):
        raise ValueError("require nonempty outputs from exactly three tools")
    shared = set.intersection(*(set(outputs[tool]) for tool in TOOLS))
    positive = {
        identifier
        for identifier in shared
        if all(
            outputs[tool][identifier]["positive_length_nt"] is not None
            for tool in TOOLS
        )
    }
    return {
        "common_reads": len(shared),
        "common_positive_reads": len(positive),
        "tools": {
            tool: {
                "all_reads": len(rows),
                "all_status_counts": dict(
                    Counter(r["status_category"] for r in rows.values())
                ),
                "common_status_counts": dict(
                    Counter(rows[i]["status_category"] for i in shared)
                ),
                "outside_common_reads": len(rows) - len(shared),
            }
            for tool, rows in outputs.items()
        },
        "pairwise_common_reads": {
            f"{left}/{right}": len(outputs[left].keys() & outputs[right].keys())
            for index, left in enumerate(TOOLS)
            for right in TOOLS[index + 1 :]
        },
        "limitations": [
            "All-available distributions have different read populations "
            "and selection rules.",
            "Shared IDs align input populations; "
            "each tool still has its own positive subset.",
            "Shared-positive comparisons condition on positive outputs "
            "from all three tools.",
            "PolyAnalysis is interrupted, sequentially selected, "
            "and uncalibrated interval/30.",
            "Holypoly candidate regression is not gated by the status classifier.",
            "Native zero, no candidate, other type and error are distinct; "
            "no zero imputation.",
            "No per-read truth: no accuracy, biological recall, "
            "or tool ranking is inferred.",
        ],
    }


def extract(args: argparse.Namespace) -> None:
    """Read existing result tables/BAM once and write a new work-root bundle."""
    config = load_config(args.config)
    if config.paths.work_root is None or config.paths.data_root is None:
        raise ValueError("configure work_root and data_root")
    work, data = config.paths.work_root.resolve(), config.paths.data_root.resolve()
    inputs = [
        (work / p).resolve()
        for p in (args.holypoly, args.dorado_bam, args.polyanalysis)
    ]
    completion = inputs[0].parent / "summary.json"
    for path in [*inputs, completion]:
        if not path.is_file() or not path.is_relative_to(work):
            raise ValueError("comparison inputs must exist inside work_root")
    output = (work / args.output).resolve()
    if (
        output.exists()
        or output == work
        or not output.is_relative_to(work)
        or output.is_relative_to(data)
        or output.is_relative_to(Path(__file__).resolve().parents[3])
        or any(path.is_relative_to(output) for path in inputs)
    ):
        raise ValueError("output must be a new external work-root directory")
    fingerprints = [fingerprint(path) for path in (inputs[0], inputs[2], completion)]
    bam_stat = inputs[1].stat()
    holypoly = read_holypoly(inputs[0])
    if (
        len(holypoly)
        != json.loads(completion.read_text())["selected_primary_bam_reads"]
    ):
        raise ValueError("Holypoly output is incomplete")
    polyanalysis, audit = read_polyanalysis(inputs[2])
    print(
        f"Loaded Holypoly={len(holypoly):,}; PolyAnalysis={len(polyanalysis):,}",
        flush=True,
    )
    print(
        f"CPU-only, one read-only BAM pass: {bam_stat.st_size / 2**30:.2f} GiB",
        flush=True,
    )
    dorado, bam_metadata = read_dorado(inputs[1])
    outputs = {"Holypoly": holypoly, "Dorado": dorado, "PolyAnalysis": polyanalysis}
    summary = cohort_summary(outputs)
    for item in fingerprints:
        if fingerprint(Path(item["path"])) != item:
            raise ValueError("TSV/completion inputs changed during extraction")
    after = inputs[1].stat()
    if (after.st_size, after.st_mtime_ns) != (bam_stat.st_size, bam_stat.st_mtime_ns):
        raise ValueError("Dorado BAM changed during extraction")
    output.mkdir(parents=True, exist_ok=False)
    for tool, rows in outputs.items():
        with (output / f"{tool.lower()}_results.tsv").open("x", newline="") as stream:
            writer = csv.writer(stream, delimiter="\t")
            writer.writerow(["read_id", *FIELDS])
            for identifier, row in rows.items():
                writer.writerow([identifier, *(row[field] for field in FIELDS)])
    with (output / "common_read_comparison.tsv").open("x", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["read_id", *[f"{t}_{f}" for t in TOOLS for f in FIELDS]])
        for identifier in holypoly:
            if identifier in dorado and identifier in polyanalysis:
                writer.writerow(
                    [
                        identifier,
                        *[outputs[t][identifier][f] for t in TOOLS for f in FIELDS],
                    ]
                )
    summary.update(
        inputs=[
            *fingerprints,
            {
                "path": str(inputs[1]),
                "bytes": bam_stat.st_size,
                "mtime_ns": bam_stat.st_mtime_ns,
            },
        ],
        polyanalysis_audit=audit,
        dorado_metadata=bam_metadata,
        sources_unchanged=True,
        no_gpu_model_or_raw_signal_used=True,
        finished_utc=datetime.now(UTC).isoformat(),
        git_commit=subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status_categories=CATEGORIES,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                "common_reads": summary["common_reads"],
                "common_positive_reads": summary["common_positive_reads"],
                "tools": summary["tools"],
                "output": str(output),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "holypoly", "dorado-bam", "polyanalysis", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    extract(parser.parse_args())
