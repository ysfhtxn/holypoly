"""Extract unlabeled cDNA tool outputs on a shared read cohort, without GPU use."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from holypoly.core.config import load_config
from holypoly.evaluation.fixed_len import fingerprint, normalize_table, read_dorado

TOOLS = ("Holypoly", "Dorado")
CATEGORIES = (
    "positive_length",
    "zero_output",
    "no_candidate",
    "other_or_ambiguous",
    "error_or_missing",
)


def read_holypoly(path: Path) -> dict[str, dict[str, Any]]:
    """Retain only read IDs and native poly(A) results, discarding signal fields."""
    rows = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        required = {"read_id", "status", "poly_type", "predicted_length_nt"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Holypoly predictions lack required columns")
        for row in reader:
            identifier = row["read_id"]
            if not identifier or identifier in rows:
                raise ValueError("empty or duplicate Holypoly read_id")
            rows[identifier] = normalize_table(row, "Holypoly")
    if not rows:
        raise ValueError("empty Holypoly predictions")
    return rows


def cohort_summary(
    holypoly: dict[str, dict[str, Any]], dorado: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Keep full-tool counts separate from the common primary-read denominator."""
    shared = holypoly.keys() & dorado.keys()
    if not shared:
        raise ValueError("tools have no common read IDs")
    outputs = {"Holypoly": holypoly, "Dorado": dorado}
    counts = {
        tool: {
            "all_primary_reads": len(rows),
            "all_status_counts": dict(
                Counter(r["status_category"] for r in rows.values())
            ),
            "common_status_counts": dict(
                Counter(rows[i]["status_category"] for i in shared)
            ),
        }
        for tool, rows in outputs.items()
    }
    return {
        "common_reads": len(shared),
        "holypoly_only_reads": len(holypoly) - len(shared),
        "dorado_only_reads": len(dorado) - len(shared),
        "common_positive_reads": sum(
            holypoly[i]["positive_length_nt"] is not None
            and dorado[i]["positive_length_nt"] is not None
            for i in shared
        ),
        "tools": counts,
        "limitations": [
            "Unlabeled biological samples: no length accuracy or biological recall",
            "Positive outputs do not establish biological tail identity "
            "or completeness",
            "Dorado native zero tags are not verified biological zero tails",
            "Common read IDs are used for comparison; "
            "tool-specific positive subsets differ",
            "Holypoly raw regression is not gated by its status classifier",
        ],
    }


def extract(args: argparse.Namespace) -> None:
    """Read existing TSV/BAM once and write a fresh external comparison bundle."""
    config = load_config(args.config)
    if config.paths.work_root is None or config.paths.data_root is None:
        raise ValueError("configure work_root and data_root")
    work, data = config.paths.work_root.resolve(), config.paths.data_root.resolve()
    holo_path = (work / args.holypoly).resolve()
    dorado_path = (work / args.dorado_bam).resolve()
    completion = holo_path.parent / "summary.json"
    for path in (holo_path, completion, dorado_path):
        if not path.is_file() or not path.is_relative_to(work):
            raise FileNotFoundError(
                f"required completed result missing: {path}; "
                "wait for prediction to finish"
            )
    output = (work / args.output).resolve()
    if (
        output.exists()
        or output == work
        or not output.is_relative_to(work)
        or output.is_relative_to(data)
        or output.is_relative_to(Path(__file__).resolve().parents[3])
        or any(p.is_relative_to(output) for p in (holo_path, dorado_path))
    ):
        raise ValueError(
            "output must be a new work-root child outside data/repo/inputs"
        )
    holo_fingerprint = fingerprint(holo_path)
    completion_fingerprint = fingerprint(completion)
    bam_stat = dorado_path.stat()
    holypoly = read_holypoly(holo_path)
    expected = json.loads(completion.read_text())["selected_primary_bam_reads"]
    if len(holypoly) != expected:
        raise ValueError("Holypoly rows do not match completed prediction summary")
    print(
        f"Read-only scan: {bam_stat.st_size / 2**30:.2f} GiB Dorado BAM; "
        "CPU only, no models or raw signals. May take minutes and 1-2 GiB RAM.",
        flush=True,
    )
    dorado, bam_metadata = read_dorado(dorado_path)
    summary = cohort_summary(holypoly, dorado)
    if (
        fingerprint(holo_path) != holo_fingerprint
        or fingerprint(completion) != completion_fingerprint
    ):
        raise ValueError("Holypoly inputs changed during extraction")
    after = dorado_path.stat()
    if (after.st_size, after.st_mtime_ns) != (bam_stat.st_size, bam_stat.st_mtime_ns):
        raise ValueError("Dorado BAM changed during extraction; wait for completion")
    output.mkdir(parents=True)
    with (output / "holypoly_lengths.tsv").open("x", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(["read_id", "positive_length_nt"])
        for identifier, row in holypoly.items():
            if row["positive_length_nt"] is not None:
                writer.writerow([identifier, row["positive_length_nt"]])
    with (output / "common_read_comparison.tsv").open("x", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t")
        writer.writerow(
            [
                "read_id",
                *[
                    f"{t}_{field}"
                    for t in TOOLS
                    for field in (
                        "native_status",
                        "status_category",
                        "reported_length_nt",
                        "positive_length_nt",
                        "native_pa_has_interval",
                    )
                ],
            ]
        )
        for identifier, h in holypoly.items():
            if identifier in dorado:
                writer.writerow(
                    [
                        identifier,
                        *[
                            r[field]
                            for r in (h, dorado[identifier])
                            for field in (
                                "native_status",
                                "status_category",
                                "reported_length_nt",
                                "positive_length_nt",
                                "native_pa_has_interval",
                            )
                        ],
                    ]
                )
    summary.update(
        {
            "inputs": [
                holo_fingerprint,
                completion_fingerprint,
                {
                    "path": str(dorado_path),
                    "bytes": bam_stat.st_size,
                    "mtime_ns": bam_stat.st_mtime_ns,
                    "fingerprint_method": "size and mtime; no extra full-file hashing",
                },
            ],
            "dorado_metadata": bam_metadata,
            "sources_unchanged": True,
            "no_gpu_model_or_raw_signal_used": True,
        }
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--holypoly", type=Path, required=True)
    parser.add_argument("--dorado-bam", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    extract(parser.parse_args())
