"""Select a small balanced v4 manifest without reading signals or changing labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from holypoly.core.config import load_config

HEAD_POLY_VERSION = "construct-nominal-head-poly-v4"
SPLITS = ("train", "validation", "test")


def select_subset(
    rows: list[dict[str, str]], *, per_group: int, seed: int
) -> list[dict[str, str]]:
    """Sample 80/10/10 within existing splits for each nominal-length/type group.

    Every selected row is retained unchanged. Counts must be achievable in every
    source stratum; missing splits or insufficient counts are explicit errors.
    """
    if per_group < 10 or per_group % 10:
        raise ValueError("per_group must be a positive multiple of 10, at least 10")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if not rows:
        raise ValueError("manifest is empty")
    pools: dict[tuple[float, str], dict[str, list[dict[str, str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    seen: set[str] = set()
    for row in rows:
        if (
            row.get("manifest_version") != HEAD_POLY_VERSION
            or row.get("supervision_mode") != "length_only"
        ):
            raise ValueError("subset requires a v4 length_only manifest")
        read_id = row["read_id"]
        if not read_id or read_id in seen:
            raise ValueError("manifest contains empty or duplicate read IDs")
        seen.add(read_id)
        nominal = float(row["nominal_length_nt"])
        if not math.isfinite(nominal) or nominal < 0:
            raise ValueError("nominal_length_nt must be finite and non-negative")
        split = row["split"]
        if split not in SPLITS:
            raise ValueError("manifest contains an unsupported split")
        pools[(nominal, row["poly_type"])][split].append(row)
    requested = {
        "train": per_group * 8 // 10,
        "validation": per_group // 10,
        "test": per_group // 10,
    }
    selected: list[dict[str, str]] = []
    rng = random.Random(seed)
    for group, split_rows in sorted(pools.items()):
        for split in SPLITS:
            candidates = sorted(split_rows[split], key=lambda row: row["read_id"])
            if len(candidates) < requested[split]:
                raise ValueError(
                    f"insufficient {group}/{split}: need {requested[split]}, "
                    f"have {len(candidates)}"
                )
            selected.extend(
                sorted(
                    rng.sample(candidates, requested[split]),
                    key=lambda row: row["read_id"],
                )
            )
    return selected


def run(args: argparse.Namespace) -> dict[str, object]:
    """Read only the source manifest; write a new external manifest and summary."""
    config = load_config(
        args.config, cli_data_root=args.data_root, cli_work_root=args.work_root
    )
    if config.paths.data_root is None or config.paths.work_root is None:
        raise ValueError("configure data_root and work_root first")
    data = config.paths.data_root.expanduser().resolve()
    work = config.paths.work_root.expanduser().resolve()
    project = Path(__file__).resolve().parents[3]
    if not data.is_dir() or not work.is_dir():
        raise FileNotFoundError("configured data_root/work_root must exist")
    if work.is_relative_to(data) or work.is_relative_to(project):
        raise ValueError("work_root must be external and outside raw data")
    source = (work / args.manifest.expanduser()).resolve()
    output = (work / args.output_dir.expanduser()).resolve()
    for path in (source, output):
        if (
            not path.is_relative_to(work)
            or path == work
            or path.is_relative_to(data)
            or path.is_relative_to(project)
        ):
            raise ValueError("manifest/output must be inside external work_root")
    if not source.is_file():
        raise FileNotFoundError(source)
    if output.exists():
        raise FileExistsError(f"choose a new output directory: {output}")
    with source.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(
            stream, delimiter="\t" if source.suffix == ".tsv" else ","
        )
        fields = reader.fieldnames
        required = {
            "read_id",
            "split",
            "nominal_length_nt",
            "poly_type",
            "manifest_version",
            "supervision_mode",
        }
        if fields is None or not required.issubset(fields):
            raise ValueError("manifest is missing required subset fields")
        rows = list(reader)
    if any(None in row or any(value is None for value in row.values()) for row in rows):
        raise ValueError("manifest contains malformed rows")
    selected = select_subset(rows, per_group=args.per_group, seed=args.seed)
    with source.open("rb") as stream:
        source_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()
    summary: dict[str, object] = {
        "source_manifest": str(source),
        "source_manifest_sha256": source_sha256,
        "manifest_version": HEAD_POLY_VERSION,
        "per_group": args.per_group,
        "seed": args.seed,
        "source_rows": len(rows),
        "selected_rows": len(selected),
        "split_counts": dict(Counter(row["split"] for row in selected)),
        "selected_counts": dict(
            Counter(
                f"{row['split']}:{row['nominal_length_nt']}:{row['poly_type']}"
                for row in selected
            )
        ),
        "selection_policy": "random_within_existing_length_type_split",
        "limitations": (
            "Small development subset; unchanged construct nominal weak labels. "
            "Inherited source selection bias; not independent per-read truth. "
            "Few optimization steps may underfit; no formal accuracy claims."
        ),
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        output.mkdir(parents=True)
        with (output / "manifest.tsv").open(
            "w", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            writer.writerows(selected)
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    """Provide a parent-environment CLI with no Torch/POD5 dependencies."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--work-root", type=Path)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--per-group", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args), sort_keys=True))
    except (OSError, ValueError, KeyError) as error:
        print(f"Manifest subsetting failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
