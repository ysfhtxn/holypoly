"""Organize existing comparison artifacts and add explicitly independent cohorts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import posixpath
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from plot_nano3p_context import plot_context  # noqa: E402

from holypoly.core.config import load_config  # noqa: E402
from holypoly.evaluation.fixed_len import TOOLS, fingerprint  # noqa: E402
from holypoly.evaluation.xlsx_csv import (  # noqa: E402
    NS,
    RELATIONSHIP_ID,
    _rows,
    _shared_strings,
)


def value_status(text: str) -> tuple[str, float | None]:
    """Classify native nt values without imputing missing/failure values as zero."""
    if text.strip().lower() in {"", "na", "n/a", "none", "null"}:
        return "missing_value", None
    try:
        value = float(text)
    except ValueError:
        return "invalid_value", None
    if not math.isfinite(value):
        return "nonfinite_value", None
    if value < 0:
        return "negative_value", value
    return ("positive_length" if value > 0 else "zero_value"), value


def read_tailfindr(path: Path) -> dict[str, dict]:
    """Stream native CSV lengths by replicate; do not open referenced FAST5 files."""
    groups: dict[str, dict] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"tail_length", "replicate"} <= set(reader.fieldnames or []):
            raise ValueError("tailfindr requires tail_length and replicate")
        for row in reader:
            replicate = row["replicate"].strip()
            if not replicate:
                raise ValueError("tailfindr replicate is missing")
            group = groups.setdefault(
                f"rep{replicate}", {"values": [], "status_counts": Counter()}
            )
            status, value = value_status(row["tail_length"])
            group["status_counts"][status] += 1
            if status == "positive_length":
                group["values"].append(value)
    if not groups or any(not group["values"] for group in groups.values()):
        raise ValueError("tailfindr contains an empty positive replicate")
    return dict(sorted(groups.items()))


def read_tailfindr_dna(path: Path) -> tuple[dict[str, dict], dict]:
    """Keep valid DNA tail lengths separate by native poly type and replicate.

    All source rows are counted, including invalid types, native invalid tails,
    missing lengths and zero values. No signal paths are opened or nominal
    construct lengths inferred from filenames.
    """
    groups: dict[str, dict] = {}
    native: dict[tuple[str, str, str], Counter] = {}
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"replicate", "read_type", "tail_is_valid", "tail_length"}
        if not required <= set(reader.fieldnames or []):
            raise ValueError("missing tailfindr DNA type/validity/length columns")
        for row in reader:
            replicate, poly_type = row["replicate"].strip(), row["read_type"].strip()
            valid = row["tail_is_valid"].strip().upper()
            if not replicate or valid not in {"TRUE", "FALSE"}:
                raise ValueError("invalid tailfindr DNA replicate or validity flag")
            status, value = value_status(row["tail_length"])
            native.setdefault((replicate, poly_type, valid), Counter())[status] += 1
            if poly_type not in {"polyA", "polyT"}:
                continue
            name = f"DNA {poly_type} rep{replicate}"
            group = groups.setdefault(
                name,
                {
                    "values": [],
                    "status_counts": Counter(),
                    "molecule": "DNA",
                    "cohort": "tailfindr_DNA_LSK108_109",
                    "poly_type": poly_type,
                    "replicate": replicate,
                    "display_label": f"DNA {poly_type[-1]}\nrep{replicate}",
                },
            )
            group["status_counts"][status if valid == "TRUE" else "native_invalid"] += 1
            if valid == "TRUE" and status == "positive_length":
                group["values"].append(value)
    if not groups or any(not group["values"] for group in groups.values()):
        raise ValueError("tailfindr DNA has an empty valid positive group")
    return dict(sorted(groups.items())), {
        "reported_rows": sum(sum(counts.values()) for counts in native.values()),
        "native_groups": [
            {
                "replicate": key[0],
                "read_type": key[1],
                "tail_is_valid": key[2],
                "length_status_counts": dict(counts),
            }
            for key, counts in sorted(native.items())
        ],
        "selection": "tail_is_valid=TRUE; native read_type=polyA/polyT; "
        "finite tail_length > 0",
        "nominal_length_assigned": False,
    }


DNA_NOMINAL_LENGTHS = (10, 30, 40, 60, 100, 150)


def read_barcoded_dna(
    length_path: Path, barcode_path: Path, *, joined_output: Path | None = None
) -> tuple[dict[str, dict], dict]:
    """Join different basecalling runs by (replicate, read_id), never row order.

    Barcode threshold and tie flags select labels independently of estimated
    tail length. Native read-length flags are audited but add no extra filter.
    Conflicting poly directions across basecallers are excluded explicitly.
    Output groups retain replicate, nominal construct nt, and native direction.
    """
    barcodes = {}
    barcode_counts, native_flags = Counter(), Counter()
    with barcode_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        required = {
            "replicate",
            "read_id",
            "read_type",
            "barcode",
            "barcode_tie",
            "barcode_passed_threshold",
            "read_too_long",
            "read_too_short",
        }
        if not required <= set(reader.fieldnames or []):
            raise ValueError("missing decoded barcode identity/quality columns")
        for row in reader:
            key = (row["replicate"].strip(), row["read_id"].strip())
            if not all(key) or key in barcodes:
                raise ValueError("missing or duplicate barcode (replicate, read_id)")
            native_flags[
                tuple(
                    row[field].strip().upper()
                    for field in (
                        "barcode_passed_threshold",
                        "barcode_tie",
                        "read_too_long",
                        "read_too_short",
                    )
                )
            ] += 1
            label = row["barcode"].strip()
            if row["barcode_passed_threshold"].strip().upper() != "TRUE":
                reason = "threshold_not_passed"
            elif row["barcode_tie"].strip().upper() != "FALSE":
                reason = "tied_or_missing_tie_flag"
            elif label not in {str(n) for n in DNA_NOMINAL_LENGTHS}:
                reason = "outside_six_nominal_groups"
            elif row["read_type"].strip() not in {"polyA", "polyT"}:
                reason = "invalid_barcode_read_type"
            else:
                reason = "accepted_barcode"
            barcode_counts[reason] += 1
            barcodes[key] = (label, row["read_type"].strip(), reason)

    groups: dict[str, dict] = {}
    seen, matched = set(), set()
    join_counts, eligible_states, direction_pairs = Counter(), Counter(), Counter()
    output = joined_output.open("x", newline="") if joined_output else None
    writer = None
    if output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "read_id",
                "replicate",
                "nominal_length_nt",
                "read_type",
                "barcode_read_type",
                "tail_is_valid",
                "tail_length",
                "length_status",
                "selection_status",
                "label_source",
            ],
            delimiter="\t",
        )
        writer.writeheader()
    try:
        with length_path.open(newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "replicate",
                "read_id",
                "read_type",
                "tail_is_valid",
                "tail_length",
            }
            if not required <= set(reader.fieldnames or []):
                raise ValueError("missing DNA length identity/type/validity columns")
            for row in reader:
                key = (row["replicate"].strip(), row["read_id"].strip())
                if not all(key) or key in seen:
                    raise ValueError("missing or duplicate length (replicate, read_id)")
                seen.add(key)
                if key not in barcodes:
                    join_counts["no_barcode_row"] += 1
                    continue
                matched.add(key)
                label, barcode_type, reason = barcodes[key]
                if reason != "accepted_barcode":
                    join_counts[reason] += 1
                    continue
                native_type = row["read_type"].strip()
                valid = row["tail_is_valid"].strip().upper()
                if valid not in {"TRUE", "FALSE"}:
                    raise ValueError("invalid DNA tail validity flag")
                status, value = value_status(row["tail_length"])
                direction_pairs[(barcode_type, native_type)] += 1
                if native_type not in {"polyA", "polyT"}:
                    selection = "invalid_length_read_type"
                elif native_type != barcode_type:
                    selection = "direction_disagreement"
                elif valid != "TRUE":
                    selection = "native_invalid"
                else:
                    selection = status
                join_counts[selection] += 1
                eligible_states[(key[0], label, native_type, selection)] += 1
                if native_type == barcode_type and native_type in {"polyA", "polyT"}:
                    name = f"DNA {label}nt {native_type} rep{key[0]}"
                    group = groups.setdefault(
                        name,
                        {
                            "values": [],
                            "status_counts": Counter(),
                            "molecule": "DNA",
                            "cohort": "tailfindr_DNA_LSK108_109",
                            "poly_type": native_type,
                            "replicate": key[0],
                            "nominal_length_nt": int(label),
                            "label_source": "decoded barcode; flipflop basecalling",
                        },
                    )
                    group["status_counts"][selection] += 1
                    if selection == "positive_length":
                        group["values"].append(value)
                if writer:
                    writer.writerow(
                        {
                            "read_id": key[1],
                            "replicate": key[0],
                            "nominal_length_nt": label,
                            "read_type": native_type,
                            "barcode_read_type": barcode_type,
                            "tail_is_valid": valid,
                            "tail_length": row["tail_length"],
                            "length_status": status,
                            "selection_status": selection,
                            "label_source": "flipflop_decoded_barcode",
                        }
                    )
    finally:
        if output:
            output.close()
    if not any(g["values"] for g in groups.values()):
        raise ValueError("no barcode-matched valid positive DNA lengths")
    return dict(
        sorted(
            groups.items(),
            key=lambda item: (
                item[1]["nominal_length_nt"],
                item[1]["poly_type"],
                item[1]["replicate"],
            ),
        )
    ), {
        "nominal_lengths_nt": DNA_NOMINAL_LENGTHS,
        "nominal_length_assigned": True,
        "label_limitation": "Construct nominal labels, not per-read ground truth",
        "join_key": ["replicate", "read_id"],
        "length_basecalling": "standard",
        "barcode_basecalling": "flipflop",
        "barcode_rows": len(barcodes),
        "length_rows": len(seen),
        "matched_keys": len(matched),
        "barcode_only_keys": len(barcodes) - len(matched),
        "length_only_keys": len(seen) - len(matched),
        "barcode_selection_counts": dict(barcode_counts),
        "length_row_selection_counts": dict(join_counts),
        "selection": "barcode threshold TRUE, tie FALSE, six allowed labels; "
        "matching native polyA/polyT directions; tail_is_valid TRUE; finite "
        "positive nt",
        "read_length_flags": "audited, no additional selection",
        "native_barcode_flags": [
            {
                "passed_threshold": k[0],
                "tie": k[1],
                "too_long": k[2],
                "too_short": k[3],
                "rows": v,
            }
            for k, v in sorted(native_flags.items())
        ],
        "direction_pairs": [
            {"barcode_type": k[0], "length_type": k[1], "rows": v}
            for k, v in sorted(direction_pairs.items())
        ],
        "group_selection_counts": [
            {
                "replicate": k[0],
                "nominal_length_nt": int(k[1]),
                "read_type": k[2],
                "status": k[3],
                "rows": v,
            }
            for k, v in sorted(eligible_states.items())
        ],
    }


def pool_dna_replicates(groups: dict[str, dict]) -> dict[str, dict]:
    """Pool repetitions only within the same nominal construct and direction."""
    pooled: dict[str, dict] = {}
    for group in groups.values():
        name = f"DNA {group['nominal_length_nt']}nt {group['poly_type']} rep1+rep2"
        target = pooled.setdefault(
            name,
            {
                **group,
                "values": [],
                "status_counts": Counter(),
                "replicate": "1+2",
            },
        )
        target["values"].extend(group["values"])
        target["status_counts"].update(group["status_counts"])
    return pooled


def render_dna_replicates(groups: dict[str, dict], output: Path) -> None:
    """Show the barcode-defined DNA groups separately for each repetition."""
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    figure.subplots_adjust(left=0.07, right=0.98, top=0.8, bottom=0.24, wspace=0.12)
    rows = []
    for axis, poly_type in zip(axes, ("polyA", "polyT"), strict=True):
        for group in groups.values():
            if group["poly_type"] != poly_type:
                continue
            nominal, replicate = group["nominal_length_nt"], group["replicate"]
            values = group["values"]
            position = nominal + (-1.4 if replicate == "1" else 1.4)
            if values:
                axis.boxplot(
                    [values],
                    positions=[position],
                    widths=2.3,
                    manage_ticks=False,
                    patch_artist=True,
                    showmeans=True,
                    meanprops={
                        "marker": "o",
                        "markersize": 3,
                        "markerfacecolor": "white",
                        "markeredgecolor": "#202020",
                    },
                    medianprops={"color": "#202020", "linewidth": 1.1},
                    boxprops={
                        "facecolor": "#882255" if replicate == "1" else "#CC88AA"
                    },
                    flierprops={"marker": ".", "markersize": 1, "alpha": 0.2},
                )
            rows.append(
                {
                    "nominal_length_nt": nominal,
                    "poly_type": poly_type,
                    "replicate": replicate,
                    **group_audit({"group": group})["group"],
                }
            )
        axis.plot(DNA_NOMINAL_LENGTHS, DNA_NOMINAL_LENGTHS, "--", color="#555555", lw=1)
        axis.set(
            xticks=DNA_NOMINAL_LENGTHS,
            xlim=(3, 158),
            ylim=(0, None),
            title=poly_type,
            xlabel="Barcode-defined nominal tail length (nt)",
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Positive reported length (nt)")
    figure.suptitle("tailfindr DNA: six barcode-defined constructs, two replicates")
    figure.legend(
        handles=[
            Line2D([], [], color=c, lw=7, label=f"rep{i}")
            for i, c in ((1, "#882255"), (2, "#CC88AA"))
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.92),
        ncol=2,
    )
    figure.text(
        0.07,
        0.1,
        "LSK108/109 independent DNA dataset; standard-basecalling lengths "
        "joined to flipflop-decoded barcodes by replicate + read ID.",
        fontsize=8,
    )
    figure.text(
        0.07,
        0.065,
        "Barcode threshold passed, no tie, agreeing A/T direction, valid "
        "positive tail. Nominal lengths are construct labels, not per-read "
        "truth.",
        fontsize=8,
    )
    figure.text(
        0.07,
        0.03,
        "Boxes: median/IQR, 1.5 IQR whiskers; all outliers; circle: mean. "
        "Full positive counts and summaries are in the source tables.",
        fontsize=8,
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output / f"tailfindr_dna_six_lengths_by_replicate.{suffix}", dpi=300
        )
    for axis in axes:
        axis.set_ylim(0, 220)
    figure.text(0.7, 0.94, "Focus: 0–220 nt; full statistics retained", fontsize=8)
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(
            output / f"tailfindr_dna_six_lengths_by_replicate_focus_0_220nt.{suffix}",
            dpi=300,
        )
    plt.close(figure)
    # Counter dictionaries are serialized, not expanded into scientific metrics.
    write_table(
        output / "replicate_group_summary.tsv",
        [{**row, "status_counts": json.dumps(row["status_counts"])} for row in rows],
    )


def read_shuyi(
    path: Path, *, truncated_policy: str = "error"
) -> tuple[dict[str, dict], dict]:
    """Flatten raw replicate lists, preserving list-entry rather than read identity.

    Transcript medians and CPM values are not used. Missing, invalid and native
    zero entries remain separately counted. List lengths are checked against
    the provided polyA_reads_count fields before plotting.
    """
    if truncated_policy not in {"error", "prefix", "exclude"}:
        raise ValueError("unknown truncated-list policy")
    groups = {f"rep{i}": {"values": [], "status_counts": Counter()} for i in (1, 2)}
    audit: dict = {
        "sheet": "raw",
        "transcript_rows": 0,
        "count_mismatches": [],
        "truncated_cells": [],
        "truncated_policy": truncated_policy,
        "reported_length_count": 0,
        "available_length_entries": 0,
    }
    with ZipFile(path) as book:
        from xml.etree import ElementTree as ET

        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        sheets = workbook.findall("s:sheets/s:sheet", NS)
        sheet = next((s for s in sheets if s.get("name") == "raw"), None)
        if sheet is None:
            raise ValueError("Shuyi workbook requires the raw worksheet")
        relations = {
            e.attrib["Id"]: e.attrib["Target"]
            for e in ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        }
        target = relations[sheet.attrib[RELATIONSHIP_ID]]
        target = posixpath.normpath("xl/" + target) if target[0] != "/" else target[1:]
        if not target.startswith("xl/"):
            raise ValueError("worksheet outside workbook")
        rows = _rows(book, target, _shared_strings(book))
        _, header = next(rows)
        required = [
            f"polyA_{field}_293T_total_rep{i}"
            for i in (1, 2)
            for field in ("length", "reads_count")
        ]
        if not set(required) <= set(header):
            raise ValueError("missing Shuyi raw length-list/count fields")
        audit["length_columns"] = [c for c in required if "_length_" in c]
        for number, cells in rows:
            row = dict(
                zip(header, cells + [""] * (len(header) - len(cells)), strict=True)
            )
            audit["transcript_rows"] += 1
            for i in (1, 2):
                group = groups[f"rep{i}"]
                raw_text = row[f"polyA_length_293T_total_rep{i}"]
                text = raw_text.strip()
                count = row[f"polyA_reads_count_293T_total_rep{i}"]
                expected = int(count) if count.strip() else None
                if expected is not None:
                    audit["reported_length_count"] += expected
                truncated = False
                if text.lower() in {"", "na", "nan", "none"}:
                    group["status_counts"]["missing_list"] += 1
                    entries = []
                else:
                    try:
                        entries = json.loads(text)
                    except ValueError as error:
                        if (
                            truncated_policy == "error"
                            or len(raw_text) != 32767
                            or not text.startswith("[")
                            or text.endswith("]")
                        ):
                            raise ValueError(
                                f"malformed Shuyi list at row {number}, rep{i}"
                            ) from error
                        # Only numbers followed by a stored comma are complete.
                        # Never use the final token, even if it looks numeric.
                        prefix = json.loads(text.rsplit(",", 1)[0] + "]")
                        entries = prefix if truncated_policy == "prefix" else []
                        audit["truncated_cells"].append(
                            {
                                "source_row": number,
                                "replicate": i,
                                "reported_count": expected,
                                "complete_prefix_entries": len(prefix),
                                "retained_entries": len(entries),
                            }
                        )
                        truncated = True
                    if entries == 0 and count.strip() and float(count) == 0:
                        group["status_counts"]["empty_list_zero_sentinel"] += 1
                        entries = []
                    if not isinstance(entries, list):
                        raise ValueError(f"non-list at row {number}, rep{i}")
                audit["available_length_entries"] += len(entries)
                if expected is not None and expected != len(entries) and not truncated:
                    audit["count_mismatches"].append(
                        {
                            "source_row": number,
                            "replicate": i,
                            "reported_count": count,
                            "list_entries": len(entries),
                        }
                    )
                for entry in entries:
                    if isinstance(entry, (list, dict, bool)):
                        raise ValueError(f"non-scalar length at row {number}")
                    status, value = value_status(str(entry))
                    group["status_counts"][status] += 1
                    if status == "positive_length":
                        group["values"].append(value)
    if any(not group["values"] for group in groups.values()):
        raise ValueError("Shuyi contains an empty positive replicate")
    audit["identity_limitation"] = (
        "Length-list entries lack read IDs; unique reads and cross-transcript "
        "duplication cannot be verified. No deduplication or cross-dataset matching."
    )
    return groups, audit


def write_json(path: Path, value: object) -> None:
    """Create an audit file without replacing an existing artifact."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def write_table(path: Path, rows: list[dict]) -> None:
    """Create a TSV containing descriptive statistics or complete histogram bins."""
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def verify_original_statistics(
    fixed_source: Path,
    image_source: Path,
    fixed_records: list[dict],
    image_records: list[dict],
    *,
    local_tools: tuple[str, ...] = TOOLS,
) -> None:
    """Require original groups and Image #1 summaries to remain unchanged."""
    with fixed_source.open() as stream:
        originals = list(csv.DictReader(stream))
    new = {(r["cohort"], r["tool"], r["group"]): r for r in fixed_records}
    for original in originals:
        if (
            original["cohort"] == "local_fixed_len"
            and original["tool"] not in local_tools
        ):
            continue
        row = new[original["cohort"], original["tool"], original["group"]]
        for key, text in original.items():
            if key in {"cohort", "tool", "group"}:
                continue
            if text == "":
                if row[key] is not None:
                    raise ValueError(f"original fixed-len field changed: {key}")
            elif not math.isclose(float(text), float(row[key]), rel_tol=1e-12):
                raise ValueError(f"original fixed-len statistic changed: {key}")
    with image_source.open() as stream:
        originals = [
            r
            for r in csv.DictReader(stream, delimiter="\t")
            if r["cohort"] == "local_common_positive_with_external_hela"
            and (r["tool"] in local_tools or r["tool"].startswith("Nano3P-seq"))
        ]
    new = {r["series"]: r for r in image_records}
    for original in originals:
        row = new[original["tool"]]
        for key in ("mean_nt", "median_nt", "q25_nt", "q75_nt", "min_nt", "max_nt"):
            if not math.isclose(float(original[key]), row[key], rel_tol=1e-12):
                raise ValueError(f"original Image #1 statistic changed: {key}")
        if int(original["positive_rows"]) != row["positive_entries"]:
            raise ValueError("original Image #1 positive count changed")
    if len(originals) != len(local_tools) + 1:
        raise ValueError("expected selected local tools plus original HeLa series")


def group_audit(groups: dict[str, dict]) -> dict:
    """Summarize complete positive entries separately for each source replicate."""
    result = {}
    for name, group in groups.items():
        array = np.asarray(group["values"])
        result[name] = {
            "status_counts": dict(group["status_counts"]),
            "positive_entries": len(array),
            "mean_nt": float(array.mean()) if len(array) else None,
            "median_nt": float(np.median(array)) if len(array) else None,
            "min_nt": float(array.min()) if len(array) else None,
            "max_nt": float(array.max()) if len(array) else None,
        }
    return result


def read_image_cohorts(local: Path) -> tuple[dict[str, list[float]], int]:
    """Reuse exactly the Image #1 shared-positive local subset and HeLa rows."""
    series: dict[str, list[float]] = {tool: [] for tool in TOOLS}
    with (local / "common_read_comparison.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            texts = [row[f"{tool}_positive_length_nt"] for tool in TOOLS]
            if all(texts):
                for tool, text in zip(TOOLS, texts, strict=True):
                    status, value = value_status(text)
                    if status != "positive_length":
                        raise ValueError("invalid shared-positive local output")
                    series[tool].append(value)
    summary = json.loads((local / "summary.json").read_text())
    if any(len(v) != summary["common_positive_reads"] for v in series.values()):
        raise ValueError("local shared-positive cohort changed")
    series["Nano3P-seq / HeLa*"] = []
    zeros = 0
    with (local / "nano3pseq_hela_fig3d.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            if row["positive_length_nt"]:
                series["Nano3P-seq / HeLa*"].append(float(row["positive_length_nt"]))
            zeros += row["source_status"] == "zero_value"
    return series, zeros


def render_shuyi_context(
    series: dict, output: Path, zeros: int, shuyi_audit: dict
) -> list[dict]:
    """Plot full-subset histogram percentages and exact full-range empirical CDFs."""
    styles = {
        "Holypoly": ("#4477AA", "-"),
        "Dorado": ("#CC9933", "-"),
        "PolyAnalysis": ("#8866AA", "-"),
        "Nano3P-seq / HeLa*": ("#228877", "-"),
        "Shuyi / 293T-R9 (rep1+rep2)": ("#BB6677", "-"),
        "tailfindr DNA / polyA (rep1+rep2)": ("#882255", "-"),
        "tailfindr DNA / polyT (rep1+rep2)": ("#882255", "--"),
    }
    has_dna = any(name.startswith("tailfindr DNA") for name in series)
    figure, axes = plt.subplots(1, 2, figsize=(16, 9))
    figure.subplots_adjust(left=0.065, right=0.98, top=0.64, bottom=0.32, wspace=0.25)
    figure.suptitle(
        "Length distributions: local cDNA + independent RNA/DNA sources"
        if has_dna
        else "Length distributions: local cDNA + independent sources",
        fontsize=15,
        y=0.97,
    )
    figure.text(
        0.5,
        0.925,
        "Local HEK293T cDNA | published Nano3P-seq HeLa | Shuyi 293T/R9 | "
        "tailfindr DNA LSK108/109: different datasets"
        if has_dna
        else "Local HEK293T cDNA | published Nano3P-seq HeLa | Shuyi 293T/R9: "
        "different datasets",
        ha="center",
        fontsize=11,
    )
    maximum = max(max(v) for v in series.values())
    bins = np.arange(math.ceil(max(maximum, 300) / 5) + 1) * 5
    if len(bins) > 100000:
        raise ValueError("unexpected extreme length; review source before plotting")
    records, hist_rows, handles, labels = [], [], [], []
    for name, values in series.items():
        color, linestyle = styles[name]
        array = np.asarray(values)
        if np.any(~np.isfinite(array)) or np.any(array <= 0):
            raise ValueError("all plotted values must be finite and positive")
        mean, median = float(array.mean()), float(np.median(array))
        hist, _ = np.histogram(array, bins)
        if int(hist.sum()) != len(array):
            raise ValueError("histogram dropped values")
        axes[0].stairs(100 * hist / len(array), bins, color=color, lw=1.5, ls=linestyle)
        for value, style in ((mean, "--"), (median, ":")):
            axes[0].axvline(value, color=color, ls=style, lw=1, alpha=0.8)
        unique, counts = np.unique(array, return_counts=True)
        axes[1].step(
            np.r_[0, unique],
            np.r_[0, counts.cumsum() / len(array)],
            where="post",
            color=color,
            lw=1.5,
            ls=linestyle,
        )
        outside = int((array > 300).sum())
        records.append(
            {
                "series": name,
                "positive_entries": len(array),
                "mean_nt": mean,
                "median_nt": median,
                "above_300nt": outside,
                "q25_nt": float(np.quantile(array, 0.25)),
                "q75_nt": float(np.quantile(array, 0.75)),
                "min_nt": float(array.min()),
                "max_nt": float(array.max()),
            }
        )
        handles.append(Line2D([], [], color=color, lw=1.8, ls=linestyle))
        labels.append(
            f"{name}\nn+={len(array):,}; >300 nt: {outside:,}\n"
            f"Mean={mean:.2f}; Median={median:.2f}"
        )
        hist_rows.extend(
            {
                "series": name,
                "bin_left_nt": left,
                "bin_right_nt": right,
                "count": int(count),
                "percent_of_all_positive": 100 * count / len(array),
            }
            for left, right, count in zip(bins[:-1], bins[1:], hist, strict=True)
        )
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.51, 0.875),
        ncol=3,
        fontsize=9,
        columnspacing=3,
    )
    axes[0].set(
        xlim=(0, 300),
        xlabel="Reported length (nt)",
        ylabel="All positive entries per bin (%)",
        title="a  Length distribution (5-nt bins; focus window)",
    )
    axes[0].legend(
        handles=[
            Line2D([], [], color="#555555", ls=style, label=label)
            for style, label in (("--", "Mean"), (":", "Median"))
        ],
        loc="upper right",
        fontsize=9,
    )
    axes[1].set_xscale("symlog", linthresh=20)
    axes[1].set(
        xlim=(0, None),
        ylim=(0, 1.02),
        xlabel="Reported length (nt; symlog scale)",
        ylabel="Cumulative fraction of positive entries",
        title="b  Full-range empirical CDF",
    )
    notes = [
        f"Local tools: the same {len(series['Holypoly']):,} HEK293T IDs "
        "from the original fixed subset. Independent sources are not read-matched.",
        "Nano3P-seq: published source workbook, Fig3d Species=Human; "
        "*HeLa designation inherited from the original comparison. "
        f"Native zeros: {zeros:,}.",
        "Shuyi: provided 293T/R9 poly(A)caller data, raw worksheet, "
        "polyA_length_293T_total_rep1 / rep2; both replicate lists combined.",
        "Shuyi source has Excel-truncated lists: "
        f"{shuyi_audit['available_length_entries']:,} available entries / "
        f"{shuyi_audit['reported_length_count']:,} reported; "
        "n+ is list entries, not verified unique reads.",
        "Different datasets; descriptive context only, no accuracy/recall ranking. "
        "Missing values are not imputed as zero; replicate summaries are provided.",
        "Mean/median and ECDF retain every positive value, including values "
        "outside the histogram focus window.",
    ]
    if has_dna:
        notes.insert(
            4,
            "tailfindr DNA: paper LSK108/109 standard-basecalling CSV; "
            "valid poly(A)/poly(T) lengths, shown separately; "
            "two replicates combined per type.",
        )
    if "PolyAnalysis" in series:
        notes[-1] += " PolyAnalysis is an uncalibrated interval_samples/30 proxy."
    for index, note in enumerate(notes):
        figure.text(0.065, 0.245 - 0.029 * index, note, fontsize=8.5)
    stem = (
        "length_distributions_with_hela_shuyi_tailfindr_dna"
        if has_dna
        else (
            "five_methods_local_common_positive_with_hela_shuyi_context"
            if "PolyAnalysis" in series
            else "four_methods_local_common_positive_with_hela_shuyi_context"
        )
    )
    for suffix in ("png", "pdf", "svg"):
        figure.savefig(output / f"{stem}.{suffix}", dpi=300, facecolor="white")
    plt.close(figure)
    write_table(output / "length_summary.tsv", records)
    write_table(output / "histogram_source_data.tsv", hist_rows)
    return records


def plan_layout(root: Path) -> list[tuple[Path, Path]]:
    """Classify only known top-level result/source directories."""
    moves = []
    for source in sorted(root.iterdir()):
        if source.is_symlink() or not source.is_dir():
            continue
        if source.name.startswith("fixed-len-tool-comparison-"):
            category = "local/fixed_len"
        elif source.name.startswith("hek293t") and "three-tools" in source.name:
            category = "local/hek293t"
        elif source.name.startswith("nano3p-seq-published-"):
            category = "published/nano3p_seq"
        elif source.name in {"tailfindr_paper_data", "shuyi_polyacaller_data"}:
            category = "sources"
        else:
            continue
        moves.append((source, root / category / source.name))
    return moves


def relocated_path(path: Path, moves: list[tuple[Path, Path]]) -> Path:
    """Map an old path to its new physical location without creating links."""
    for source, destination in moves:
        if path.is_relative_to(source):
            return destination / path.relative_to(source)
    return path


def find_comparison(root: Path, pattern: str, category: str) -> Path:
    """Find one result directory in either the original or organized layout."""
    matches = list(root.glob(pattern)) + list((root / category).glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} directory, found {len(matches)}")
    return matches[0]


def organize(root: Path, moves: list[tuple[Path, Path]], run: str) -> dict:
    """Move result directories unchanged without retaining old-path links."""
    before = {}
    for source, destination in moves:
        if destination.exists():
            raise ValueError(f"destination exists: {destination}")
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError("unexpected symlink inside source tree")
            if path.is_file():
                relative = path.relative_to(root)
                before[str(relative)] = fingerprint(path)
    audit = {
        "created_utc": datetime.now(UTC).isoformat(),
        "files_before": before,
        "moves": [],
        "old_paths_retained_as_relative_symlinks": False,
    }
    write_json(root / f"organization_before_{run}.json", audit)
    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        audit["moves"].append(
            {
                "from": str(source.relative_to(root)),
                "to": str(destination.relative_to(root)),
            }
        )
    for relative, original in before.items():
        current = fingerprint(relocated_path(root / relative, moves))
        if any(current[k] != original[k] for k in ("bytes", "sha256")):
            raise ValueError(f"moved file changed: {relative}")
    audit["all_original_file_bytes_verified_unchanged"] = True
    write_json(root / f"organization_verified_{run}.json", audit)
    return audit


def main() -> None:
    """Dry-run by default; execute only with explicit external-work-root validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--exclude-polyanalysis", action="store_true")
    parser.add_argument("--tailfindr-dna", type=Path)
    parser.add_argument("--tailfindr-dna-barcodes", type=Path)
    parser.add_argument(
        "--shuyi-truncated-policy",
        choices=("error", "prefix", "exclude"),
        default="error",
    )
    args = parser.parse_args()
    if args.tailfindr_dna_barcodes and not args.tailfindr_dna:
        parser.error("--tailfindr-dna-barcodes requires --tailfindr-dna")
    config = load_config(args.config)
    if config.paths.work_root is None or config.paths.data_root is None:
        raise ValueError("configure work_root and data_root")
    work, data = config.paths.work_root.resolve(), config.paths.data_root.resolve()
    root = args.root.resolve(strict=True)
    project = Path(__file__).resolve().parents[1]
    if (
        root == work
        or not root.is_relative_to(work)
        or root.is_relative_to(data)
        or root.is_relative_to(project)
    ):
        raise ValueError("comparison root must be outside raw data/repo, inside work")
    if Path(args.run_name).name != args.run_name or args.run_name in {".", ".."}:
        raise ValueError("run-name must be one directory name")
    moves = plan_layout(root)
    local = find_comparison(root, "hek293t*three-tools*", "local/hek293t")
    fixed = find_comparison(root, "fixed-len-tool-comparison-*", "local/fixed_len")
    nano = find_comparison(root, "nano3p-seq-published-*", "published/nano3p_seq")
    tail_dir = find_comparison(root, "tailfindr_paper_data", "sources")
    shuyi_dir = find_comparison(root, "shuyi_polyacaller_data", "sources")
    tail = next(tail_dir.glob("rna-*.csv"))
    shuyi = next(shuyi_dir.glob("*/*.xlsx"))
    local_tools = tuple(
        tool
        for tool in TOOLS
        if not (args.exclude_polyanalysis and tool == "PolyAnalysis")
    )
    inputs = [
        tail,
        shuyi,
        fixed / "per_read_comparison.tsv",
        nano / "source_data/02_Fig3c.csv",
        local / "common_read_comparison.tsv",
        local / "nano3pseq_hela_fig3d.tsv",
        local / "summary.json",
        nano / "length_distribution_combined/group_distribution_summary.csv",
        local / "four_tools_length_summary.tsv",
    ]
    if args.tailfindr_dna is not None:
        inputs.append(args.tailfindr_dna.resolve(strict=True))
    if args.tailfindr_dna_barcodes is not None:
        inputs.append(args.tailfindr_dna_barcodes.resolve(strict=True))
    out = root / "comparisons" / args.run_name
    if out.exists() or out.is_symlink():
        raise ValueError("new comparison output already exists")
    print(
        json.dumps(
            {
                "moves": [
                    {"from": str(a.relative_to(root)), "to": str(b.relative_to(root))}
                    for a, b in moves
                ],
                "input_bytes": sum(p.stat().st_size for p in inputs),
                "output": str(out),
                "apply": args.apply,
            },
            indent=2,
        ),
        flush=True,
    )
    if not args.apply:
        return
    before = [fingerprint(p) for p in inputs]
    organization = (
        organize(root, moves, args.run_name) if moves else {"files_before": {}}
    )
    inputs = [relocated_path(p, moves) for p in inputs]
    local, tail, shuyi = (relocated_path(p, moves) for p in (local, tail, shuyi))
    out.mkdir(parents=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    print("Reading existing tailfindr CSV and Shuyi workbook...", flush=True)
    tail_groups = read_tailfindr(tail)
    shuyi_groups, shuyi_audit = read_shuyi(
        shuyi, truncated_policy=args.shuyi_truncated_policy
    )
    dna_groups, dna_audit = (
        read_tailfindr_dna(inputs[9]) if args.tailfindr_dna else ({}, {})
    )
    grouped_dna, grouped_audit = {}, {}
    if args.tailfindr_dna_barcodes:
        print("Matching DNA barcodes by replicate and read ID...", flush=True)
        dna_out = out / "tailfindr_dna_six_lengths"
        dna_out.mkdir()
        grouped_dna, grouped_audit = read_barcoded_dna(
            inputs[9], inputs[10], joined_output=dna_out / "barcode_matched_lengths.tsv"
        )
        render_dna_replicates(grouped_dna, dna_out)
    write_json(
        out / "source_audit.json",
        {
            "tailfindr": group_audit(tail_groups),
            "shuyi": {**shuyi_audit, "replicates": group_audit(shuyi_groups)},
            "tailfindr_dna": {**dna_audit, "groups": group_audit(dna_groups)},
            "tailfindr_dna_barcoded": {
                **grouped_audit,
                "groups": group_audit(grouped_dna),
            },
            "tailfindr_rna_nominal_groups": {
                "user_provided_expected_lengths_nt": DNA_NOMINAL_LENGTHS,
                "read_to_nominal_mapping_available": False,
                "display_policy": "replicate mixtures; no inferred group assignment",
            },
        },
    )
    print(
        "Source summaries:",
        json.dumps(
            {
                "tailfindr": group_audit(tail_groups),
                "shuyi": group_audit(shuyi_groups),
                "shuyi_count_mismatches": len(shuyi_audit["count_mismatches"]),
                "tailfindr_dna": group_audit(dna_groups),
            }
        ),
        flush=True,
    )
    if shuyi_audit["count_mismatches"]:
        raise ValueError("Shuyi count/list mismatch; see source audit before plotting")
    fixed_out, shuyi_out = out / "fixed_len_with_tailfindr", out / "hek293t_with_shuyi"
    fixed_out.mkdir()
    shuyi_out.mkdir()
    combined_tail_groups = {
        name: {**group, "display_label": f"RNA\n{name}"}
        for name, group in tail_groups.items()
    } | (pool_dna_replicates(grouped_dna) if grouped_dna else dna_groups)
    fixed_records = plot_context(
        inputs[2],
        inputs[3],
        fixed_out,
        combined=True,
        focus_max_nt=220,
        tailfindr_groups=combined_tail_groups,
        local_tools=local_tools,
    )
    series, zeros = read_image_cohorts(local)
    if args.exclude_polyanalysis:
        del series["PolyAnalysis"]
    series["Shuyi / 293T-R9 (rep1+rep2)"] = [
        value for group in shuyi_groups.values() for value in group["values"]
    ]
    for poly_type in sorted({group["poly_type"] for group in dna_groups.values()}):
        series[f"tailfindr DNA / {poly_type} (rep1+rep2)"] = [
            value
            for group in dna_groups.values()
            if group["poly_type"] == poly_type
            for value in group["values"]
        ]
    shuyi_records = render_shuyi_context(series, shuyi_out, zeros, shuyi_audit)
    verify_original_statistics(
        inputs[7], inputs[8], fixed_records, shuyi_records, local_tools=local_tools
    )
    after = [fingerprint(p) for p in inputs]
    if any(
        original[key] != current[key]
        for original, current in zip(before, after, strict=True)
        for key in ("bytes", "sha256")
    ):
        raise ValueError("an input file changed")
    write_json(
        out / "figure_contract_and_qa.json",
        {
            "created_utc": datetime.now(UTC).isoformat(),
            "inputs": before,
            "relocated_inputs": after,
            "input_sources_unchanged": True,
            "script_fingerprints": [
                fingerprint(Path(__file__)),
                fingerprint(Path(__file__).with_name("plot_nano3p_context.py")),
            ],
            "local_tools": local_tools,
            "local_cohort": "Fixed original shared-positive subset; "
            "not recomputed after removing a plotted tool",
            "dna_poly_types_kept_separate": True,
            "dna_barcode_groups_assigned": bool(grouped_dna),
            "dna_nominal_labels": grouped_audit,
            "rna_expected_lengths_nt": DNA_NOMINAL_LENGTHS,
            "rna_read_to_nominal_mapping_available": False,
            "grouped_dna_replicates": "pooled within nominal length and A/T; "
            "separate in DNA supplement",
            "pooled_dna_image_context": "all native valid positive lengths; "
            "no barcode selection, preserving previous cohort",
            "tailfindr_dna_source": "User-provided tailfindr paper LSK108/109 "
            "DNA standard-basecalling CSV"
            if dna_groups
            else None,
            "organization_performed": bool(moves),
            "claim": "Describe length distributions from independent datasets; "
            "no matched cross-cohort accuracy or recall inference.",
            "archetype": "quantitative grid",
            "backend": "python/matplotlib",
            "exports": ["PNG 300 dpi", "PDF", "SVG editable text"],
            "fixed_len_groups": fixed_records,
            "image_context_series": shuyi_records,
            "tailfindr_source": "User-supplied tailfindr paper CSV; RNA001 standard "
            "basecalling, two replicate labels; exact DOI not supplied.",
            "shuyi_source": "Shuyi-provided 293T/R9 workbook; raw length lists; "
            "poly(A)caller attribution from provided source directory; "
            "exact protocol/tool version not provided.",
            "shuyi_replicates_combined_within_source_only": True,
            "different_datasets_pooled": False,
            "unique_shuyi_reads_verified": False,
            "zero_imputation": False,
            "full_positive_statistics_retained": True,
            "original_group_and_image_statistics_verified_unchanged": True,
            "visual_review": "pending",
            "no_gpu_model_manifest_or_raw_signal_used": True,
            "python": platform.python_version(),
            "matplotlib": matplotlib.__version__,
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "command": sys.argv,
            "original_file_count_verified": len(organization["files_before"]),
        },
    )
    readme = (
        "# 工具比较目录\n\n"
        "- `local/fixed_len/`：原定长三工具比较。\n"
        "- `local/hek293t/`：原 HEK293T 三工具及 Nano3P-seq 背景图。\n"
        "- `published/nano3p_seq/`：原 Nano3P-seq Fig3c 来源及比较图。\n"
        "- `sources/tailfindr_paper_data/`：用户提供的 tailfindr 论文结果 CSV。\n"
        "- `sources/shuyi_polyacaller_data/`：Shuyi 提供的 293T/R9 Excel。\n"
        f"- [本次更新](comparisons/{args.run_name}/README.md)。\n\n"
        "文件已移动到上述实体目录，不保留旧路径或兼容符号链接。"
        "迁移前后逐文件 SHA256 一致，详见 organization_verified 审计文件。\n"
    )
    root_readme = root / "README.md"
    if root_readme.exists():
        previous = root_readme.read_text()
        root_readme.write_text(
            previous + f"\n最新比较：`comparisons/{args.run_name}/`。"
            "本版仅展示所选本地工具，并加入 tailfindr DNA 的独立 A/T 结果。\n"
        )
    else:
        with root_readme.open("x") as stream:
            stream.write(readme)
    with (out / "README.md").open("x") as stream:
        if dna_groups:
            stream.write(
                "# 更新的独立来源分布比较\n\n"
                f"本地工具：{', '.join(local_tools)}。\n\n"
                "`fixed_len_with_tailfindr/`：定长 + Nano3P-seq + tailfindr RNA/DNA，"
                "含完整范围及 0–220 nt 图；DNA A/T 分开展示。\n\n"
                + (
                    "DNA 按新增 barcode 表分为预期 10、30、40、60、100、150 nt 六组，"
                    "置于实际 nt 数值横轴，同一长度和方向内合并两个重复。"
                    "`tailfindr_dna_six_lengths/` 另含两个重复分开的图、分组统计"
                    "及匹配条目表。标称长度是构建标签，不是逐 read 真值。\n\n"
                    "匹配键为 replicate + read_id；长度来自 standard basecalling，"
                    "barcode 来自 flipflop basecalling。要求 barcode 通过阈值且无并列，"
                    "仅接受六种预期长度；A/T 方向须一致；长度有效且为有限正值。"
                    "barcode 原生过长/过短 read 标记仅审计，不另加长度筛选。\n\n"
                    if grouped_dna
                    else "DNA 类别未指定标称 nt。\n\n"
                )
                + "RNA 预期长度同为 10、30、40、60、100、150 nt（用户提供），"
                "但目前 RNA CSV 无逐 read barcode/构建分组；图中保留各重复混合分布，"
                "待提供对应表后才能拆为六组，未按预测长度反推标签。\n\n"
                "`hek293t_with_shuyi/`：保留原图本地 4,402 条 read 子集，"
                "加入 Shuyi 两重复合并，以及 tailfindr DNA A/T "
                "各自两重复合并的分布；这张混合分布图保持原 DNA 全部有效正值队列，"
                "不因新增 barcode 表重新筛选。\n\n"
                "tailfindr DNA 来源：用户提供的 LSK108/109 "
                "standard-basecalling 论文结果 CSV。"
                "仅绘制 tail_is_valid=TRUE、read_type=polyA/polyT 的有限正长度，"
                "A/T 不混合，失败、缺失和零值另行统计。"
                "RNA 来源为 RNA001 两重复 CSV；Nano3P-seq 来源 Fig3c/Fig3d Human 组"
                "（沿用原图 HeLa 指定）；Shuyi 来源为提供的 293T/R9 Excel raw 表。\n\n"
                f"Shuyi 保留 {shuyi_audit['available_length_entries']:,} 条目，"
                f"原计数 {shuyi_audit['reported_length_count']:,}；"
                "部分 Excel 单元格截断，只保留完整数字，"
                "缺少 read ID，无法验证唯一 read 数。\n\n"
                "不同来源是独立数据集，不进行 read 匹配或准确率/召回率排名。"
                "均值、中位数、ECDF 均使用全部正值。"
                "PNG/PDF/SVG 及来源审计保存在本目录；"
                "不创建符号链接，也不更改历史数据或图件。\n"
            )
        else:
            stream.write(
                "# 新增跨数据来源分布图\n\n"
                "- [定长 + Nano3P-seq + tailfindr（0–220 nt）]"
                "(fixed_len_with_tailfindr/"
                "fixed_len_with_nano3p_tailfindr_combined_focus_0_220nt.png)\n"
                "- [定长 + Nano3P-seq + tailfindr（完整范围）]"
                "(fixed_len_with_tailfindr/fixed_len_with_nano3p_tailfindr_combined.png)\n"
                "- [原图加入 Shuyi 数据]"
                "(hek293t_with_shuyi/"
                "five_methods_local_common_positive_with_hela_shuyi_context.png)\n\n"
                "均有 PDF/SVG。tailfindr 两个重复单独展示于右侧 RNA 类别，"
                "未指定标称长度；"
                "原有 cDNA 数值横轴间距保留。Shuyi 使用 raw 表的两次重复长度列表合并，"
                "不使用转录本中位数；逐重复统计在 source_audit.json。"
                "部分来源列表被 Excel 截断，图中注明可用条目数与原计数；"
                "截断处理策略与逐单元格记录保存在 source_audit.json。"
                "列表缺少 read ID，n+ 表示长度条目数，"
                "不能验证独立 read 数或转录本间重复。\n\n"
                "来源：本地既有三工具输出；Nano3P-seq 发表工作簿 Fig3c/Fig3d（Human 组"
                "沿用原图的 HeLa 指定）；用户提供的 tailfindr RNA001 两重复结果 CSV；"
                "Shuyi 提供的 293T/R9 工作簿 raw 表 polyA_length_293T_total_rep1/rep2。"
                "后两者精确论文 DOI/工具版本未提供；"
                "poly(A)caller 归属按来源文件夹记录。\n\n"
                "不同来源未匹配 read，不推断准确率、召回率或工具排名。"
                "零值和缺失值分开审计，"
                "不填零；均值、中位数和 ECDF 使用全部正值，放大图不重算统计。"
                "PolyAnalysis 保留 interval_samples/30 未校准代理值定义。"
                "输入路径、SHA256、命令、环境和图契约在 figure_contract_and_qa.json。\n"
            )
    print(f"Finished: {out}", flush=True)


if __name__ == "__main__":
    main()
