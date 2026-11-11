"""Add published Nano3P-seq HeLa distributions to existing HEK293T results."""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import posixpath
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from holypoly.core.config import load_config  # noqa: E402
from holypoly.evaluation.fixed_len import TOOLS, fingerprint  # noqa: E402
from holypoly.evaluation.xlsx_csv import (  # noqa: E402
    NS,
    RELATIONSHIP_ID,
    _rows,
    _shared_strings,
)

NANO = "Nano3P-seq / HeLa*"
COLORS = {
    "Holypoly": "#4477AA",
    "Dorado": "#CC9933",
    "PolyAnalysis": "#8866AA",
    NANO: "#228877",
}
SOURCE_COLUMNS = [
    "Read_id",
    "Quality",
    "Gene_Name",
    "Biotype",
    "tail_length",
    "Species",
]


def extract_hela(
    source: Path, destination: Path, *, source_species: str = "HeLa"
) -> tuple[list[float], dict]:
    """Stream Fig3d and select an explicit source species without quality filters.

    Original cell text, worksheet row and native zero/missing values are retained.
    Values are published table rows, not an inferred denominator of attempted reads.
    """
    positives: list[float] = []
    species_counts: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    with ZipFile(source) as book:
        sheets = ET.fromstring(book.read("xl/workbook.xml"))
        matches = [
            s
            for s in sheets.findall("s:sheets/s:sheet", NS)
            if s.get("name") == "Fig3d"
        ]
        if len(matches) != 1:
            raise ValueError("expected exactly one Fig3d worksheet")
        relations = {
            e.get("Id"): e.attrib["Target"]
            for e in ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        }
        target = relations[matches[0].attrib[RELATIONSHIP_ID]]
        target = (
            posixpath.normpath("xl/" + target)
            if not target.startswith("/")
            else target[1:]
        )
        if not target.startswith("xl/"):
            raise ValueError("worksheet target outside workbook")
        strings = _shared_strings(book)
        header = None
        preamble = []
        with destination.open("x", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                delimiter="\t",
                fieldnames=[
                    "source_row",
                    *SOURCE_COLUMNS,
                    "source_status",
                    "positive_length_nt",
                ],
            )
            writer.writeheader()
            for number, cells in _rows(book, target, strings):
                if header is None:
                    if sum(bool(cell) for cell in cells) < 2:
                        preamble.append({"row": number, "values": cells})
                        continue
                    if len(cells) != len(SOURCE_COLUMNS) or set(cells) != set(
                        SOURCE_COLUMNS
                    ):
                        raise ValueError("unexpected Fig3d columns")
                    header = cells
                    continue
                if len(cells) > len(header):
                    raise ValueError("data beyond Fig3d header columns")
                row = dict(
                    zip(header, cells + [""] * (len(header) - len(cells)), strict=True)
                )
                species = row["Species"].strip()
                species_counts[species] += 1
                if species.casefold() != source_species.casefold():
                    continue
                text = row["tail_length"].strip()
                value = float(text) if text else None
                status = (
                    "missing_value"
                    if value is None
                    else "nonfinite_value"
                    if not math.isfinite(value)
                    else "negative_value"
                    if value < 0
                    else "zero_value"
                    if value == 0
                    else "positive_length"
                )
                statuses[status] += 1
                if status == "positive_length":
                    positives.append(value)
                writer.writerow(
                    {
                        "source_row": number,
                        **row,
                        "source_status": status,
                        "positive_length_nt": value
                        if status == "positive_length"
                        else "",
                    }
                )
    if not positives:
        raise ValueError(
            f"Fig3d contains no positive {source_species} lengths; "
            "no species substitution"
        )
    return positives, {
        "sheet": "Fig3d",
        "species_filter": "Species.strip().casefold() == "
        f"{source_species.casefold()!r}",
        "source_species": source_species,
        "display_cell_line": "HeLa",
        "display_label_basis": "User explicitly identifies the Fig3d Human group "
        "as HeLa"
        if source_species == "Human"
        else "Source species label",
        "quality_filter": None,
        "biotype_filter": None,
        "columns": header,
        "preamble": preamble,
        "worksheet_species_counts": dict(species_counts),
        "hela_source_status_counts": dict(statuses),
        "hela_reported_rows": sum(statuses.values()),
        "denominator": "HeLa rows in the published table, not all attempted reads",
        "row_identity": "worksheet row; no deduplication or matching to HEK293T IDs",
    }


def read_local(directory: Path) -> tuple[dict, dict, dict, dict]:
    """Reuse extracted native positives and shared local IDs without reading BAM."""
    summary = json.loads((directory / "summary.json").read_text())
    all_values = {tool: [] for tool in TOOLS}
    shared = {tool: [] for tool in TOOLS}
    paired = {tool: [] for tool in TOOLS}
    for tool in TOOLS:
        with (directory / f"{tool.lower()}_results.tsv").open() as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["positive_length_nt"]:
                    all_values[tool].append(float(row["positive_length_nt"]))
    common_count = 0
    with (directory / "common_read_comparison.tsv").open() as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            common_count += 1
            values = {tool: row[f"{tool}_positive_length_nt"] for tool in TOOLS}
            for tool, text in values.items():
                if text:
                    shared[tool].append(float(text))
            if all(values.values()):
                for tool, text in values.items():
                    paired[tool].append(float(text))
    if common_count != summary["common_reads"]:
        raise ValueError("local shared-ID denominator changed")
    for tool in TOOLS:
        if (
            len(all_values[tool])
            != summary["tools"][tool]["all_status_counts"]["positive_length"]
        ):
            raise ValueError("local positive count disagrees with summary")
        if (
            len(shared[tool])
            != summary["tools"][tool]["common_status_counts"]["positive_length"]
        ):
            raise ValueError("local shared positive count disagrees with summary")
        if len(paired[tool]) != summary["common_positive_reads"]:
            raise ValueError("local shared-positive count disagrees with summary")
    return all_values, shared, paired, summary


def render_context(
    series: dict,
    output: Path,
    stem: str,
    title: str,
    cohort_note: str,
    zero_count: int,
    bin_width: float,
    focus: float,
    source_species: str = "HeLa",
) -> tuple[list, list]:
    """Draw one histogram and exact ECDF with a shared cohort-specific legend."""
    if any(
        not math.isfinite(v) or v <= 0 for values in series.values() for v in values
    ):
        raise ValueError("all plotted lengths must be finite and positive")
    maximum = max((max(values, default=0) for values in series.values()), default=0)
    number = max(1, math.ceil(max(maximum, focus) / bin_width))
    if number > 10000:
        raise ValueError("too many bins; increase bin width")
    bins = np.arange(number + 1) * bin_width
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.5))
    fig.subplots_adjust(left=0.07, right=0.98, top=0.70, bottom=0.30, wspace=0.25)
    fig.suptitle(title, fontsize=13, y=0.97)
    fig.text(
        0.5,
        0.915,
        "Local tools: HEK293T | Nano3P-seq: published HeLa, different dataset",
        ha="center",
        fontsize=10,
    )
    handles, labels, stats, hist_rows = [], [], [], []
    for tool, values in series.items():
        if not values:
            continue
        array = np.asarray(values)
        mean, median = float(np.mean(array)), float(np.median(array))
        outside = int((array > focus).sum())
        hist, _ = np.histogram(array, bins=bins)
        if int(hist.sum()) != len(values):
            raise ValueError("histogram discarded positive values")
        color = COLORS[tool]
        axes[0].stairs(100 * hist / len(values), bins, color=color, linewidth=1.5)
        axes[0].axvline(mean, color=color, linestyle="--", linewidth=1, alpha=0.85)
        axes[0].axvline(median, color=color, linestyle=":", linewidth=1, alpha=0.85)
        xs, counts = np.unique(array, return_counts=True)
        axes[1].step(
            np.r_[0, xs],
            np.r_[0, counts.cumsum() / len(values)],
            where="post",
            color=color,
            linewidth=1.5,
        )
        name = tool + " (different dataset)" if tool == NANO else tool
        handles.append(Line2D([], [], color=color, linewidth=1.5))
        labels.append(
            f"{name}\nn={len(values):,}; >{focus:g} nt: {outside:,}\n"
            f"Mean={mean:.2f}; Median={median:.2f}"
        )
        stats.append(
            {
                "tool": tool,
                "positive_rows": len(values),
                "above_focus_rows": outside,
                "mean_nt": mean,
                "median_nt": median,
                "q25_nt": float(np.quantile(array, 0.25)),
                "q75_nt": float(np.quantile(array, 0.75)),
                "min_nt": float(array.min()),
                "max_nt": float(array.max()),
            }
        )
        for left, right, count in zip(bins[:-1], bins[1:], hist, strict=True):
            hist_rows.append(
                {
                    "tool": tool,
                    "bin_left_nt": left,
                    "bin_right_nt": right,
                    "count": int(count),
                    "percent_of_all_positive": 100 * count / len(values),
                }
            )
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.87),
        ncol=min(4, len(handles)),
        fontsize=8,
        columnspacing=2,
    )
    axes[0].set(
        xlim=(0, focus),
        xlabel="Reported length / proxy (nt)",
        ylabel="All positive outputs per bin (%)",
        title=f"a  Length distribution ({bin_width:g}-nt bins; focus window)",
    )
    axes[0].legend(
        handles=[
            Line2D([], [], color="#555555", linestyle=style, label=label)
            for style, label in (("--", "Mean"), (":", "Median"))
        ],
        loc="upper right",
        fontsize=8,
    )
    axes[1].set_xscale("symlog", linthresh=20)
    axes[1].set(
        xlim=(0, None),
        ylim=(0, 1.02),
        xlabel="Reported length / proxy (nt; symlog scale)",
        ylabel="Cumulative fraction of positive outputs",
        title="b  Full-range empirical CDF",
    )
    notes = [
        cohort_note,
        f"*HeLa: {'user-designated ' if source_species == 'Human' else ''}"
        f"Fig3d Species={source_species} group. Nano3P-seq is a different dataset; "
        "no four-way read matching or accuracy ranking.",
        "Positive outputs only. Nano3P-seq native zeros retained separately: "
        f"{zero_count:,}. "
        "Missing/no-candidate results are never imputed as zero.",
        "PolyAnalysis: partial sequential results; "
        "uncalibrated interval_samples/30 proxy. "
        "Mean/median include every positive value, including above the focus window.",
    ]
    for index, note in enumerate(notes):
        fig.text(0.07, 0.205 - 0.038 * index, note, fontsize=8)
    for extension in ("png", "pdf", "svg"):
        fig.savefig(output / f"{stem}.{extension}", dpi=300, facecolor="white")
    plt.close(fig)
    return stats, hist_rows


def save_table(path: Path, rows: list[dict]) -> None:
    """Save tab-separated descriptive statistics or complete histogram bins."""
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Write new four-method artifacts alongside existing three-tool results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    parser.add_argument("--local-results", type=Path, required=True)
    parser.add_argument("--nano3p-xlsx", type=Path, required=True)
    parser.add_argument("--nano3p-species", choices=("HeLa", "Human"), default="HeLa")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bin-width-nt", type=float, default=5)
    parser.add_argument("--focus-max-nt", type=float, default=300)
    args = parser.parse_args()
    if any(
        not math.isfinite(v) or v <= 0 for v in (args.bin_width_nt, args.focus_max_nt)
    ):
        raise ValueError("bin width and focus must be finite and positive")
    config = load_config(args.config)
    if config.paths.work_root is None or config.paths.data_root is None:
        raise ValueError("configure external work_root and data_root")
    work, data = config.paths.work_root.resolve(), config.paths.data_root.resolve()
    output = (work / args.output).resolve()
    local = (work / args.local_results).resolve(strict=True)
    source = args.nano3p_xlsx.resolve(strict=True)
    project = Path(__file__).resolve().parents[1]
    if (
        output == work
        or not output.is_relative_to(work)
        or output.is_relative_to(data)
        or output.is_relative_to(project)
    ):
        raise ValueError("output must be an external work-root child outside raw data")
    stems = [
        "four_tools_all_available",
        "four_tools_local_common_with_hela_context",
        "four_tools_local_common_positive_with_hela_context",
        "nano3pseq_hela_polya_distribution",
    ]
    new_tables = [
        "nano3pseq_hela_fig3d.tsv",
        "four_tools_length_summary.tsv",
        "four_tools_histogram_source_data.tsv",
        "four_tools_summary.json",
    ]
    destinations = [output / name for name in new_tables]
    destinations += [
        output / f"{stem}.{ext}" for stem in stems for ext in ("png", "pdf", "svg")
    ]
    if any(path.exists() for path in destinations):
        raise FileExistsError(
            "new four-tool artifact already exists; original artifacts remain untouched"
        )
    inputs = [source, local / "summary.json", local / "common_read_comparison.tsv"]
    inputs += [local / f"{tool.lower()}_results.tsv" for tool in TOOLS]
    if any(path.resolve() in inputs for path in destinations):
        raise ValueError("output collides with input")
    before = [fingerprint(path) for path in inputs]
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    hela, audit = extract_hela(
        source, destinations[0], source_species=args.nano3p_species
    )
    print(json.dumps(audit, ensure_ascii=False), flush=True)
    all_values, shared, paired, summary = read_local(local)
    zero = audit["hela_source_status_counts"].get("zero_value", 0)
    cohorts = [
        (
            "all_available",
            {**all_values, NANO: hela},
            stems[0],
            "Poly(A) length distributions: four methods",
            "Each local tool uses all of its available HEK293T positive outputs; "
            "Nano3P-seq uses all positive HeLa rows from the published source table.",
        ),
        (
            "local_common_ids_with_external_hela",
            {**shared, NANO: hela},
            stems[1],
            "Four-method context: local shared read IDs + independent HeLa",
            f"Local tools: {summary['common_reads']:,} shared HEK293T IDs, "
            "each tool's own positive subset. "
            "Nano3P-seq: the complete independent HeLa positive subset.",
        ),
        (
            "local_common_positive_with_external_hela",
            {**paired, NANO: hela},
            stems[2],
            "Four-method context: local all-positive reads + independent HeLa",
            f"Local tools: the same {summary['common_positive_reads']:,} "
            "HEK293T IDs positive in all three. "
            "Nano3P-seq: the complete independent HeLa positive subset.",
        ),
        (
            "published_hela",
            {NANO: hela},
            stems[3],
            "Nano3P-seq HeLa: published Fig3d poly(A) length distribution",
            f"All positive rows with source Species={args.nano3p_species}; "
            "no quality or biotype filtering. "
            "This population is independent of the local HEK293T experiment.",
        ),
    ]
    statistics, bins = [], []
    for cohort, series, stem, title, note in cohorts:
        stats, hist = render_context(
            series,
            output,
            stem,
            title,
            note,
            zero,
            args.bin_width_nt,
            args.focus_max_nt,
            args.nano3p_species,
        )
        statistics.extend({"cohort": cohort, **row} for row in stats)
        bins.extend({"cohort": cohort, **row} for row in hist)
    save_table(output / new_tables[1], statistics)
    save_table(output / new_tables[2], bins)
    if [fingerprint(path) for path in inputs] != before:
        raise ValueError("input files changed during analysis")
    metadata = {
        "claim": "Describe four native positive-length distributions; "
        "published Nano3P-seq HeLa is a different dataset, "
        "not a matched fourth tool run.",
        "archetype": "quantitative grid",
        "backend": "python/matplotlib",
        "inputs": before,
        "sources_unchanged": True,
        "nano3pseq": audit,
        "local_common_reads": summary["common_reads"],
        "local_common_positive_reads": summary["common_positive_reads"],
        "four_way_shared_read_cohort": None,
        "cohorts_pooled": False,
        "accuracy_computed": False,
        "bin_width_nt": args.bin_width_nt,
        "histogram_focus_nt": [0, args.focus_max_nt],
        "histogram_denominator": "all positive values of each plotted series",
        "ecdf": "exact full-range symlog, linear threshold 20 nt",
        "mean_median_denominator": "complete positive subset of each tool/cohort",
        "zero_imputation": False,
        "local_limitations": summary["limitations"],
        "figure_stems": stems,
        "visual_review": "pending",
        "python_version": platform.python_version(),
        "matplotlib_version": matplotlib.__version__,
        "no_gpu_model_manifest_or_raw_signal_used": True,
        "finished_utc": datetime.now(UTC).isoformat(),
    }
    (output / new_tables[3]).write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Finished: {output}")


if __name__ == "__main__":
    main()
