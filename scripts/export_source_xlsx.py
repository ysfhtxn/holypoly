"""Convert a published source workbook to separate, lossless-value CSV tables."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from holypoly.core.config import load_config
from holypoly.evaluation.fixed_len import fingerprint
from holypoly.evaluation.xlsx_csv import export_workbook


def main() -> None:
    """Export only to a fresh external work directory and record provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/local.yaml"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sheets", nargs="+", help="Only export these worksheet names")
    args = parser.parse_args()
    config = load_config(args.config)
    if config.paths.work_root is None or config.paths.data_root is None:
        raise ValueError("configure work_root and data_root")
    work, data = config.paths.work_root.resolve(), config.paths.data_root.resolve()
    source = args.input.resolve(strict=True)
    output = (work / args.output).resolve()
    project = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or output == work
        or not output.is_relative_to(work)
        or output.is_relative_to(data)
        or output.is_relative_to(project)
        or source.is_relative_to(output)
    ):
        raise ValueError("output must be a new work-root child outside data/repo/input")
    before = fingerprint(source)
    sheets = export_workbook(source, output, sheet_names=args.sheets)
    if fingerprint(source) != before:
        raise ValueError("source changed during conversion")
    metadata = {
        "created_utc": datetime.now(UTC).isoformat(),
        "source": before,
        "command": ["scripts/export_source_xlsx.py", *sys.argv[1:]],
        "source_unchanged": True,
        "sheets": sheets,
        "normalization": "Only title preambles excluded; blank cells and numeric "
        "XML strings preserved. No rounding, filtering, label mapping or imputation.",
        "csv_fingerprints": [fingerprint(output / s["csv"]) for s in sheets],
    }
    (output / "conversion.json").write_text(json.dumps(metadata, indent=2) + "\n")
    for sheet in sheets:
        print(
            f"{sheet['sheet']}: {sheet['data_rows']:,} rows -> {sheet['csv']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
