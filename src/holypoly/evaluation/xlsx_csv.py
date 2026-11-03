"""Stream plain XLSX source tables to CSV without optional Excel dependencies.

Numeric cached values are retained as their original XML strings; no rounding,
date interpretation, formula evaluation or biological relabeling is performed.
"""

from __future__ import annotations

import csv
import posixpath
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET
from zipfile import ZipFile

NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
RELATIONSHIP_ID = (
    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
)


def _shared_strings(book: ZipFile) -> list[str]:
    """Read string values in their indexed order while discarding XML nodes."""
    if "xl/sharedStrings.xml" not in book.namelist():
        return []
    values = []
    with book.open("xl/sharedStrings.xml") as stream:
        events = ET.iterparse(stream, events=("start", "end"))
        _, root = next(events)
        for event, element in events:
            if event == "end" and element.tag == f"{{{NS['s']}}}si":
                values.append(
                    "".join(t.text or "" for t in element.findall(".//s:t", NS))
                )
                root.clear()
    return values


def _column_index(coordinate: str) -> int:
    """Convert an Excel cell coordinate to a zero-based column index."""
    match = re.fullmatch(r"([A-Z]+)[1-9][0-9]*", coordinate)
    if match is None:
        raise ValueError(f"invalid Excel coordinate: {coordinate}")
    index = 0
    for letter in match[1]:
        index = 26 * index + ord(letter) - ord("A") + 1
    return index - 1


def _rows(
    book: ZipFile, target: str, strings: list[str]
) -> Iterator[tuple[int, list[str]]]:
    """Yield populated rows, restoring empty cells and cached scalar values."""
    with book.open(target) as stream:
        sheet_data = None
        for event, element in ET.iterparse(stream, events=("start", "end")):
            if event == "start" and element.tag == f"{{{NS['s']}}}sheetData":
                sheet_data = element
            if event != "end" or element.tag != f"{{{NS['s']}}}row":
                continue
            values: list[str] = []
            for cell in element.findall("s:c", NS):
                index = _column_index(cell.attrib["r"])
                values.extend([""] * max(0, index + 1 - len(values)))
                value = cell.find("s:v", NS)
                if cell.find("s:f", NS) is not None and value is None:
                    raise ValueError(
                        "formula has no cached value; cannot export safely"
                    )
                kind = cell.get("t", "n")
                text = value.text or "" if value is not None else ""
                if kind == "s":
                    text = strings[int(text)]
                elif kind == "inlineStr":
                    text = "".join(t.text or "" for t in cell.findall(".//s:t", NS))
                values[index] = text
            if any(values):
                yield int(element.attrib["r"]), values
            element.clear()
            if sheet_data is not None:
                sheet_data.clear()


def export_workbook(
    source: Path, output: Path, *, sheet_names: list[str] | None = None
) -> list[dict[str, Any]]:
    """Export each worksheet after its first multi-cell header into a new CSV.

    Single-cell title rows and empty rows preceding the header are recorded in
    metadata rather than mixed into the tabular CSV. Callers enforce work-root
    safety. The output directory must not exist; partial failures remain visible.
    """
    metadata = []
    with ZipFile(source) as book:
        workbook = ET.fromstring(book.read("xl/workbook.xml"))
        sheets = workbook.findall("s:sheets/s:sheet", NS)
        if sheet_names is not None and (
            not sheet_names or set(sheet_names) - {s.attrib["name"] for s in sheets}
        ):
            raise ValueError("requested worksheets are empty or absent from workbook")
        output.mkdir(parents=True, exist_ok=False)
        strings = _shared_strings(book)
        relations = {
            el.attrib["Id"]: el.attrib["Target"]
            for el in ET.fromstring(book.read("xl/_rels/workbook.xml.rels"))
        }
        for index, sheet in enumerate(sheets, 1):
            name = sheet.attrib["name"]
            if sheet_names is not None and name not in sheet_names:
                continue
            target = relations[sheet.attrib[RELATIONSHIP_ID]]
            target = (
                posixpath.normpath("xl/" + target)
                if not target.startswith("/")
                else target[1:]
            )
            if not target.startswith("xl/"):
                raise ValueError("worksheet target outside workbook")
            filename = re.sub(r"[^\w.-]+", "_", name).strip(".") or f"sheet{index}"
            csv_path = output / f"{index:02d}_{filename}.csv"
            preamble, header = [], None
            count = 0
            with csv_path.open("x", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                for number, values in _rows(book, target, strings):
                    if header is None:
                        if sum(bool(v) for v in values) < 2:
                            preamble.append({"row": number, "values": values})
                            continue
                        header = values
                        if len(set(header)) != len(header) or not all(header):
                            raise ValueError(f"ambiguous header in worksheet {name}")
                        writer.writerow(header)
                        header_row = number
                        continue
                    if len(values) > len(header):
                        if any(values[len(header) :]):
                            raise ValueError(f"data beyond header columns in {name}")
                        values = values[: len(header)]
                    writer.writerow(values + [""] * (len(header) - len(values)))
                    count += 1
            if header is None:
                raise ValueError(f"no tabular header in worksheet {name}")
            metadata.append(
                {
                    "sheet": name,
                    "csv": csv_path.name,
                    "columns": header,
                    "header_row": header_row,
                    "data_rows": count,
                    "preamble": preamble,
                    "numeric_values": "original cached XML text",
                }
            )
    return metadata
