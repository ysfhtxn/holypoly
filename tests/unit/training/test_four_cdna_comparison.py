"""Synthetic HeLa selection checks for an independent published source cohort."""

import csv
from pathlib import Path
from runpy import run_path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pytest


def write_source(path: Path, rows: list[tuple[str, str]]) -> None:
    """Create one Fig3d sheet with inline strings and no external dependencies."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    sheet = ET.Element("worksheet", xmlns=ns)
    data = ET.SubElement(sheet, "sheetData")
    header = ["Read_id", "Quality", "Gene_Name", "Biotype", "tail_length", "Species"]
    contents = [
        header,
        *[
            [f"synthetic-{index}", "0", "synthetic-gene", "unknown", length, species]
            for index, (species, length) in enumerate(rows)
        ],
    ]
    for number, values in enumerate(contents, 1):
        row = ET.SubElement(data, "row", r=str(number))
        for column, value in zip("ABCDEF", values, strict=True):
            cell = ET.SubElement(row, "c", r=f"{column}{number}", t="inlineStr")
            ET.SubElement(ET.SubElement(cell, "is"), "t").text = value
    with ZipFile(path, "w") as book:
        book.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships"><sheets>'
            '<sheet name="Fig3d" sheetId="7" r:id="source"/></sheets></workbook>',
        )
        book.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships><Relationship Id="source" '
            'Target="worksheets/measurements.xml"/></Relationships>',
        )
        book.writestr("xl/worksheets/measurements.xml", ET.tostring(sheet))


def extract_function():
    """Load the executable script without invoking its real-data command."""
    path = Path(__file__).resolve().parents[3] / "scripts/compare_four_cdna_tools.py"
    return run_path(str(path))["extract_hela"]


def test_hela_only_preserves_native_zeros_missing_and_source_rows(tmp_path):
    source, output = tmp_path / "source.xlsx", tmp_path / "hela.tsv"
    write_source(
        source,
        [
            ("Yeast", "900"),
            ("HeLa", "10.2500"),
            (" hela ", "20"),
            ("HeLa", "0"),
            ("HeLa", ""),
            ("HeLa", "-1"),
            ("Human", "800"),
        ],
    )
    before = source.read_bytes()
    values, audit = extract_function()(source, output)
    assert values == [10.25, 20]
    assert audit["hela_reported_rows"] == 5
    assert audit["hela_source_status_counts"] == {
        "positive_length": 2,
        "zero_value": 1,
        "missing_value": 1,
        "negative_value": 1,
    }
    with output.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert [r["tail_length"] for r in rows] == ["10.2500", "20", "0", "", "-1"]
    assert rows[0]["source_row"] == "3" and rows[0]["Quality"] == "0"
    assert rows[2]["positive_length_nt"] == rows[3]["positive_length_nt"] == ""
    assert source.read_bytes() == before


def test_no_hela_does_not_substitute_another_human_cohort(tmp_path):
    source = tmp_path / "source.xlsx"
    write_source(source, [("Human", "50")])
    with pytest.raises(ValueError, match="no positive HeLa.*no species substitution"):
        extract_function()(source, tmp_path / "hela.tsv")


def test_explicit_human_selection_preserves_source_label_and_records_mapping(tmp_path):
    source = tmp_path / "source.xlsx"
    write_source(source, [("Human", "50"), ("Yeast", "10")])
    output = tmp_path / "human.tsv"
    values, audit = extract_function()(source, output, source_species="Human")
    assert values == [50]
    assert audit["source_species"] == "Human"
    assert audit["display_cell_line"] == "HeLa"
    assert "User explicitly identifies" in audit["display_label_basis"]
    with output.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert rows[0]["Species"] == "Human"
