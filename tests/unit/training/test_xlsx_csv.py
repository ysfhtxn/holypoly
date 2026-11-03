"""Synthetic XLSX conversion checks; no real measurements or read IDs."""

import csv
from pathlib import Path
from zipfile import ZipFile

import pytest

from holypoly.evaluation.xlsx_csv import export_workbook


def workbook(path: Path, *, formula: str = "") -> None:
    """Create an OOXML fixture with sparse cells, rich text and cached numbers."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with ZipFile(path, "w") as book:
        book.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships"><sheets>'
            '<sheet name="Standards" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        book.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships><Relationship Id="rId1" '
            'Target="worksheets/sheet1.xml"/></Relationships>',
        )
        book.writestr(
            "xl/sharedStrings.xml",
            f'<sst xmlns="{ns}"><si><t>Source title</t></si>'
            "<si><r><t>R</t></r><r><t>1</t></r></si></sst>",
        )
        book.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{ns}"><sheetData>'
            '<row r="1"><c r="A1" t="s"><v>0</v></c></row>'
            '<row r="3"><c r="A3" t="inlineStr"><is><t>Group</t></is></c>'
            '<c r="B3" t="inlineStr"><is><t>Note</t></is></c>'
            '<c r="C3" t="inlineStr"><is><t>tail_length</t></is></c></row>'
            '<row r="4"><c r="A4" t="s"><v>1</v></c>'
            '<c r="C4"><v>33.590000000000003</v></c></row>'
            '<row r="6"><c r="A6" t="inlineStr"><is><t>RNA,样本</t></is></c>'
            f'<c r="C6">{formula}<v>0</v></c></row>'
            "</sheetData></worksheet>",
        )


def test_export_preserves_sparse_cells_zeros_and_original_precision(tmp_path):
    source = tmp_path / "source.xlsx"
    workbook(source)
    meta = export_workbook(source, tmp_path / "csv")
    assert meta[0]["data_rows"] == 2
    assert meta[0]["header_row"] == 3
    assert meta[0]["preamble"] == [{"row": 1, "values": ["Source title"]}]
    with (tmp_path / "csv" / meta[0]["csv"]).open() as stream:
        rows = list(csv.DictReader(stream))
    assert rows == [
        {"Group": "R1", "Note": "", "tail_length": "33.590000000000003"},
        {"Group": "RNA,样本", "Note": "", "tail_length": "0"},
    ]
    with pytest.raises(FileExistsError):
        export_workbook(source, tmp_path / "csv")


def test_uncached_formula_is_rejected(tmp_path):
    source = tmp_path / "source.xlsx"
    workbook(source)
    with ZipFile(source) as book:
        files = {name: book.read(name) for name in book.namelist()}
    key = "xl/worksheets/sheet1.xml"
    files[key] = files[key].replace(
        b"<v>0</v></c></row></sheetData>", b"<f>1+1</f></c></row></sheetData>"
    )
    with ZipFile(source, "w") as book:
        for name, value in files.items():
            book.writestr(name, value)
    with pytest.raises(ValueError, match="no cached value"):
        export_workbook(source, tmp_path / "csv")


def test_explicit_sheet_selection_and_unknown_name(tmp_path):
    source = tmp_path / "source.xlsx"
    workbook(source)
    with ZipFile(source) as book:
        files = {name: book.read(name) for name in book.namelist()}
    files["xl/workbook.xml"] = files["xl/workbook.xml"].replace(
        b"</sheets>", b'<sheet name="Unselected" sheetId="2" r:id="rId2"/></sheets>'
    )
    files["xl/_rels/workbook.xml.rels"] = files["xl/_rels/workbook.xml.rels"].replace(
        b"</Relationships>",
        b'<Relationship Id="rId2" Target="worksheets/absent.xml"/></Relationships>',
    )
    with ZipFile(source, "w") as book:
        for name, value in files.items():
            book.writestr(name, value)
    meta = export_workbook(source, tmp_path / "selected", sheet_names=["Standards"])
    assert [s["sheet"] for s in meta] == ["Standards"]
    assert len(list((tmp_path / "selected").glob("*.csv"))) == 1
    with pytest.raises(ValueError, match="absent"):
        export_workbook(source, tmp_path / "bad", sheet_names=["Unknown"])
    assert not (tmp_path / "bad").exists()
