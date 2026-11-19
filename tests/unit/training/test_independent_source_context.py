"""Synthetic audits for independent cohorts and lossless result organization."""

import csv
from collections import Counter
from pathlib import Path
from runpy import run_path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import pytest


@pytest.fixture
def workflow(monkeypatch):
    """Import the CLI helpers without reading external sources or running main."""
    scripts = Path(__file__).resolve().parents[3] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    return run_path(str(scripts / "refresh_tool_comparison.py"))


def test_tailfindr_replication_and_native_failure_values(tmp_path, workflow):
    source = tmp_path / "tails.csv"
    source.write_text(
        "tail_length,replicate\n12,1\n0,1\nNA,1\nnan,1\n-1,1\ninvalid,1\n24,2\n"
    )
    groups = workflow["read_tailfindr"](source)
    assert groups["rep1"]["values"] == [12]
    assert groups["rep2"]["values"] == [24]
    assert groups["rep1"]["status_counts"] == {
        "positive_length": 1,
        "zero_value": 1,
        "missing_value": 1,
        "nonfinite_value": 1,
        "negative_value": 1,
        "invalid_value": 1,
    }


def test_dna_types_validity_and_native_zero_are_separate(tmp_path, workflow):
    source = tmp_path / "dna.csv"
    source.write_text(
        "replicate,read_type,tail_is_valid,tail_length\n"
        "1,polyA,TRUE,12\n1,polyA,TRUE,0\n1,polyA,FALSE,99\n"
        "1,polyA,TRUE,NA\n1,polyT,TRUE,24\n1,invalid,FALSE,\n"
        "2,polyT,TRUE,30\n"
    )
    groups, audit = workflow["read_tailfindr_dna"](source)
    assert groups["DNA polyA rep1"]["values"] == [12]
    assert groups["DNA polyT rep1"]["values"] == [24]
    assert groups["DNA polyT rep2"]["values"] == [30]
    assert groups["DNA polyA rep1"]["status_counts"] == {
        "positive_length": 1,
        "zero_value": 1,
        "native_invalid": 1,
        "missing_value": 1,
    }
    assert audit["reported_rows"] == 7
    assert not audit["nominal_length_assigned"]
    assert groups["DNA polyT rep1"]["poly_type"] == "polyT"


def write_barcodes(path, entries):
    """Create a synthetic native decoder table including its quality flags."""
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "replicate",
                "read_id",
                "read_type",
                "barcode",
                "barcode_passed_threshold",
                "barcode_tie",
                "read_too_long",
                "read_too_short",
            ]
        )
        writer.writerows(entries)


def test_barcode_join_uses_identity_and_quality_not_estimated_length(
    tmp_path, workflow
):
    lengths, barcodes = tmp_path / "length.csv", tmp_path / "barcodes.csv"
    lengths.write_text(
        "replicate,read_id,read_type,tail_is_valid,tail_length\n"
        "1,shared,polyA,TRUE,999\n2,shared,polyT,TRUE,6\n"
        "1,fail,polyA,TRUE,10\n1,tie,polyA,TRUE,30\n"
        "1,unmatched,polyA,TRUE,40\n1,disagree,polyT,TRUE,60\n"
        "1,missing,polyA,TRUE,NA\n1,zero,polyA,TRUE,0\n"
        "1,invalid,polyA,FALSE,150\n1,other,polyA,TRUE,1\n"
    )
    write_barcodes(
        barcodes,
        [
            ["2", "shared", "polyT", "150", "TRUE", "FALSE", "TRUE", "FALSE"],
            ["1", "shared", "polyA", "10", "TRUE", "FALSE", "FALSE", "FALSE"],
            ["1", "fail", "polyA", "30", "FALSE", "FALSE", "FALSE", "FALSE"],
            ["1", "tie", "polyA", "30", "TRUE", "TRUE", "FALSE", "FALSE"],
            ["2", "unmatched", "polyA", "40", "TRUE", "FALSE", "FALSE", "FALSE"],
            ["1", "disagree", "polyA", "60", "TRUE", "FALSE", "FALSE", "FALSE"],
            ["1", "missing", "polyA", "10", "TRUE", "FALSE", "FALSE", "FALSE"],
            ["1", "zero", "polyA", "10", "TRUE", "FALSE", "FALSE", "FALSE"],
            ["1", "invalid", "polyA", "10", "TRUE", "FALSE", "FALSE", "FALSE"],
            ["1", "other", "polyA", "999", "TRUE", "FALSE", "FALSE", "FALSE"],
        ],
    )
    joined = tmp_path / "joined.tsv"
    groups, audit = workflow["read_barcoded_dna"](
        lengths, barcodes, joined_output=joined
    )
    assert groups["DNA 10nt polyA rep1"]["values"] == [999]
    assert groups["DNA 150nt polyT rep2"]["values"] == [6]
    assert audit["length_only_keys"] == audit["barcode_only_keys"] == 1
    assert audit["length_row_selection_counts"] == {
        "positive_length": 2,
        "threshold_not_passed": 1,
        "tied_or_missing_tie_flag": 1,
        "no_barcode_row": 1,
        "direction_disagreement": 1,
        "missing_value": 1,
        "zero_value": 1,
        "native_invalid": 1,
        "outside_six_nominal_groups": 1,
    }
    assert groups["DNA 10nt polyA rep1"]["status_counts"]["zero_value"] == 1
    with joined.open() as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    assert len(rows) == 6
    assert rows[0]["nominal_length_nt"] == "10"
    assert audit["length_basecalling"] != audit["barcode_basecalling"]


@pytest.mark.parametrize("duplicate_table", ["barcode", "length"])
def test_barcode_join_rejects_duplicate_identity(tmp_path, workflow, duplicate_table):
    lengths, barcodes = tmp_path / "length.csv", tmp_path / "barcodes.csv"
    length_row = "1,same,polyA,TRUE,12\n"
    lengths.write_text(
        "replicate,read_id,read_type,tail_is_valid,tail_length\n"
        + length_row
        + (length_row if duplicate_table == "length" else "")
    )
    barcode_row = ["1", "same", "polyA", "10", "TRUE", "FALSE", "FALSE", "FALSE"]
    write_barcodes(barcodes, [barcode_row] * (2 if duplicate_table == "barcode" else 1))
    with pytest.raises(ValueError, match="duplicate"):
        workflow["read_barcoded_dna"](lengths, barcodes)


def test_nominal_dna_numeric_positions_and_separate_replication(tmp_path, workflow):
    local, published = tmp_path / "local.tsv", tmp_path / "published.csv"
    local.write_text(
        "tool\tnominal_length_nt\tpositive_length_nt\n"
        "Holypoly\t20\t18\nDorado\t20\t23\n"
    )
    published.write_text("Group,Std_Type,tail_length\ncDNA_15,cDNA,14\nR1,RNA,25\n")
    groups = {}
    for nt in (10, 30, 40, 60, 100, 150):
        for rep in ("1", "2"):
            groups[f"{nt}/{rep}"] = {
                "nominal_length_nt": nt,
                "replicate": rep,
                "poly_type": "polyA",
                "values": [nt + int(rep), 500],
                "molecule": "DNA",
                "status_counts": Counter(positive_length=2),
                "cohort": "tailfindr_DNA_LSK108_109",
                "label_source": "decoded barcode",
            }
    pooled = workflow["pool_dna_replicates"](groups)
    records = workflow["plot_context"](
        local,
        published,
        tmp_path,
        combined=True,
        focus_max_nt=220,
        local_tools=("Holypoly", "Dorado"),
        tailfindr_groups=pooled,
    )
    dna = [r for r in records if r["cohort"] == "tailfindr_DNA_LSK108_109"]
    assert [r["nominal_length_nt"] for r in dna] == [10, 30, 40, 60, 100, 150]
    assert [r["plot_x_nt"] for r in dna] == [11, 31, 41, 61, 101, 151]
    assert all(r["positive_rows"] == 4 for r in dna)
    assert dna[0]["mean_nt"] == (11 + 500 + 12 + 500) / 4
    assert next(r for r in records if r["group"] == "R1")["nominal_length_nt"] is None
    svg = (tmp_path / "fixed_len_with_nano3p_tailfindr_combined.svg").read_text()
    assert "replicate mixtures" in svg
    assert "PolyAnalysis" not in svg
    workflow["render_dna_replicates"](groups, tmp_path)
    assert (tmp_path / "replicate_group_summary.tsv").exists()


def write_raw_book(path, *, count="4", values="[10, 0, -1, null]"):
    """Create raw list fields plus intentionally misleading transcript medians."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    header = [
        "mRNA",
        "polyA_length_293T_total_rep1",
        "polyA_reads_count_293T_total_rep1",
        "polyA_length_293T_total_rep2",
        "polyA_reads_count_293T_total_rep2",
        "polyA_median_293T_total_rep1",
    ]
    sheet = ET.Element("worksheet", xmlns=ns)
    data = ET.SubElement(sheet, "sheetData")
    for number, entries in enumerate(
        [header, ["synthetic-transcript", values, count, "[20, 30]", "2", "9999"]], 1
    ):
        row = ET.SubElement(data, "row", r=str(number))
        for column, text in zip("ABCDEF", entries, strict=True):
            cell = ET.SubElement(row, "c", r=f"{column}{number}", t="inlineStr")
            ET.SubElement(ET.SubElement(cell, "is"), "t").text = text
    with ZipFile(path, "w") as book:
        book.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/'
            'officeDocument/2006/relationships"><sheets>'
            '<sheet name="raw" sheetId="1" r:id="raw"/></sheets></workbook>',
        )
        book.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships><Relationship Id="raw" '
            'Target="worksheets/raw.xml"/></Relationships>',
        )
        book.writestr("xl/worksheets/raw.xml", ET.tostring(sheet))


def test_shuyi_uses_lengths_instead_of_transcript_medians(tmp_path, workflow):
    source = tmp_path / "source.xlsx"
    write_raw_book(source)
    groups, audit = workflow["read_shuyi"](source)
    assert groups["rep1"]["values"] == [10]
    assert groups["rep2"]["values"] == [20, 30]
    assert audit["count_mismatches"] == []
    assert groups["rep1"]["status_counts"]["zero_value"] == 1
    assert groups["rep1"]["status_counts"]["missing_value"] == 1
    assert "lack read IDs" in audit["identity_limitation"]


def test_shuyi_count_mismatch_is_explicit(tmp_path, workflow):
    source = tmp_path / "source.xlsx"
    write_raw_book(source, count="100")
    _, audit = workflow["read_shuyi"](source)
    assert audit["count_mismatches"] == [
        {"source_row": 2, "replicate": 1, "reported_count": "100", "list_entries": 4}
    ]


def test_shuyi_truncated_list_is_not_silently_used(tmp_path, workflow):
    source = tmp_path / "source.xlsx"
    write_raw_book(source, values="[10, 20,")
    with pytest.raises(ValueError, match="malformed Shuyi list"):
        workflow["read_shuyi"](source)


@pytest.mark.parametrize("last_token", ["123", "12 "])
def test_excel_truncation_keeps_only_comma_terminated_numbers(
    tmp_path, workflow, last_token
):
    source = tmp_path / "source.xlsx"
    prefix = "[" + "10," * 10921
    text = prefix + last_token
    assert len(text) == 32767
    write_raw_book(source, count="12000", values=text)
    groups, audit = workflow["read_shuyi"](source, truncated_policy="prefix")
    assert len(groups["rep1"]["values"]) == 10921
    assert set(groups["rep1"]["values"]) == {10}
    assert audit["truncated_cells"][0]["retained_entries"] == 10921
    assert audit["count_mismatches"] == []
    assert audit["reported_length_count"] == 12002
    assert audit["available_length_entries"] == 10923


def test_count_zero_scalar_is_empty_list_not_zero_nt(tmp_path, workflow):
    source = tmp_path / "source.xlsx"
    write_raw_book(source, count="0", values="0")
    # Keep one positive row in the same replicate so the source remains usable.
    with ZipFile(source) as book:
        contents = {name: book.read(name) for name in book.namelist()}
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    sheet = ET.fromstring(contents["xl/worksheets/raw.xml"])
    data = sheet.find(f"{{{ns}}}sheetData")
    row = ET.SubElement(data, f"{{{ns}}}row", r="3")
    for column, text in zip(
        "ABCDEF", ["synthetic-2", "[5]", "1", "[7]", "1", "9"], strict=True
    ):
        cell = ET.SubElement(row, f"{{{ns}}}c", r=f"{column}3", t="inlineStr")
        ET.SubElement(ET.SubElement(cell, f"{{{ns}}}is"), f"{{{ns}}}t").text = text
    contents["xl/worksheets/raw.xml"] = ET.tostring(sheet)
    with ZipFile(source, "w") as book:
        for name, content in contents.items():
            book.writestr(name, content)
    groups, audit = workflow["read_shuyi"](source)
    assert groups["rep1"]["values"] == [5]
    assert groups["rep1"]["status_counts"] == {
        "positive_length": 1,
        "empty_list_zero_sentinel": 1,
    }
    assert audit["count_mismatches"] == []


def test_reorganization_keeps_original_bytes_without_links(tmp_path, workflow):
    source = tmp_path / "tailfindr_paper_data"
    source.mkdir()
    (source / "synthetic.csv").write_text("tail_length,replicate\n10,1\n")
    moves = workflow["plan_layout"](tmp_path)
    audit = workflow["organize"](tmp_path, moves, "synthetic")
    assert not source.exists()
    assert not source.is_symlink()
    assert audit["all_original_file_bytes_verified_unchanged"]
    destination = tmp_path / "sources/tailfindr_paper_data/synthetic.csv"
    assert destination.read_text().endswith("10,1\n")
    assert not audit["old_paths_retained_as_relative_symlinks"]
    assert workflow["relocated_path"](source / "synthetic.csv", moves) == destination
    assert workflow["find_comparison"](tmp_path, "tailfindr_paper_data", "sources") == (
        destination.parent
    )


def test_added_tailfindr_does_not_remap_cdna_or_recompute_focus(tmp_path, workflow):
    local, published = tmp_path / "local.tsv", tmp_path / "published.csv"
    local.write_text(
        "tool\tnominal_length_nt\tpositive_length_nt\n"
        "Holypoly\t20\t18\nDorado\t20\t23\nPolyAnalysis\t20\t17\n"
    )
    published.write_text(
        "Group,Std_Type,tail_length\ncDNA_0,cDNA,0\ncDNA_15,cDNA,14\nR1,RNA,25\n"
    )
    plot = workflow["plot_context"]
    baseline, extended = tmp_path / "baseline", tmp_path / "extended"
    baseline.mkdir()
    extended.mkdir()
    before = plot(local, published, baseline, combined=True)
    after = plot(
        local,
        published,
        extended,
        combined=True,
        focus_max_nt=220,
        tailfindr_groups={
            "rep1": {
                "values": [50, 500],
                "status_counts": Counter(positive_length=2, zero_value=1),
            }
        },
    )
    assert after[:-1] == before
    assert after[-1]["mean_nt"] == 275
    assert after[-1]["reported_rows"] == 3
    assert after[-1]["zero_rows"] == 1
    svg = (extended / "fixed_len_with_nano3p_tailfindr_combined.svg").read_text()
    assert "different RNA dataset" in svg
    assert "no assigned nt" in svg
    assert 'id="axes_2"' not in svg


def test_remove_polyanalysis_and_add_independent_dna(tmp_path, workflow):
    local, published = tmp_path / "local.tsv", tmp_path / "published.csv"
    local.write_text(
        "tool\tnominal_length_nt\tpositive_length_nt\n"
        "Holypoly\t20\t18\nDorado\t20\t23\nPolyAnalysis\t20\tinvalid\n"
    )
    published.write_text("Group,Std_Type,tail_length\ncDNA_15,cDNA,14\nR1,RNA,25\n")
    records = workflow["plot_context"](
        local,
        published,
        tmp_path,
        combined=True,
        focus_max_nt=220,
        local_tools=("Holypoly", "Dorado"),
        tailfindr_groups={
            "RNA rep1": {"values": [50], "status_counts": Counter(positive_length=1)},
            "DNA polyT rep1": {
                "values": [60, 600],
                "status_counts": Counter(positive_length=2),
                "molecule": "DNA",
                "cohort": "tailfindr_DNA_LSK108_109",
                "display_label": "DNA T\nrep1",
            },
        },
    )
    assert not any(row["tool"] == "PolyAnalysis" for row in records)
    dna = next(row for row in records if row["cohort"] == "tailfindr_DNA_LSK108_109")
    assert dna["mean_nt"] == 330
    for name in (
        "fixed_len_with_nano3p_tailfindr_combined.svg",
        "fixed_len_with_nano3p_tailfindr_combined_focus_0_220nt.svg",
    ):
        svg = (tmp_path / name).read_text()
        assert "PolyAnalysis" not in svg
        assert "DNA T" in svg
        assert "different dataset" in svg


def test_dna_context_removes_proxy_and_keeps_all_histogram_values(tmp_path, workflow):
    series = {
        "Holypoly": [10, 20],
        "Dorado": [12, 22],
        "Nano3P-seq / HeLa*": [30, 40],
        "Shuyi / 293T-R9 (rep1+rep2)": [50, 500],
        "tailfindr DNA / polyA (rep1+rep2)": [60, 70],
        "tailfindr DNA / polyT (rep1+rep2)": [80, 90],
    }
    records = workflow["render_shuyi_context"](
        series,
        tmp_path,
        0,
        {"available_length_entries": 2, "reported_length_count": 2},
    )
    assert len(records) == 6
    assert next(r for r in records if r["series"].startswith("Shuyi"))["mean_nt"] == 275
    svg = (
        tmp_path / "length_distributions_with_hela_shuyi_tailfindr_dna.svg"
    ).read_text()
    assert "PolyAnalysis" not in svg
    assert "proxy" not in svg
    assert "polyT" in svg
