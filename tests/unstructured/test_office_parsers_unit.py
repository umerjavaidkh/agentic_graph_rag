"""Word, PowerPoint and Excel as document sources.

The pipeline accepted .pdf and nothing else. A real corpus is mostly Office
files, and a PDF export of one throws away the structure these formats state
outright -- which is the same structure the PDF path has to guess at, and
currently gets wrong ("Preamble", "Smith Street", "Dry" as section titles).
"""
import pytest

from src.unstructured.document.parser_registry import get_parser, supported_extensions

pytest.importorskip("docx")
pytest.importorskip("pptx")
pytest.importorskip("openpyxl")


def _headings(ir):
    return [b.text for p in ir.pages for b in p.blocks if b.extra.get("heading_hint") == "heading"]


@pytest.fixture
def docx_path(tmp_path):
    import docx

    d = docx.Document()
    d.add_heading("Acme Security Policy", 0)
    for title in ("Access Control", "Incident Response", "Data Retention",
                  "Vendor Review", "Training"):
        d.add_heading(title, 1)
        d.add_paragraph(f"Body text under {title}.")
    table = d.add_table(rows=1, cols=2)
    table.cell(0, 0).text, table.cell(0, 1).text = "Region", "Owner"
    path = tmp_path / "policy.docx"
    d.save(path)
    return path


def test_word_headings_are_read_not_inferred(docx_path):
    """Word states its headings in the paragraph style. Nothing is guessed
    from font size, which is what produces junk titles on the PDF path."""
    ir = get_parser(docx_path).parse_ir(docx_path)
    assert _headings(ir) == [
        "Acme Security Policy", "Access Control", "Incident Response",
        "Data Retention", "Vendor Review", "Training",
    ]


def test_word_supplies_a_real_outline(docx_path):
    """Six declared headings clear the same five-entry bar the PDF path uses
    for a usable embedded outline, so Axis-1 builds from structure."""
    ir = get_parser(docx_path).parse_ir(docx_path)
    assert ir.toc and len(ir.toc) == 6


def test_word_tables_are_regions(docx_path):
    ir = get_parser(docx_path).parse_ir(docx_path)
    assert any(b.kind == "table" for p in ir.pages for b in p.regions)


def test_a_slide_is_a_page_and_its_title_is_the_heading(tmp_path):
    """No invention needed: PowerPoint already has one title per slide."""
    from pptx import Presentation

    deck = Presentation()
    for i in (1, 2, 3):
        slide = deck.slides.add_slide(deck.slide_layouts[1])
        slide.shapes.title.text = f"Slide {i} Title"
        slide.placeholders[1].text = f"Content {i}"
    path = tmp_path / "deck.pptx"
    deck.save(path)

    ir = get_parser(path).parse_ir(path)
    assert ir.page_count == 3
    assert _headings(ir) == ["Slide 1 Title", "Slide 2 Title", "Slide 3 Title"]


def test_a_sheet_is_a_page_and_its_name_is_the_heading(tmp_path):
    from openpyxl import Workbook

    wb = Workbook()
    wb.active.title = "Checklist"
    wb.active.append(["Control", "Status"])
    wb.create_sheet("Register").append(["Risk", "Level"])
    path = tmp_path / "book.xlsx"
    wb.save(path)

    ir = get_parser(path).parse_ir(path)
    assert ir.page_count == 2
    assert _headings(ir) == ["Checklist", "Register"]


def test_too_few_headings_is_not_an_outline(tmp_path):
    """Two headings are not a structure worth trusting -- same bar the PDF
    path applies before preferring an outline over its heuristics."""
    from pptx import Presentation

    deck = Presentation()
    for i in (1, 2):
        deck.slides.add_slide(deck.slide_layouts[1]).shapes.title.text = f"S{i}"
    path = tmp_path / "small.pptx"
    deck.save(path)
    assert get_parser(path).parse_ir(path).toc is None


def test_the_registry_now_dispatches_office_formats():
    assert {".pdf", ".docx", ".pptx", ".xlsx", ".xlsm"} <= supported_extensions()


def test_a_stated_heading_is_trusted_whatever_produced_it():
    """The hint used to be honoured only when block.source == 'rtldoc', so a
    parser reading headings from markup had its answer thrown away in favour
    of guessing from font size."""
    import inspect

    from src.unstructured.graph import axis1_structural

    src = inspect.getsource(axis1_structural)
    assert 'block.source == "rtldoc" and block.extra.get("heading_hint")' not in src
    assert 'block.extra.get("heading_hint") == "heading"' in src
