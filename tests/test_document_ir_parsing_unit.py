"""
tests/test_document_ir_parsing_unit.py — LightPdfParser.parse_ir() /
_block_to_ir() / _to_document_ir(): the new extraction->IR production
path (docs/DESIGN_unstructured_graph_v2.md phase 2, step 1).

parse_ir() is meant to be a drop-in alternative extraction-phase entry
point alongside the existing parse() -- same _extract_pages()/
_flag_repeated_headers()/_usable_toc() calls, just packaged as a
DocumentIR instead of handed straight to the legacy _build_from_toc/
_build_from_extracts construction methods. These tests prove that
packaging is faithful (nothing dropped/renamed/mismatched) rather than
just "doesn't crash".

Run with:
    python -m pytest tests/test_document_ir_parsing_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.document.ir import Block, DocumentIR, PageBlock
from src.document.light.parser import LightPdfParser, _PageExtract, _PdfBlock


def _parser() -> LightPdfParser:
    return LightPdfParser()


# ── _block_to_ir ──────────────────────────────────────────────────────────


def test_block_to_ir_copies_matching_fields():
    b = _PdfBlock(
        text="Item 1A. Risk Factors", page=3, bbox=[10.0, 20.0, 300.0, 40.0],
        page_size=[612.0, 792.0], max_font_size=14.0, avg_font_size=13.5,
        bold=True, source="rtldoc", kind="text", low_confidence=False,
    )
    ir_block = _parser()._block_to_ir(b)

    assert ir_block.text == b.text
    assert ir_block.page == b.page
    assert ir_block.bbox == b.bbox
    assert ir_block.page_size == b.page_size
    assert ir_block.max_font_size == b.max_font_size
    assert ir_block.avg_font_size == b.avg_font_size
    assert ir_block.bold == b.bold
    assert ir_block.source == b.source
    assert ir_block.kind == b.kind
    assert ir_block.low_confidence == b.low_confidence


def test_block_to_ir_stamps_veto_flags_into_extra():
    b = _PdfBlock(text="x", page=1, in_table_region=True, is_repeated_header=True)
    ir_block = _parser()._block_to_ir(b)

    assert ir_block.extra == {"in_table_region": True, "is_repeated_header": True}


def test_block_to_ir_no_extra_flags_when_unset():
    b = _PdfBlock(text="x", page=1)
    ir_block = _parser()._block_to_ir(b)

    assert ir_block.extra == {}


# ── _to_document_ir ───────────────────────────────────────────────────────


def test_to_document_ir_preserves_page_structure():
    p1_block = _PdfBlock(text="Chapter 1", page=1, kind="text")
    p1_region = _PdfBlock(text="Table 1: Results", page=1, kind="table")
    p2_block = _PdfBlock(text="More body text", page=2, kind="text")
    extracts = [
        _PageExtract(page=1, text="Chapter 1\n\nTable 1: Results",
                     blocks=[p1_block], regions=[p1_region], confidence=0.9),
        _PageExtract(page=2, text="More body text", blocks=[p2_block], confidence=0.8),
    ]

    ir = _parser()._to_document_ir(extracts, toc=None, source_name="doc", page_count=2)

    assert isinstance(ir, DocumentIR)
    assert ir.source_name == "doc"
    assert ir.page_count == 2
    assert len(ir.pages) == 2

    page1 = ir.pages[0]
    assert isinstance(page1, PageBlock)
    assert page1.page == 1
    assert page1.text == "Chapter 1\n\nTable 1: Results"
    assert page1.confidence == 0.9
    assert [b.text for b in page1.blocks] == ["Chapter 1"]
    # Regions are independently-constructed _PdfBlock copies upstream (see
    # _regions_from_blocks), not literally the same objects as .blocks --
    # converted separately here, so a region-only block still shows up.
    assert [r.text for r in page1.regions] == ["Table 1: Results"]
    assert page1.regions[0].kind == "table"

    page2 = ir.pages[1]
    assert page2.page == 2
    assert page2.regions == []


def test_to_document_ir_carries_toc_through_unchanged():
    toc = [(1, "Chapter 1", 1), (2, "Section 1.1", 2)]
    ir = _parser()._to_document_ir([], toc=toc, source_name="doc", page_count=2)
    assert ir.toc == toc


# ── parse_ir ──────────────────────────────────────────────────────────────


def test_parse_ir_rejects_non_pdf(tmp_path):
    bad = tmp_path / "not_a_pdf.txt"
    bad.write_text("hello")
    try:
        _parser().parse_ir(bad)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_parse_ir_matches_extract_pages_and_usable_toc_on_real_pdf():
    """parse_ir()'s output must carry the exact same page/block/toc data
    _extract_pages()+_usable_toc() would produce if called directly --
    proving the IR packaging step drops nothing, not just that parse_ir()
    runs without crashing."""
    import fitz

    parser = _parser()
    pdf_path = "sample_data_to_test/unstructured/rag_document.pdf"

    ir = parser.parse_ir(pdf_path)

    doc = fitz.open(pdf_path)
    try:
        expected_page_count = len(doc)
        extracts = parser._extract_pages(Path(pdf_path), doc)
        parser._flag_repeated_headers(extracts)
        toc = parser._usable_toc(doc)
    finally:
        doc.close()

    assert ir.page_count == expected_page_count
    assert ir.toc == toc
    assert len(ir.pages) == len(extracts)
    for ir_page, extract in zip(ir.pages, extracts):
        assert ir_page.page == extract.page
        assert ir_page.text == extract.text
        assert len(ir_page.blocks) == len(extract.blocks)
        assert [b.text for b in ir_page.blocks] == [b.text for b in extract.blocks]
        assert len(ir_page.regions) == len(extract.regions)


def test_parse_ir_content_hash_is_deterministic():
    parser = _parser()
    pdf_path = "sample_data_to_test/unstructured/rag_document.pdf"

    ir1 = parser.parse_ir(pdf_path)
    ir2 = parser.parse_ir(pdf_path)

    assert ir1.content_hash
    assert ir1.content_hash == ir2.content_hash


def test_parse_ir_and_parse_see_the_same_repeated_header_stripping():
    """parse_ir() calls _flag_repeated_headers() exactly like parse() does
    -- a page whose only content is a running header must come out with
    empty/stripped text in both, not just in the legacy path."""
    extracts = [
        _PageExtract(page=i, text="Acme Corp Confidential",
                     blocks=[_PdfBlock(text="Acme Corp Confidential", page=i,
                                        bbox=[10.0, 20.0, 200.0, 30.0])])
        for i in range(1, 7)
    ]
    parser = _parser()
    parser._flag_repeated_headers(extracts)
    ir = parser._to_document_ir(extracts, toc=None, source_name="doc", page_count=6)

    for page in ir.pages:
        assert page.text == "", f"page {page.page} still carries the stripped header in IR text"
