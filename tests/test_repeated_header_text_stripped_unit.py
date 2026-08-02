"""
tests/test_repeated_header_text_stripped_unit.py — repeated running
headers/footers are actually excluded from page/section body text, not
just prevented from being misclassified as a heading.

Regression: _flag_repeated_headers (originally table_aware/parser.py-only)
set block.is_repeated_header AFTER each page's .text had already been
joined during _extract_pages -- the flag only ever vetoed heading
classification (TableAwarePdfParser._is_heading), so the repeated text
(a company name/document title printed on every page) still flowed into
every PAGE/Section node's text, still got embedded, and still got sent
through NER. Also: the mechanism lived only in TableAwarePdfParser, so
RtldocPdfParser (the default backend) never benefited from it at all --
verified RtldocPdfParser doesn't call super()._extract_pages() on its
main (successful) path, so a fix placed inside LightPdfParser._extract_pages
itself wouldn't reach it either; it has to run from parse(), the one
method no backend overrides.

Fixed by: moving the detection to LightPdfParser, calling it once from
parse() (after _extract_pages returns, regardless of which backend
produced the extracts), recomputing each affected page's .text to
exclude flagged blocks, and skipping flagged blocks entirely in
_build_from_extracts's section-body-text accumulation loop.

Run with:
    python -m pytest tests/test_repeated_header_text_stripped_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.document.light.parser import LightPdfParser, _PageExtract, _PdfBlock
from src.models import NodeType


def _extract(page: int, texts_and_y: list[tuple[str, float]]) -> _PageExtract:
    blocks = [
        _PdfBlock(text=text, page=page, bbox=[10.0, y0, 200.0, y0 + 10.0])
        for text, y0 in texts_and_y
    ]
    return _PageExtract(page=page, text="\n".join(t for t, _ in texts_and_y), blocks=blocks)


def test_flagged_header_text_is_removed_from_extract_text():
    extracts = [
        _extract(i, [("Acme Corp Confidential", 20.0), (f"Real body content {i}", 100.0)])
        for i in range(1, 7)
    ]
    parser = LightPdfParser()
    parser._flag_repeated_headers(extracts)

    for extract in extracts:
        assert "Acme Corp Confidential" not in extract.text, f"page {extract.page} still has the header in .text"
        assert f"Real body content {extract.page}" in extract.text


def test_unflagged_pages_keep_original_text_untouched():
    """Pages with no matching repeated block must not have .text rebuilt
    at all -- confirms the fix is scoped to only the pages it touches."""
    extracts = [
        _extract(i, [("Acme Corp Confidential", 20.0), (f"Real body content {i}", 100.0)])
        for i in range(1, 6)
    ] + [_extract(6, [("Unrelated one-off page", 50.0)])]
    parser = LightPdfParser()
    original_text_page6 = extracts[-1].text
    parser._flag_repeated_headers(extracts)

    assert extracts[-1].text == original_text_page6


def test_section_body_text_excludes_repeated_header_via_full_pipeline():
    """End-to-end through _build_from_extracts: a repeated header must
    never appear inside any Section node's text, even though it's a
    plain, non-heading-looking block that would otherwise just flow into
    the current section's accumulated body text."""
    blocks_per_page = []
    for i in range(1, 7):
        blocks_per_page.append([
            _PdfBlock(text="Acme Corp Confidential", page=i, bbox=[10.0, 20.0, 200.0, 30.0]),
            _PdfBlock(
                text="Chapter One", page=i, bbox=[10.0, 40.0, 200.0, 60.0],
                avg_font_size=20.0, max_font_size=20.0,
            ) if i == 1 else None,
            _PdfBlock(text=f"Body paragraph on page {i}.", page=i, bbox=[10.0, 100.0, 200.0, 120.0],
                      avg_font_size=10.0, max_font_size=10.0),
        ])
    extracts = [
        _PageExtract(page=i, text="", blocks=[b for b in page_blocks if b is not None])
        for i, page_blocks in enumerate(blocks_per_page, start=1)
    ]

    parser = LightPdfParser()
    parser._flag_repeated_headers(extracts)
    nodes, _edges = parser._build_from_extracts(extracts, "sample-doc", page_count=6)

    sections = [n for n in nodes if n.type == NodeType.SECTION]
    assert sections, "expected at least one section to be built"
    for section in sections:
        assert "Acme Corp Confidential" not in section.text
