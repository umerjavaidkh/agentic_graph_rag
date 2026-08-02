"""
tests/test_parser_rtldoc_unit.py — RtldocPdfParser (the default ".pdf"/
".pdf:rtldoc" backend as of its introduction).

Covers: registry wiring, the per-page fallback to plain PyMuPDF extraction
when rtldoc declines a page (born_digital=False or empty blocks) or isn't
importable/errors entirely, role-based heading detection (trusts rtldoc's
own classification instead of re-deriving from font size), and a real-PDF
smoke test against the committed sample document.

Run with:
    python -m pytest tests/test_parser_rtldoc_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Same fresh-registry-fixture rationale as test_document_parser_registry_unit.py
# and test_parser_table_aware_unit.py: other test modules stub src.document*
# with bare fakes at import time, which pytest collection can leave in
# sys.modules before this file's own tests run.
def _drop_fake_document_stubs() -> None:
    for name in list(sys.modules):
        if name == "src.document" or name.startswith("src.document."):
            mod = sys.modules[name]
            if getattr(mod, "__file__", None) is None and getattr(mod, "__path__", None) is None:
                del sys.modules[name]


from src.document.rtldoc_backend.parser import RtldocPdfParser
from src.graph.axis1_structural import Axis1StructuralBuilder


def _is_heading(pdf_block, font_threshold: float, *, parser: RtldocPdfParser | None = None) -> bool:
    """Heading detection now happens in two steps -- a backend's
    _block_to_ir stamps hints (role classification, veto flags) onto the
    IR Block, and Axis1StructuralBuilder._is_heading reads them -- rather
    than one per-class _is_heading override. This helper chains both
    steps so the tests below can still express "is this _PdfBlock a
    heading?" in one call."""
    ir_block = (parser or RtldocPdfParser())._block_to_ir(pdf_block)
    return Axis1StructuralBuilder()._is_heading(ir_block, font_threshold)


# ── registry wiring ───────────────────────────────────────────────────────


def test_registered_as_default_pdf_backend():
    _drop_fake_document_stubs()
    from src.document.parser_registry import get_parser

    parser = get_parser("doc.pdf")
    assert type(parser).__name__ == "RtldocPdfParser"
    assert type(parser).__module__ == "src.document.rtldoc_backend.parser"


def test_registered_under_rtldoc_backend_qualifier():
    _drop_fake_document_stubs()
    from src.document.parser_registry import get_parser

    parser = get_parser("doc.pdf", backend="rtldoc")
    assert type(parser).__name__ == "RtldocPdfParser"


def test_light_and_table_aware_backends_still_available():
    """The whole point of registering rtldoc as a new backend rather than
    replacing the others in place: explicit opt-out must still work."""
    _drop_fake_document_stubs()
    from src.document.parser_registry import get_parser

    assert type(get_parser("doc.pdf", backend="light")).__name__ == "LightPdfParser"
    assert type(get_parser("doc.pdf", backend="table-aware")).__name__ == "TableAwarePdfParser"


# ── heading detection trusts rtldoc's own role classification ────────────


def _rtl_block(role: str, text: str, bbox=(0.0, 0.0, 10.0, 10.0)):
    return SimpleNamespace(role=role, text=text, bbox=bbox)


def test_heading_role_becomes_a_heading_regardless_of_font_size():
    parser = RtldocPdfParser()
    blocks, _regions = parser._convert_blocks(
        [_rtl_block("heading", "5.1 Forces")], page_no=1, page=None
    )
    assert len(blocks) == 1
    assert _is_heading(blocks[0], font_threshold=999.0, parser=parser) is True


def test_passage_role_is_not_a_heading():
    parser = RtldocPdfParser()
    blocks, _regions = parser._convert_blocks(
        [_rtl_block("passage", "Ordinary body text that just happens to be short.")],
        page_no=1,
        page=None,
    )
    assert _is_heading(blocks[0], font_threshold=999.0, parser=parser) is False


def test_table_and_figure_roles_become_regions():
    parser = RtldocPdfParser()
    blocks, regions = parser._convert_blocks(
        [
            _rtl_block("table", "| a | b |\n|---|---|\n| 1 | 2 |"),
            _rtl_block("figure", "A chart"),
            _rtl_block("passage", "Body text"),
        ],
        page_no=1,
        page=None,
    )
    assert len(blocks) == 3
    assert len(regions) == 2
    assert {r.kind for r in regions} == {"table", "figure"}


def test_blank_text_blocks_are_dropped():
    parser = RtldocPdfParser()
    blocks, _regions = parser._convert_blocks(
        [_rtl_block("passage", "   \n  ")], page_no=1, page=None
    )
    assert blocks == []


# ── geometric rescue when rtldoc's own role classification misses ────────
#
# Regression: verified live on a synthetic test PDF (sample-10pages.pdf)
# where "Section 1: Introduction & Overview" -- 11pt bold vs. a 9pt
# non-bold body baseline, an unambiguous heading by font size/weight
# alone -- was classified role="passage" by rtldoc, collapsing the whole
# 10-page document into one undifferentiated "Preamble" section. Fixed by
# parsing rtldoc's own Block.style ("{font}|{size}|{hex}|{flags}") into
# max_font_size/bold on the _PdfBlock, then falling back to the same
# font-size heuristic used for PyMuPDF-sourced blocks when rtldoc's role
# says "not a heading" but the block's own geometry strongly disagrees.


def _rtl_block_with_style(role: str, text: str, style: str, bbox=(0.0, 0.0, 10.0, 10.0)):
    return SimpleNamespace(role=role, text=text, bbox=bbox, style=style)


def test_parse_style_extracts_size_and_bold_flag():
    size, bold = RtldocPdfParser._parse_style("Helvetica-Bold|11.0|000000|B")
    assert size == 11.0
    assert bold is True


def test_parse_style_non_bold_flag_empty_string():
    size, bold = RtldocPdfParser._parse_style("Helvetica|9.0|262626|")
    assert size == 9.0
    assert bold is False


def test_parse_style_handles_none_and_malformed_input():
    assert RtldocPdfParser._parse_style(None) == (0.0, False)
    assert RtldocPdfParser._parse_style("") == (0.0, False)
    assert RtldocPdfParser._parse_style("not|enough|parts") == (0.0, False)
    assert RtldocPdfParser._parse_style("Helvetica|not-a-number|000000|B") == (0.0, True)


def test_convert_blocks_populates_font_metrics_from_style():
    parser = RtldocPdfParser()
    blocks, _regions = parser._convert_blocks(
        [_rtl_block_with_style("passage", "Section 1: Introduction", "Helvetica-Bold|11.0|000000|B")],
        page_no=1,
        page=None,
    )
    assert blocks[0].max_font_size == 11.0
    assert blocks[0].bold is True


def test_missing_style_attribute_does_not_crash():
    """SimpleNamespace mocks without a .style attr (as used by the older
    role-based tests above) must still convert cleanly -- getattr default,
    not an AttributeError."""
    parser = RtldocPdfParser()
    blocks, _regions = parser._convert_blocks(
        [_rtl_block("passage", "some text")], page_no=1, page=None
    )
    assert blocks[0].max_font_size == 0.0
    assert blocks[0].bold is False


def test_rtldoc_missed_heading_rescued_by_font_geometry():
    """The exact live regression: rtldoc role="passage" but the block is
    bold and clears the document's own font-size threshold -- must be
    rescued as a heading, not silently folded into body text."""
    parser = RtldocPdfParser()
    blocks, _regions = parser._convert_blocks(
        [_rtl_block_with_style(
            "passage", "Section 1: Introduction & Overview", "Helvetica-Bold|11.0|000000|B"
        )],
        page_no=1,
        page=None,
    )
    # font_threshold=10.5 mirrors _heading_font_threshold's median(9.0)+1.5
    # for this document's actual body-text size.
    assert _is_heading(blocks[0], font_threshold=10.5, parser=parser) is True


def test_rtldoc_passage_role_with_body_sized_font_stays_not_a_heading():
    """The rescue must not become a new false-positive source: an actual
    body-text block (small, non-bold) that rtldoc correctly called
    "passage" must still score as not-a-heading."""
    parser = RtldocPdfParser()
    blocks, _regions = parser._convert_blocks(
        [_rtl_block_with_style(
            "passage",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit sed do eiusmod tempor.",
            "Helvetica|9.0|262626|",
        )],
        page_no=1,
        page=None,
    )
    assert _is_heading(blocks[0], font_threshold=10.5, parser=parser) is False


def test_pymupdf_sourced_block_falls_back_to_base_heading_heuristic():
    """A block from the per-page PyMuPDF fallback path (source != 'rtldoc')
    must still go through the base font-size heuristic, not rtldoc's
    role-based one (it never had a role to begin with)."""
    from src.document.light.parser import _PdfBlock

    parser = RtldocPdfParser()
    block = _PdfBlock(text="CHAPTER ONE", page=1, bold=True, max_font_size=20.0, source="pymupdf")
    # Should not raise, and should defer to LightPdfParser's own heuristic
    # (a bold, large, short, all-caps line clears its heading bar).
    assert _is_heading(block, font_threshold=10.0, parser=parser) is True


# ── per-page fallback when rtldoc declines a page ────────────────────────


def test_falls_back_to_base_parser_entirely_when_rtldoc_not_installed():
    # Fresh, un-stubbed import of the actual class under test -- patching
    # LightPdfParser._extract_pages by string path below re-resolves
    # src.document.light.parser from current sys.modules state, which must
    # be the SAME module object this parser's class hierarchy points to, or
    # the patch silently lands on a different (stub-polluted) copy of the
    # class and this test would exercise the real method instead of the
    # mock -- same cross-file sys.modules pollution risk documented in
    # test_parser_table_aware_unit.py.
    _drop_fake_document_stubs()
    from src.document.light.parser import LightPdfParser as _FreshLightPdfParser
    from src.document.rtldoc_backend.parser import RtldocPdfParser as _FreshRtldocPdfParser

    parser = _FreshRtldocPdfParser()
    with patch.dict(sys.modules, {"rtldoc": None, "rtldoc.pipeline": None}):
        with patch.object(
            _FreshLightPdfParser, "_extract_pages", return_value=["sentinel"]
        ) as base_extract:
            result = parser._extract_pages(Path("doc.pdf"), doc=SimpleNamespace(page_count=0))
    assert result == ["sentinel"]
    base_extract.assert_called_once()


def test_falls_back_to_base_parser_entirely_when_rtldoc_raises():
    _drop_fake_document_stubs()
    from src.document.light.parser import LightPdfParser as _FreshLightPdfParser
    from src.document.rtldoc_backend.parser import RtldocPdfParser as _FreshRtldocPdfParser

    parser = _FreshRtldocPdfParser()
    with patch(
        "rtldoc.pipeline.parse_document", side_effect=RuntimeError("boom")
    ):
        with patch.object(
            _FreshLightPdfParser, "_extract_pages", return_value=["sentinel"]
        ) as base_extract:
            result = parser._extract_pages(Path("doc.pdf"), doc=SimpleNamespace(page_count=0))
    assert result == ["sentinel"]
    base_extract.assert_called_once()


# ── real-PDF smoke test ───────────────────────────────────────────────────


def test_parses_real_sample_pdf_without_error():
    """Same shape as TableAwarePdfParser's equivalent test — not asserting
    exact structure (content-specific), just that the whole pipeline runs
    end-to-end against a real, small, committed sample PDF without
    crashing, and produces the expected node types."""
    parser = RtldocPdfParser()
    nodes, edges = parser.parse("sample_data_to_test/unstructured/rag_document.pdf")
    assert nodes
    assert edges
    types = {(n.type.value if hasattr(n.type, "value") else n.type) for n in nodes}
    assert "Page" in types
    assert "Document" in types
