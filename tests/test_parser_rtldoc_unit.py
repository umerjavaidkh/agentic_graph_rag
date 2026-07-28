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
    assert parser._is_heading(blocks[0], font_threshold=999.0) is True


def test_passage_role_is_not_a_heading():
    parser = RtldocPdfParser()
    blocks, _regions = parser._convert_blocks(
        [_rtl_block("passage", "Ordinary body text that just happens to be short.")],
        page_no=1,
        page=None,
    )
    assert parser._is_heading(blocks[0], font_threshold=999.0) is False


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


def test_pymupdf_sourced_block_falls_back_to_base_heading_heuristic():
    """A block from the per-page PyMuPDF fallback path (source != 'rtldoc')
    must still go through the base font-size heuristic, not rtldoc's
    role-based one (it never had a role to begin with)."""
    from src.document.light.parser import _PdfBlock

    parser = RtldocPdfParser()
    block = _PdfBlock(text="CHAPTER ONE", page=1, bold=True, max_font_size=20.0, source="pymupdf")
    # Should not raise, and should defer to LightPdfParser's own heuristic
    # (a bold, large, short, all-caps line clears its heading bar).
    assert parser._is_heading(block, font_threshold=10.0) is True


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
