"""
tests/test_search_text_derivation_unit.py — DKGNode.search_text derivation
(docs/DESIGN_unstructured_graph_v2.md phase 3): a chunk-bounded property
Neo4j keeps for lexical matching once `text` itself is no longer written
there.

Page/Region nodes: search_text == text exactly (StructuralChunker is
already one chunk per page, so there's no aggregation to bound). Chapter/
Section nodes: search_text is derived from the same page-bucket walk as
the full aggregated `.text`, capped at _SEARCH_TEXT_CHAR_BUDGET -- NOT
simply a truncated prefix of `.text` (verified live: for a section whose
own extracted blocks are thin, page_buckets can carry more content than
what streamed into `.text`, so search_text can legitimately be longer or
differently-worded than `.text`, never assume one contains the other).

Run with:
    python -m pytest tests/test_search_text_derivation_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


def _drop_fake_stubs() -> None:
    """Drop stale fake stubs another test file left in sys.modules (e.g.
    test_scalable_pipeline_unit.py stubs src.unstructured.graph/src.unstructured.document as bare
    types.ModuleType with no __path__) — only if they're fake, never a
    genuinely-imported real module. Pytest collects every test file's
    module-level code before running any test function, so this must run
    at IMPORT time here (not just once at collection start), same pattern
    already used by test_parser_table_aware_unit.py/test_parser_rtldoc_
    unit.py's own _drop_fake_document_stubs."""
    for _mod_name in list(sys.modules):
        if _mod_name == "src.unstructured.graph" or _mod_name.startswith("src.unstructured.graph.") \
                or _mod_name == "src.unstructured.document" or _mod_name.startswith("src.unstructured.document."):
            _mod = sys.modules[_mod_name]
            if not hasattr(_mod, "__file__") and not hasattr(_mod, "__path__"):
                del sys.modules[_mod_name]


_drop_fake_stubs()

from src.unstructured.document.light.parser import LightPdfParser
from src.unstructured.graph.axis1_structural import (
    _SEARCH_TEXT_CHAR_BUDGET,
    Axis1StructuralBuilder,
    _derive_search_text,
)
from src.unstructured.graph.chunker import StructuralChunker


# ── _derive_search_text (pure, isolated) ─────────────────────────────────


def test_no_body_content_falls_back_to_title():
    assert _derive_search_text("My Title", {}, page_start=1, page_end=3) == "My Title"


def test_body_within_budget_is_title_plus_full_body():
    buckets = {1: ["short body"]}
    result = _derive_search_text("Title", buckets, page_start=1, page_end=1)
    assert result == "Title\n\nshort body"


def test_multi_page_body_joins_in_page_order():
    buckets = {1: ["page one text"], 2: ["page two text"]}
    result = _derive_search_text("Title", buckets, page_start=1, page_end=2)
    assert result == "Title\n\npage one text\n\npage two text"


def test_body_is_capped_at_char_budget():
    huge_chunk = "x" * (_SEARCH_TEXT_CHAR_BUDGET * 3)
    buckets = {1: [huge_chunk]}
    result = _derive_search_text("Title", buckets, page_start=1, page_end=1)
    body = result[len("Title\n\n"):]
    assert len(body) <= _SEARCH_TEXT_CHAR_BUDGET


def test_pages_outside_range_are_excluded():
    buckets = {1: ["in range"], 5: ["out of range, should not appear"]}
    result = _derive_search_text("Title", buckets, page_start=1, page_end=1)
    assert "out of range" not in result


def test_pages_with_no_bucket_entry_are_skipped_not_errored():
    # page 2 has no bucket at all (e.g. a blank page) -- must not KeyError.
    buckets = {1: ["a"], 3: ["c"]}
    result = _derive_search_text("Title", buckets, page_start=1, page_end=3)
    assert result == "Title\n\na\n\nc"


# ── Integration: real PDF through Axis1StructuralBuilder ────────────────


@pytest.fixture(scope="module")
def built_graph():
    parser = LightPdfParser()
    ir = parser.parse_ir("sample_data_to_test/unstructured/rag_document.pdf")
    chunks = StructuralChunker().chunk(ir)
    return Axis1StructuralBuilder().build(ir, chunks)


def _nodes_of_type(nodes, type_name):
    return [n for n in nodes if (n.type.value if hasattr(n.type, "value") else n.type) == type_name]


def test_document_search_text_matches_text_exactly(built_graph):
    nodes, _ = built_graph
    for n in _nodes_of_type(nodes, "Document"):
        assert n.search_text == n.text


def test_page_search_text_matches_text_exactly(built_graph):
    nodes, _ = built_graph
    for n in _nodes_of_type(nodes, "Page"):
        assert n.search_text == n.text


def test_region_search_text_matches_text_exactly(built_graph):
    nodes, _ = built_graph
    for n in _nodes_of_type(nodes, "Region"):
        assert n.search_text == n.text


def test_chapter_and_section_search_text_starts_with_title(built_graph):
    nodes, _ = built_graph
    for type_name in ("Chapter", "Section"):
        for n in _nodes_of_type(nodes, type_name):
            assert n.search_text
            assert n.search_text.startswith(n.title)


def test_chapter_and_section_search_text_body_respects_budget(built_graph):
    nodes, _ = built_graph
    for type_name in ("Chapter", "Section"):
        for n in _nodes_of_type(nodes, type_name):
            body = n.search_text[len(n.title):]
            assert len(body) <= _SEARCH_TEXT_CHAR_BUDGET + 4  # +4: "\n\n" prefix/joins slack
