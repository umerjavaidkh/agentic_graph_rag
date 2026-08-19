"""
tests/test_parser_toc_based_structure_unit.py — embedded-PDF-outline-based
chapter/section structure building (LightPdfParser._usable_toc /
Axis1StructuralBuilder._build_from_toc).

_build_from_toc moved from LightPdfParser to Axis1StructuralBuilder
(docs/DESIGN_unstructured_graph_v2.md phase 2) -- these tests now build a
DocumentIR from the same _PageExtract fixtures via LightPdfParser's own
_to_document_ir (unchanged) and call the new builder, rather than calling
a since-removed method on LightPdfParser directly.

Built in response to font-size/regex heading heuristics misclassifying
equation fragments, exercise prompts, and repeated running headers as new
chapter headings (1,002 spurious Chapter nodes on a real physics textbook
that should have ~17). A PDF's own embedded outline/bookmarks, when
present, are ground truth and sidestep the heuristic entirely.

Regression covered here: a PDF's outline can contain a dangling/
unresolvable bookmark -- PyMuPDF reports its page as -1 rather than
raising. The first version of this code clamped invalid pages via
max(1, page), which turned an invalid entry into a phantom chapter
starting at page 1 with nothing after it to bound page_end -- swallowing
the entire rest of the document into one node's text (~2.4M characters on
the real textbook that surfaced this, verified live: caused an
openai.RateLimitError trying to synthesize an answer from it). Fixed by
dropping invalid-page entries instead of clamping them -- a general fix,
since any PDF with a broken outline bookmark hits the identical failure
mode, not just this one document.

Run with:
    python -m pytest tests/test_parser_toc_based_structure_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock


from src.document.light.parser import LightPdfParser, _PageExtract, _PdfBlock
from src.graph.axis1_structural import Axis1StructuralBuilder
from src.graph.chunker import StructuralChunker
from src.models import NodeType


def _extract(page_num: int, text: str) -> _PageExtract:
    block = _PdfBlock(text=text, page=page_num, bbox=[10.0, 10.0, 200.0, 20.0])
    return _PageExtract(page=page_num, text=text, blocks=[block])


def _parser() -> LightPdfParser:
    return LightPdfParser()


def _build_from_toc(extracts, toc, name, count):
    ir = _parser()._to_document_ir(extracts, toc, name, count)
    chunks = StructuralChunker().chunk(ir)
    return Axis1StructuralBuilder()._build_from_toc(ir, chunks)


# ── _usable_toc ────────────────────────────────────────────────────────


def test_usable_toc_returns_none_for_too_few_entries():
    doc = MagicMock()
    doc.get_toc.return_value = [[1, "Chapter 1", 1], [1, "Chapter 2", 10]]
    assert _parser()._usable_toc(doc) is None


def test_usable_toc_returns_none_without_any_level_one_entry():
    doc = MagicMock()
    doc.get_toc.return_value = [[2, f"Section {i}", i] for i in range(1, 8)]
    assert _parser()._usable_toc(doc) is None


def test_usable_toc_returns_none_on_exception():
    doc = MagicMock()
    doc.get_toc.side_effect = RuntimeError("no outline")
    assert _parser()._usable_toc(doc) is None


def test_usable_toc_returns_toc_when_valid():
    doc = MagicMock()
    toc = [[1, f"Chapter {i}", i * 10] for i in range(1, 8)]
    doc.get_toc.return_value = toc
    assert _parser()._usable_toc(doc) == toc


# ── _build_from_toc: normal structure ────────────────────────────────────


def test_build_from_toc_creates_nested_chapters_and_sections():
    extracts = [_extract(p, f"page {p} body text") for p in range(1, 21)]
    toc = [
        (1, "Chapter 1", 1),
        (2, "Section 1.1", 2),
        (2, "Section 1.2", 6),
        (1, "Chapter 2", 11),
        (2, "Section 2.1", 12),
    ]

    nodes, edges = _build_from_toc(extracts, toc, "book", 20)

    chapters = [n for n in nodes if n.type == NodeType.CHAPTER]
    sections = [n for n in nodes if n.type == NodeType.SECTION]
    assert len(chapters) == 2
    assert len(sections) == 3
    ch1 = next(n for n in chapters if n.title == "Chapter 1")
    assert ch1.page_start == 1
    assert ch1.page_end == 10  # bounded by Chapter 2's start (11) - 1
    ch2 = next(n for n in chapters if n.title == "Chapter 2")
    assert ch2.page_start == 11
    assert ch2.page_end == 20  # last entry, bounded by page_count


# ── _build_from_toc: dangling/invalid outline entry regression ──────────


def test_build_from_toc_drops_entry_with_negative_page_instead_of_clamping():
    extracts = [_extract(p, f"page {p} body text") for p in range(1, 21)]
    toc = [
        (1, "Chapter 1", 1),
        (1, "Chapter 2", 11),
        # Dangling bookmark: PyMuPDF reports an unresolvable link as page -1.
        (1, "book.pdf", -1),
        (2, "Blank Page", 2),
    ]

    nodes, edges = _build_from_toc(extracts, toc, "book", 20)

    chapters = [n for n in nodes if n.type == NodeType.CHAPTER]
    sections = [n for n in nodes if n.type == NodeType.SECTION]

    # The invalid entry (and its would-be child) must never become nodes.
    assert not any(n.title == "book.pdf" for n in chapters)
    assert not any(n.title == "Blank Page" for n in sections)
    assert len(chapters) == 2

    # Chapter 2 (the real last chapter) must still be bounded by the real
    # page count, not swallow-everything from a dropped phantom entry.
    ch2 = next(n for n in chapters if n.title == "Chapter 2")
    assert ch2.page_end == 20
    assert len(ch2.text) < 1000  # nowhere near a whole-document-sized blob


def test_build_from_toc_drops_entry_with_zero_page():
    extracts = [_extract(p, f"page {p} body text") for p in range(1, 21)]
    toc = [
        (1, "Chapter 1", 1),
        (1, "junk", 0),
        (1, "Chapter 2", 11),
    ]

    nodes, _edges = _build_from_toc(extracts, toc, "book", 20)
    chapters = [n for n in nodes if n.type == NodeType.CHAPTER]
    assert {n.title for n in chapters} == {"Chapter 1", "Chapter 2"}
