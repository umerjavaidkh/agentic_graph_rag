"""tests/test_page_strategy_figure_number_unit.py — bare "Figure N" questions
resolve via the OCR/visual pathway without a page number.

Regression: "What does Figure 1 show in this report?" has no page number, so
_structural_page_visual_retrieve bailed out immediately (`if pdf_page is None
and not doc_page: return []`), and is_visual_page_question's gate also
required either a page number or a "show/display the image/page" phrasing
that "Figure 1 show" doesn't match ("show" isn't immediately followed by
image/figure/page). The figure's actual visual_content was never reached,
even though the page-caption text ("Figure 1: ...") needed to find it was
already sitting in page.text from ingestion. Verified live: this exact
question returned "The content of Figure 1 is not provided..." against a
real ingested document that does have a captioned Figure 1.

Run with:
    python -m pytest tests/test_page_strategy_figure_number_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


from src.unstructured.retrieval.query_intent import is_page_question, is_visual_page_question
from src.unstructured.retrieval.strategies.page import PageStrategy
from src.unstructured.retrieval.services.document_resolver import DocumentResolver
from src.unstructured.retrieval.services.formatter import ResponseFormatter
from src.unstructured.retrieval.services.graph_seeds import GraphSeedService
from src.unstructured.retrieval.services.ranking import RankingService


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    """Branches on which query is being run, matching the two distinct
    queries _structural_page_visual_retrieve issues: the figure->page
    resolution lookup (returns page_text rows) and the main page-visual
    fetch (returns a single page+regions row)."""

    def __init__(self, figure_rows=None, page_row=None):
        self.figure_rows = figure_rows or []
        self.page_row = page_row
        self.queries: list[str] = []

    def run(self, cypher, **kwargs):
        self.queries.append(cypher)
        if "p.pdf_page AS pdf_page, p.search_text AS page_text" in cypher:
            return _FakeResult(self.figure_rows)
        return _FakeResult([self.page_row] if self.page_row else [])


@pytest.fixture()
def page() -> PageStrategy:
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    resolver.resolve_document_for_query = lambda *a, **kw: ("doc-1", "Doc Title")
    return PageStrategy(resolver, ResponseFormatter())


def test_bare_figure_question_is_visual_page_question():
    assert is_visual_page_question("What does Figure 1 show in this report?") is True
    assert is_page_question("What does Figure 1 show in this report?") is False


def test_ordinary_question_is_not_visual_page_question():
    assert is_visual_page_question("What are the compliance requirements for gifts?") is False


def test_resolve_pdf_page_by_figure_number_finds_correct_page(page):
    session = _FakeSession(
        figure_rows=[
            {"pdf_page": 5, "page_text": "Some unrelated figure discussion, Figure 2: budget chart"},
            {"pdf_page": 8, "page_text": "GO.DATA > Annual Report 2021\nFigure 1: Outbreak map overview"},
        ]
    )
    result = page._resolve_pdf_page_by_figure_number(session, "doc-1", "", "1")
    assert result == 8


def test_resolve_pdf_page_by_figure_number_returns_none_when_not_found(page):
    session = _FakeSession(figure_rows=[{"pdf_page": 5, "page_text": "Figure 2: budget chart"}])
    result = page._resolve_pdf_page_by_figure_number(session, "doc-1", "", "1")
    assert result is None


def test_resolve_pdf_page_by_figure_number_returns_none_without_doc_id(page):
    session = _FakeSession(figure_rows=[{"pdf_page": 5, "page_text": "Figure 1: whatever"}])
    result = page._resolve_pdf_page_by_figure_number(session, None, "", "1")
    assert result is None


def test_structural_page_visual_retrieve_falls_back_to_figure_lookup(page):
    session = _FakeSession(
        figure_rows=[{"pdf_page": 8, "page_text": "Figure 1: Outbreak map overview"}],
        page_row={
            "page_id": "doc-1_page_8",
            "page_title": "Page 8",
            "page_text": "Figure 1: Outbreak map overview",
            "page_visual": "A map showing outbreak locations across regions.",
            "pdf_page": 8,
            "document_page": "1",
            "doc_title": "Doc Title",
            "regions": [],
        },
    )
    items = page._structural_page_visual_retrieve(session, "What does Figure 1 show in this report?")
    assert items
    assert items[0]["pdf_page"] == 8


def test_structural_page_visual_retrieve_empty_when_figure_not_found(page):
    session = _FakeSession(figure_rows=[])
    items = page._structural_page_visual_retrieve(session, "What does Figure 1 show in this report?")
    assert items == []
