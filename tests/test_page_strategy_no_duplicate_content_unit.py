"""tests/test_page_strategy_no_duplicate_content_unit.py — PageStrategy's
plain page-text answer must not repeat the same content multiple times.

Regression: verified live on a real SEC 10-K -- a "what's on page 3"
question returned the same table three times in one answer. Root cause:
_structural_page_retrieve unconditionally appended (1) the Page's own
search_text, (2) each CONTAINS-ed Region's own text -- extracted from the
same page's blocks, with no vision-only content fetched by this query, so
it never added anything Page text didn't already have -- and (3) the full
hydrated body of any Section CONTAINing this page -- built by concatenating
all of that section's child pages verbatim, so it always re-included this
exact page's content again. Fixed by dropping the Region fetch entirely and
replacing the full section-body dump with a lightweight title-only
reference (keeps the "what part of the document is this" orientation
without repeating page content).

Run with:
    python -m pytest tests/test_page_strategy_no_duplicate_content_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.retrieval.unstructured.strategies.page import PageStrategy
from src.retrieval.unstructured.services.document_resolver import DocumentResolver
from src.retrieval.unstructured.services.formatter import ResponseFormatter
from src.retrieval.unstructured.services.graph_seeds import GraphSeedService
from src.retrieval.unstructured.services.ranking import RankingService

_TABLE_TEXT = "| Downstream | 18 |\n|---|---|\n| Other Businesses | 20 |"


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def single(self):
        return self._row


class _FakeSession:
    def __init__(self, row):
        self._row = row
        self.queries: list[str] = []

    def run(self, cypher, **kwargs):
        self.queries.append(cypher)
        return _FakeResult(self._row)


@pytest.fixture()
def page() -> PageStrategy:
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    resolver.resolve_document_for_query = lambda *a, **kw: ("doc-1", "Doc Title")
    return PageStrategy(resolver, ResponseFormatter())


def test_page_answer_never_repeats_the_same_table_content(page):
    row = {
        "id": "doc-1_page_3",
        "title": "Page 3",
        "text": f"TABLE OF CONTENTS\n\n{_TABLE_TEXT}",
        "visual_content": "",
        "pdf_page": 3,
        "document_page": "3",
        "doc_title": "Doc Title",
        "section_titles": ["TABLE OF CONTENTS"],
    }
    session = _FakeSession(row)
    items = page._structural_page_retrieve(session, "what is on page 3", tenant_id="")
    assert items
    text = items[0]["text"]
    assert text.count(_TABLE_TEXT) == 1


def test_page_answer_has_no_region_or_related_section_dump(page):
    row = {
        "id": "doc-1_page_3",
        "title": "Page 3",
        "text": f"TABLE OF CONTENTS\n\n{_TABLE_TEXT}",
        "visual_content": "",
        "pdf_page": 3,
        "document_page": "3",
        "doc_title": "Doc Title",
        "section_titles": ["TABLE OF CONTENTS"],
    }
    session = _FakeSession(row)
    items = page._structural_page_retrieve(session, "what is on page 3", tenant_id="")
    text = items[0]["text"]
    assert "## Region" not in text
    assert "## Related section" not in text


def test_page_answer_includes_lightweight_section_title_reference(page):
    row = {
        "id": "doc-1_page_3",
        "title": "Page 3",
        "text": "Some page text.",
        "visual_content": "",
        "pdf_page": 3,
        "document_page": "3",
        "doc_title": "Doc Title",
        "section_titles": ["TABLE OF CONTENTS", "PART I"],
    }
    session = _FakeSession(row)
    items = page._structural_page_retrieve(session, "what is on page 3", tenant_id="")
    text = items[0]["text"]
    assert "## Part of section(s)" in text
    assert "TABLE OF CONTENTS" in text
    assert "PART I" in text


def test_page_answer_omits_section_block_when_no_sections(page):
    row = {
        "id": "doc-1_page_3",
        "title": "Page 3",
        "text": "Some page text.",
        "visual_content": "",
        "pdf_page": 3,
        "document_page": "3",
        "doc_title": "Doc Title",
        "section_titles": [],
    }
    session = _FakeSession(row)
    items = page._structural_page_retrieve(session, "what is on page 3", tenant_id="")
    text = items[0]["text"]
    assert "## Part of section(s)" not in text
