r"""tests/test_executor_note_number_unit.py — "Note N" and "Item N[Letter]"
structural references are recognized as subsection lookups.

Regression: SEC filings organize footnotes as Section nodes titled
"Note 3 — COMMITMENTS AND CONTINGENCIES", "Note 7 — SEGMENT INFORMATION",
etc. — real content that exists verbatim in the graph — but
parse_section_number only matched dotted numeric headings ("2.5"), and
is_subsection_request only fired on "subsection"/"sub section" phrasing.
A bare "Note 3" (no dot, no "subsection" wording) matched neither, so
SubsectionStrategy.retrieve() returned None immediately and the question
fell through to generic hybrid retrieval, which failed to surface the
section despite an exact title match existing. Verified live: "What does
Note 3 (Commitments and Contingencies) discuss?" against a real ingested
10-Q with that exact section returned "not provided in the retrieved
document sections."

Second regression, same class: SEC filings also number top-level items
with a letter-suffixed convention Notes don't use ("Item 9A. Controls and
Procedures", "Item 7A") — the initial Note-only fix's \d+-only pattern
didn't match the trailing letter, so "Item 9A" fell through the same way.
Verified live: "What does Item 9A report about the effectiveness of
JPMorgan's internal controls?" against a real ingested 10-K with that
exact section returned the generic Northwind-misroute fallback message.

Third regression, same class, found via the physics-textbook stress test:
worked examples are titled "Example 2.8" — word THEN a DOTTED number,
unlike Note/Item's word-then-bare-integer shape. _SECTION_NUM_RE alone
extracts the bare "2.8", but subsection.py's lookup matches on `title
STARTS WITH`, and the title starts with "Example", not "2.8" — so even a
bare-number match wouldn't have found it. Needs the same "word + number"
combined-string shape Note/Item already return. Verified live: "Tell me
about Example 2.8 Direction of Motion from Physics book." against a real
ingested textbook with that exact section (confirmed present in Neo4j)
answered "This document does not cover Example 2.8..." before this fix.

Run with:
    python -m pytest tests/test_executor_note_number_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


from src.unstructured.retrieval.executor import DocumentQueryExecutor
from src.unstructured.retrieval.strategies.subsection import SubsectionStrategy
from src.unstructured.retrieval.services.document_resolver import DocumentResolver
from src.unstructured.retrieval.services.formatter import ResponseFormatter
from src.unstructured.retrieval.services.graph_seeds import GraphSeedService
from src.unstructured.retrieval.services.ranking import RankingService


@pytest.fixture()
def executor() -> DocumentQueryExecutor:
    return DocumentQueryExecutor()


def test_parse_section_number_recognizes_note_n(executor):
    assert executor.parse_section_number("What does Note 3 discuss?") == "note 3"
    assert executor.parse_section_number("What does Note 7 (Segment Information) report?") == "note 7"


def test_parse_section_number_recognizes_item_with_letter_suffix(executor):
    assert executor.parse_section_number("What does Item 9A report?") == "item 9a"
    assert executor.parse_section_number(
        "What does Item 9A report about the effectiveness of JPMorgan's internal controls?"
    ) == "item 9a"
    # Bare numeric items (no letter suffix) must still work.
    assert executor.parse_section_number("What does Item 7 discuss?") == "item 7"


def test_parse_section_number_still_matches_dotted_numbers(executor):
    assert executor.parse_section_number("What is under section 2.5?") == "2.5"


def test_parse_section_number_recognizes_example_with_dotted_number(executor):
    assert executor.parse_section_number(
        "Tell me about Example 2.8 Direction of Motion from Physics book."
    ) == "example 2.8"
    assert executor.parse_section_number("What does Example 5.4 show?") == "example 5.4"


def test_parse_section_number_recognizes_chapter_reference(executor):
    assert executor.parse_section_number(
        "List every Check Your Understanding question in Chapter 15."
    ) == "chapter 15"
    assert executor.parse_section_number("Summarize Chapter 3") == "chapter 3"


def test_parse_section_number_returns_none_for_unrelated_query(executor):
    assert executor.parse_section_number("What was net sales for the quarter?") is None


def test_is_subsection_request_true_for_note_n(executor):
    assert executor.is_subsection_request("What does Note 3 (Commitments and Contingencies) discuss?") is True


def test_is_subsection_request_true_for_item_with_letter_suffix(executor):
    assert executor.is_subsection_request("What does Item 9A report about internal controls?") is True


def test_is_subsection_request_unaffected_for_ordinary_queries(executor):
    assert executor.is_subsection_request("What was net sales for the quarter?") is False


def test_is_subsection_request_true_for_example_with_dotted_number(executor):
    assert executor.is_subsection_request(
        "Tell me about Example 2.8 Direction of Motion from Physics book."
    ) is True


def test_has_multiple_structural_references_true_for_two_different_chapters(executor):
    assert executor.has_multiple_structural_references(
        "How does the treatment of momentum in Chapter 9 relate to angular momentum in Chapter 11?"
    ) is True


def test_has_multiple_structural_references_false_for_single_reference(executor):
    assert executor.has_multiple_structural_references(
        "List every Check Your Understanding question in Chapter 15."
    ) is False


def test_has_multiple_structural_references_false_for_same_reference_repeated(executor):
    assert executor.has_multiple_structural_references(
        "Tell me about Chapter 9. I mean Chapter 9 specifically."
    ) is False


def test_has_multiple_structural_references_false_for_no_reference(executor):
    assert executor.has_multiple_structural_references("What was net sales for the quarter?") is False


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def single(self):
        return self._row


class _FakeSession:
    def __init__(self, row):
        self.row = row

    def run(self, cypher, **kwargs):
        return _FakeResult(self.row)


@pytest.fixture()
def subsection() -> SubsectionStrategy:
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    resolver.resolve_document_for_query = lambda *a, **kw: ("amzn-10q", "AMZN 10-Q")
    resolver.document_match_terms = lambda *a, **kw: ["amazon"]
    return SubsectionStrategy(resolver, ResponseFormatter(), DocumentQueryExecutor())


def test_retrieve_returns_note_section_detail(subsection):
    session = _FakeSession({
        "sid": "amzn-10q_section_0_17",
        "stitle": "Note 3 — COMMITMENTS AND CONTINGENCIES",
        "stext": "The Company is subject to various legal proceedings...",
        "children": [],
    })
    response = subsection.retrieve(
        session, "What does Note 3 (Commitments and Contingencies) discuss?",
        tenant_id="default", limit=8, ctx=MagicMock(role=MagicMock(value="admin"), user_id="u1"),
    )
    assert response is not None
    assert response["mode"] == "section_detail"
    assert response["parent_title"] == "Note 3 — COMMITMENTS AND CONTINGENCIES"


def test_retrieve_returns_item_section_detail(subsection):
    session = _FakeSession({
        "sid": "jpm-10k_section_0_52",
        "stitle": "Item 9A. Controls and Procedures.",
        "stext": "Management's Report on Internal Control over Financial Reporting...",
        "children": [],
    })
    response = subsection.retrieve(
        session, "What does Item 9A report about the effectiveness of JPMorgan's internal controls?",
        tenant_id="default", limit=8, ctx=MagicMock(role=MagicMock(value="admin"), user_id="u1"),
    )
    assert response is not None
    assert response["mode"] == "section_detail"
    assert response["parent_title"] == "Item 9A. Controls and Procedures."


def test_retrieve_returns_example_section_detail(subsection):
    session = _FakeSession({
        "sid": "univphysics_section_1_79",
        "stitle": "Example 2.8",
        "stext": "Direction of Motion\n\nIn a Cartesian coordinate system...",
        "children": [],
    })
    response = subsection.retrieve(
        session, "Tell me about Example 2.8 Direction of Motion from Physics book.",
        tenant_id="default", limit=8, ctx=MagicMock(role=MagicMock(value="admin"), user_id="u1"),
    )
    assert response is not None
    assert response["mode"] == "section_detail"
    assert response["parent_title"] == "Example 2.8"


def test_retrieve_returns_chapter_detail_not_section_detail(subsection):
    """Chapter matches must NOT reuse "section_detail" -- that mode is in
    graph.py's _STRUCTURAL_FAST_MODES, which dumps the matched text
    verbatim with no LLM extraction. "List every X in Chapter N" needs the
    LLM to filter/extract from the chapter's content, so it needs its own
    mode name deliberately excluded from that fast-path set."""
    session = _FakeSession({
        "sid": "univphysics_chapter_15",
        "stitle": "Chapter 15. Oscillations",
        "stext": "Oscillations\n\n...chapter body...",
        "children": [],
    })
    response = subsection.retrieve(
        session, "List every Check Your Understanding question in Chapter 15.",
        tenant_id="default", limit=8, ctx=MagicMock(role=MagicMock(value="admin"), user_id="u1"),
    )
    assert response is not None
    assert response["mode"] == "chapter_detail"
    assert response["parent_title"] == "Chapter 15. Oscillations"


def test_retrieve_returns_chapter_children_not_subsection_tree(subsection):
    session = _FakeSession({
        "sid": "univphysics_chapter_15",
        "stitle": "Chapter 15. Oscillations",
        "stext": "Oscillations\n\n...chapter body...",
        "children": [
            {"id": "s15_1", "title": "15.1 Simple Harmonic Motion", "text": "...", "page_start": 753},
        ],
    })
    response = subsection.retrieve(
        session, "List every Check Your Understanding question in Chapter 15.",
        tenant_id="default", limit=8, ctx=MagicMock(role=MagicMock(value="admin"), user_id="u1"),
    )
    assert response is not None
    assert response["mode"] == "chapter_children"


class _MultiRefFakeSession:
    """Returns a different row depending on which sec_num the query was
    run with -- proves retrieve() fetches EACH named reference directly,
    not just the first one parse_section_number would match alone."""

    _ROWS = {
        "chapter 9": {
            "sid": "phys_chapter_9", "stitle": "Chapter 9. Linear Momentum and Collisions",
            "ssummary": "Chapter 9 covers linear momentum, impulse, and collisions.",
            "stext": "", "spage_start": 405,
        },
        "chapter 11": {
            "sid": "phys_chapter_11", "stitle": "Chapter 11. Angular Momentum",
            "ssummary": "Chapter 11 covers angular momentum and its conservation.",
            "stext": "", "spage_start": 549,
        },
    }

    def run(self, cypher, **kwargs):
        row = self._ROWS.get(kwargs.get("sec_num"))
        return _FakeResult(row)


def test_retrieve_fetches_both_named_chapters_for_a_comparison_query(subsection):
    response = subsection.retrieve(
        _MultiRefFakeSession(),
        "How does the treatment of momentum in Chapter 9 relate to angular momentum in Chapter 11?",
        tenant_id="default", limit=8, ctx=MagicMock(role=MagicMock(value="admin"), user_id="u1"),
    )
    assert response is not None
    assert response["mode"] == "multi_reference_detail"
    titles = {c["title"] for c in response["chunks"]}
    assert "Chapter 9. Linear Momentum and Collisions" in titles
    assert "Chapter 11. Angular Momentum" in titles


def test_retrieve_returns_none_when_no_named_reference_resolves(subsection):
    class _EmptyMultiRefSession:
        def run(self, cypher, **kwargs):
            return _FakeResult(None)

    response = subsection.retrieve(
        _EmptyMultiRefSession(),
        "How does the treatment of momentum in Chapter 9 relate to angular momentum in Chapter 11?",
        tenant_id="default", limit=8, ctx=MagicMock(role=MagicMock(value="admin"), user_id="u1"),
    )
    assert response is None


def test_retrieve_returns_none_for_unrelated_query(subsection):
    session = _FakeSession({"sid": "x", "stitle": "irrelevant", "stext": "irrelevant", "children": []})
    response = subsection.retrieve(
        session, "What was net sales for the quarter?", tenant_id="default", limit=8, ctx=MagicMock(role=MagicMock(value="admin"), user_id="u1"),
    )
    assert response is None
