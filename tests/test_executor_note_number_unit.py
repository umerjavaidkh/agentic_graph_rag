"""tests/test_executor_note_number_unit.py — "Note N" financial-statement
footnote references are recognized as subsection lookups.

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

Run with:
    python -m pytest tests/test_executor_note_number_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.retrieval.unstructured.executor import DocumentQueryExecutor
from src.retrieval.unstructured.strategies.subsection import SubsectionStrategy
from src.retrieval.unstructured.services.document_resolver import DocumentResolver
from src.retrieval.unstructured.services.formatter import ResponseFormatter
from src.retrieval.unstructured.services.graph_seeds import GraphSeedService
from src.retrieval.unstructured.services.ranking import RankingService


@pytest.fixture()
def executor() -> DocumentQueryExecutor:
    return DocumentQueryExecutor()


def test_parse_section_number_recognizes_note_n(executor):
    assert executor.parse_section_number("What does Note 3 discuss?") == "note 3"
    assert executor.parse_section_number("What does Note 7 (Segment Information) report?") == "note 7"


def test_parse_section_number_still_matches_dotted_numbers(executor):
    assert executor.parse_section_number("What is under section 2.5?") == "2.5"


def test_parse_section_number_returns_none_for_unrelated_query(executor):
    assert executor.parse_section_number("What was net sales for the quarter?") is None


def test_is_subsection_request_true_for_note_n(executor):
    assert executor.is_subsection_request("What does Note 3 (Commitments and Contingencies) discuss?") is True


def test_is_subsection_request_unaffected_for_ordinary_queries(executor):
    assert executor.is_subsection_request("What was net sales for the quarter?") is False


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


def test_retrieve_returns_none_for_unrelated_query(subsection):
    session = _FakeSession({"sid": "x", "stitle": "irrelevant", "stext": "irrelevant", "children": []})
    response = subsection.retrieve(
        session, "What was net sales for the quarter?", tenant_id="default", limit=8, ctx=MagicMock(role=MagicMock(value="admin"), user_id="u1"),
    )
    assert response is None
