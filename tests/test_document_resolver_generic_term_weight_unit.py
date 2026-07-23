"""tests/test_document_resolver_generic_term_weight_unit.py —
resolve_document_for_query's generic-term (document_match_terms) fallback
tier uses IDF weighting, not a flat count().

Regression: the same bug class already fixed once in doc_name_terms
(see test_document_resolver_hint_unit.py) had an unfixed twin in this
fallback tier. document_match_terms is deliberately broader than
doc_name_terms (includes ordinary content words like "annual"/"report",
not just anchors/proper nouns) — feeding those into a flat count()
ORDER BY hits DESC reliably favored whichever document had the most
content overall. Verified live: "What is the table of contents of this
annual report?" against a corpus with one large SEC filing and one small
WHO report resolved to the SEC filing, purely because "annual"/"report"
appear more often in a much bigger document, despite the question having
zero relation to it.

Run with:
    python -m pytest tests/test_document_resolver_generic_term_weight_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.retrieval.unstructured.services.document_resolver import DocumentResolver
from src.retrieval.unstructured.services.graph_seeds import GraphSeedService
from src.retrieval.unstructured.services.ranking import RankingService


@pytest.fixture()
def resolver() -> DocumentResolver:
    return DocumentResolver(GraphSeedService(RankingService()))


def _term_hit(term, cnt, title_match=False):
    return {"term": term, "cnt": cnt, "title_match": title_match}


def test_distinctive_term_outweighs_generic_terms_from_a_bigger_document(resolver):
    """A small document that's genuinely distinctive on one term ("godata")
    must beat a big document that only wins on generic terms ("annual",
    "report") appearing more often purely because it has more content."""
    rows = [
        {
            "id": "big-sec-filing",
            "title": "Big SEC Filing",
            "term_hits": [
                _term_hit("annual", 40),
                _term_hit("report", 55),
                _term_hit("godata", 0),
            ],
        },
        {
            "id": "small-who-report",
            "title": "Small WHO Report",
            "term_hits": [
                _term_hit("annual", 3),
                _term_hit("report", 4),
                _term_hit("godata", 12, title_match=True),
            ],
        },
    ]
    result = resolver._pick_best_by_term_weight(rows, ["annual", "report", "godata"])
    assert result is not None
    assert result[0] == "small-who-report"


def test_all_generic_terms_still_returns_a_result_not_none(resolver):
    """Pure generic-term matching (no distinctive term at all) is still
    this tier's whole point — it must not regress to refusing to answer;
    it should just no longer be dominated by document size for identical
    per-document-frequency terms."""
    rows = [
        {"id": "doc-a", "title": "Doc A", "term_hits": [_term_hit("annual", 40), _term_hit("report", 55)]},
        {"id": "doc-b", "title": "Doc B", "term_hits": [_term_hit("annual", 3), _term_hit("report", 4)]},
    ]
    result = resolver._pick_best_by_term_weight(rows, ["annual", "report"])
    assert result is not None
    # Both terms appear in both docs (df=2 for each, uniform weight) so the
    # raw-count-heavier document still wins here -- that's correct: with NO
    # distinctive term anywhere, size is the only real signal left, and
    # this tier is allowed to guess (unlike the strict resolver).
    assert result[0] == "doc-a"


def test_no_matches_returns_none(resolver):
    result = resolver._pick_best_by_term_weight([], ["anything"])
    assert result is None


def test_title_match_is_a_strong_signal(resolver):
    rows = [
        {"id": "doc-a", "title": "Doc A", "term_hits": [_term_hit("widget", 50)]},
        {"id": "godata", "title": "Go.Data Report", "term_hits": [_term_hit("widget", 1, title_match=False), _term_hit("godata", 0, title_match=True)]},
    ]
    result = resolver._pick_best_by_term_weight(rows, ["widget", "godata"])
    assert result[0] == "godata"
