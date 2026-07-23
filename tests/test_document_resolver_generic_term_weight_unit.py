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
from unittest.mock import MagicMock

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


def test_linear_idf_was_not_enough_log_idf_flips_the_winner(resolver):
    """Regression: linear IDF weighting (total_docs / doc_freq) correctly
    downweights a boilerplate term, but a raw count in the hundreds (a much
    bigger document) still nearly cancels out a genuinely distinctive
    term's contribution -- verified live with these exact counts: a query
    mentioning "amazon" alongside standard financial/legal boilerplate
    ("note", "commitments", "contingencies", "discuss") scored the huge
    JPM 10-K at 24.5 vs the correct AMZN 10-Q at 24.22 under linear IDF, a
    coin-flip margin that isn't reliable. log-IDF fixes this by scaling
    down non-distinctive terms far more aggressively while barely
    touching the truly distinctive one."""
    rows = [
        {
            "id": "aapl-10k-2024", "title": "AAPL 10-K",
            "term_hits": [_term_hit("note", 60), _term_hit("commitments", 3), _term_hit("contingencies", 6), _term_hit("discuss", 5)],
        },
        {
            "id": "jpm-10k-2017-02-28", "title": "JPM 10-K",
            "term_hits": [_term_hit("note", 559), _term_hit("commitments", 131), _term_hit("contingencies", 14), _term_hit("discuss", 249)],
        },
        {
            "id": "amzn-10q-2016-07-29", "title": "AMZN 10-Q",
            "term_hits": [_term_hit("note", 39), _term_hit("commitments", 16), _term_hit("contingencies", 16), _term_hit("discuss", 8), _term_hit("amazon", 34)],
        },
        {
            "id": "stratec-compliance-policy", "title": "STRATEC Policy",
            "term_hits": [_term_hit("note", 7), _term_hit("discuss", 4)],
        },
        {
            "id": "godata-annual-report", "title": "Go.Data Report",
            "term_hits": [_term_hit("note", 9), _term_hit("discuss", 13), _term_hit("amazon", 2)],
        },
    ]
    terms = ["note", "commitments", "contingencies", "discuss", "amazon"]
    result = resolver._pick_best_by_term_weight(rows, terms)
    assert result is not None
    assert result[0] == "amzn-10q-2016-07-29"


def test_require_distinctive_declines_when_every_term_is_universal(resolver):
    rows = [
        {"id": "doc-a", "title": "Doc A", "term_hits": [_term_hit("annual", 40), _term_hit("report", 55)]},
        {"id": "doc-b", "title": "Doc B", "term_hits": [_term_hit("annual", 3), _term_hit("report", 4)]},
    ]
    result = resolver._pick_best_by_term_weight(rows, ["annual", "report"], require_distinctive=True)
    assert result is None


def test_require_distinctive_accepts_when_one_term_excludes_a_document(resolver):
    rows = [
        {"id": "doc-a", "title": "Doc A", "term_hits": [_term_hit("annual", 40), _term_hit("report", 55), _term_hit("amazon", 0)]},
        {"id": "doc-b", "title": "Doc B", "term_hits": [_term_hit("annual", 3), _term_hit("report", 4), _term_hit("amazon", 34)]},
    ]
    result = resolver._pick_best_by_term_weight(rows, ["annual", "report", "amazon"], require_distinctive=True)
    assert result is not None
    assert result[0] == "doc-b"


def test_resolve_document_for_query_prefers_confident_term_over_vector_majority():
    """The full priority chain: a confident distinctive-term match must be
    tried BEFORE vector-majority, not after -- vector-majority has no
    defense against corpus-size skew of its own, so if it ran first here
    it would win with its own (wrong) answer before the more reliable
    term-based tier ever got a chance."""
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    resolver.resolve_document_for_query_strict = MagicMock(return_value=(None, None))
    resolver.document_match_terms = MagicMock(return_value=["amazon"])
    resolver.resolve_document_by_vector = MagicMock(
        side_effect=AssertionError("must not reach vector-majority when a confident term match exists")
    )

    class _FakeSession:
        def run(self, cypher, **kwargs):
            # A third, unrelated document with zero mentions is what makes
            # "amazon" non-universal (df=2 of 3) -- with only the two
            # candidates present, the term would trivially appear in
            # "every" document and require_distinctive would (correctly)
            # decline, same as it does for genuinely universal boilerplate.
            return [
                {"id": "jpm-10k", "title": "JPM 10-K", "term_hits": [_term_hit("amazon", 2)]},
                {"id": "amzn-10q", "title": "AMZN 10-Q", "term_hits": [_term_hit("amazon", 34)]},
                {"id": "stratec-policy", "title": "STRATEC Policy", "term_hits": [_term_hit("amazon", 0)]},
            ]

    doc_id, title = resolver.resolve_document_for_query(_FakeSession(), "what does amazon report say?", tenant_id="default")
    assert doc_id == "amzn-10q"
