"""
tests/test_scope_phrase_retrieval_unit.py — a question naming a scope
("International Upstream") retrieves THAT scope's chunk, not a sibling
segment's identically-shaped table.

Regression, found by running 10 real MD&A questions against an ingested
264-page 10-K: 3 of 10 failed, and the failures were worse than misses --
asked for International Upstream's Liquids Production (962 MBD) the pipeline
answered "126 MBD" (a different figure from a sibling segment), and asked
about a sentence quoted verbatim out of the filing it answered with another
segment's number entirely.

Root cause, measured on that document:
  * one chunk straddles two segments (section_25_20, 2,118 chars, holds BOTH
    the "International Upstream" and "U.S. Downstream" tables), so its
    embedding is a blend of the two;
  * 15 nodes contain "net oil-equivalent production", 18 contain "liquids
    production" -- every segment's table repeats the same row labels, so
    vector cosine cannot separate them;
  * the only mechanism that repaired scope (_pin_firmwide_summary_chunks)
    is gated on is_firmwide_financial_metric_question, which is False for
    precisely the segment-scoped questions that need it.

The chunk was reachable the whole time -- a fulltext query for the quoted
phrase "International Upstream" ranks it top -- so this is a retrieval-and-
pinning fix, no reingest required.

Run with:
    python -m pytest tests/test_scope_phrase_retrieval_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path


from src.retrieval.unstructured.services.ranking import RankingService


def _ranking() -> RankingService:
    return RankingService.__new__(RankingService)


# ── scope_phrases_from_query ────────────────────────────────────────────────


def test_extracts_segment_name_from_question():
    r = _ranking()
    phrases = r.scope_phrases_from_query(
        "According to the Chevron 2025 Annual Report, how did International "
        "Upstream net oil-equivalent production change from 2024 to 2025?"
    )
    assert "International Upstream" in phrases


def test_extracts_abbreviated_segment_name():
    """"U.S. Downstream" must survive tokenization -- the period is part of
    the scope name, not a sentence boundary."""
    r = _ranking()
    phrases = r.scope_phrases_from_query(
        "How did U.S. Downstream earnings change from 2024 to 2025?"
    )
    assert "U.S. Downstream" in phrases


def test_question_with_no_proper_scope_yields_nothing():
    """A question naming no scope must produce no phrases at all, so the
    whole scope path stays a no-op rather than scoping to something
    arbitrary."""
    r = _ranking()
    assert r.scope_phrases_from_query("what is the total revenue") == []


def test_year_is_not_treated_as_a_scope_phrase():
    """A four-digit year is not a scope -- every segment's table contains
    every year, so scoping by one would select nothing useful."""
    r = _ranking()
    phrases = r.scope_phrases_from_query("What changed between 2024 and 2025?")
    assert not any("2024" in p or "2025" in p for p in phrases)


def test_empty_query_is_safe():
    r = _ranking()
    assert r.scope_phrases_from_query("") == []
    assert r.scope_phrases_from_query(None) == []


# ── _pin_scope_chunks ───────────────────────────────────────────────────────


def _hit(cid: str, text: str, score: float = 1.0) -> dict:
    return {"id": cid, "title": cid, "text": text, "score": score}


def test_scope_hit_is_pinned_above_existing_items():
    """The whole point: the right-scope chunk must reach the context window
    even though sibling chunks already occupy every slot."""
    r = _ranking()
    items = [_hit(f"other_{i}", "Liquids Production 500 MBD", 9.0) for i in range(8)]
    scope = [_hit("intl_upstream", "International Upstream Liquids Production 962 MBD")]

    out = r._pin_scope_chunks(items, scope, limit=8)

    assert out[0]["id"] == "intl_upstream"
    assert len(out) == 8


def test_pinned_scope_chunk_outscores_ordinary_items():
    r = _ranking()
    items = [_hit("other", "x", 9.99)]
    scope = [_hit("scoped", "Liquids Production 962 MBD", 1.0)]

    out = r._pin_scope_chunks(items, scope, limit=5)

    assert out[0]["score"] > items[0]["score"]
    assert "via:scope_pin" in out[0]["related"]


def test_chunks_with_figures_pinned_ahead_of_prose():
    """A metric question is answered by the scope's table, not by the prose
    section that merely mentions the scope."""
    r = _ranking()
    scope = [
        _hit("prose", "International Upstream operations are discussed here."),
        _hit("table", "International Upstream | Liquids Production | 962 |"),
    ]
    out = r._pin_scope_chunks([], scope, limit=5)
    assert out[0]["id"] == "table"


def test_low_confidence_extract_ranked_last():
    r = _ranking()
    scope = [
        _hit("bad", "[low confidence extract] International Upstream 962"),
        _hit("good", "International Upstream 962 MBD"),
    ]
    out = r._pin_scope_chunks([], scope, limit=5)
    assert out[0]["id"] == "good"


def test_no_scope_hits_leaves_items_untouched():
    """Every question that names no scope must be completely unaffected --
    this path is additive, not a change to normal ranking."""
    r = _ranking()
    items = [_hit("a", "x", 3.0), _hit("b", "y", 2.0)]
    assert r._pin_scope_chunks(items, [], limit=5) == items


def test_duplicate_scope_hits_are_not_double_pinned():
    r = _ranking()
    scope = [_hit("same", "International Upstream 962"), _hit("same", "International Upstream 962")]
    out = r._pin_scope_chunks([], scope, limit=5)
    assert len(out) == 1


def test_scope_pin_respects_limit():
    r = _ranking()
    scope = [_hit(f"s{i}", f"International Upstream {i}") for i in range(10)]
    items = [_hit(f"o{i}", "other") for i in range(10)]
    out = r._pin_scope_chunks(items, scope, limit=6)
    assert len(out) == 6
