"""
tests/test_document_resolver_word_boundary_unit.py — document resolution
matches query terms against node/document title & text by WORD, not by
raw substring.

Regression, found via a live multi-turn QA session: a follow-up question
"What was diluted EPS for 2025?" (no company named, thread already
discussing a JNJ 10-K) resolved to an unrelated physics textbook instead
of staying on JNJ. Root cause: doc_name_terms() picks up "EPS" as a
document-name anchor (all-caps mid-sentence token), and
resolve_document_for_query_strict's Cypher matched it via a raw
`CONTAINS term` substring check -- which also matches "eps" INSIDE
unrelated longer words ("st-EPS-", "sw-EPS-") throughout the (much
longer) physics textbook, outnumbering the financial filing's genuine
"EPS" mentions and winning strict resolution. Strict resolution runs
BEFORE the conversation's thread-continuity hint is even consulted, so
the wrong document then got saved as the new hint -- silently corrupting
every subsequent generic-vocabulary follow-up in that thread until a
strongly distinctive term ("MedTech") happened to reset it. Verified live
against the real corpus (JNJ 10-K + physics textbook + several other
filings): "What was the effective tax rate for 2025?", "Item 1A", "Item
7", "Item 9A", and "litigation" all inherited the same wrong document
once corrupted.

Fixed via _WORD_BOUNDARY_PATTERN: every term-vs-title/text match in
document_resolver.py uses a `\\bterm\\b`-wrapped regex (`=~`) instead of
`CONTAINS`. This is a strictly narrower match than substring CONTAINS --
it can only remove accidental collisions, never introduce a match that
wasn't already semantically present (verified: still correctly matches
inside hyphenated/structured content like "non-GAAP" or "gs-10k-2026").

These tests exercise the regex semantics directly (Python's re, same
`(?s).*\\bterm\\b.*` full-match shape the Cypher pattern produces) rather
than a live Neo4j round trip -- fast, and the shape is what actually
matters; live behavior was independently confirmed against the real
Neo4j instance during development.

Run with:
    python -m pytest tests/test_document_resolver_word_boundary_unit.py -v
"""
from __future__ import annotations

import re
import sys
from pathlib import Path


from src.unstructured.retrieval.services.document_resolver import _WORD_BOUNDARY_PATTERN


def _matches(text: str, term: str) -> bool:
    """Reproduce the Cypher pattern's semantics in Python for testing:
    `(?s).*\\bterm\\b.*` as a FULL match (Cypher's =~ is a full match, not
    a search) -- same shape _WORD_BOUNDARY_PATTERN builds via string
    concatenation in Cypher."""
    pattern = r"(?s).*\b" + re.escape(term) + r"\b.*"
    return bool(re.fullmatch(pattern, text, re.I))


def test_pattern_constant_is_shaped_as_expected():
    # Sanity-check the actual Cypher-fragment string this file's tests
    # stand in for -- catches an escaping regression directly.
    assert _WORD_BOUNDARY_PATTERN == "('(?s).*\\\\b' + term + '\\\\b.*')"


def test_short_acronym_does_not_match_inside_a_longer_word():
    assert _matches("several steps forward", "eps") is False
    assert _matches("the sweeps of the pendulum", "eps") is False


def test_short_acronym_matches_as_its_own_word():
    assert _matches("diluted eps was 10.10", "eps") is True
    assert _matches("eps", "eps") is True


def test_matches_inside_hyphenated_compound_word():
    # Hyphens are non-word characters, so \b still fires around them --
    # switching to word-boundary matching must not lose this.
    assert _matches("a non-gaap measure", "gaap") is True
    assert _matches("filed as gs-10k-2026-02-25", "10k") is True


def test_matches_term_followed_by_punctuation():
    assert _matches("net income (eps).", "eps") is True


def test_does_not_match_term_as_part_of_a_different_word():
    assert _matches("the company reports epsilon particles", "eps") is False
