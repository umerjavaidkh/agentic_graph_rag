"""
tests/test_subsection_prefix_boundary_unit.py — structural section/chapter
lookups match a numbered reference by WORD, not by raw string prefix.

Regression, found via a live multi-turn QA session on a real JNJ 10-K
(which has both an "Item 7" and an "Item 7A" section): "What does Item 7
discuss regarding results of operations?" non-deterministically returned
"Item 7A" content instead, because _structural_subsections' Cypher used a
plain `title STARTS WITH sec_num` match with no ORDER BY -- "item 7"
STARTS WITH-matches "Item 7A. Quantitative and Qualitative..." just as
readily as the real "Item 7. Management's discussion...". Same failure
class threatens "chapter 1" matching "Chapter 10"-"Chapter 19", "note 3"
matching a hypothetical "Note 30", etc. -- not specific to this one
document.

Fixed via _PREFIX_BOUNDARY_CYPHER: additionally require the character
right after the matched prefix to be non-alphanumeric (or the title to be
exactly that length). "item 7." and "item 7 -- ..." still match; "item 7a"
no longer does. The existing "Item 9A" support (a genuine letter-suffixed
reference) still works because the query for "Item 9A" ends immediately
after "a" in a title like "Item 9A. Controls..." -- followed by ".", not
another alphanumeric character.

These tests exercise the boundary predicate's semantics directly (mirrors
the Cypher logic in Python) plus confirm the rendered Cypher actually
includes it -- fast, and live behavior was independently confirmed against
the real Neo4j instance during development.

Run with:
    python -m pytest tests/test_subsection_prefix_boundary_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.retrieval.unstructured.strategies.subsection import (
    SubsectionStrategy,
    _PREFIX_BOUNDARY_CYPHER,
)


def _prefix_boundary_ok(title: str, sec_num: str) -> bool:
    """Python mirror of _PREFIX_BOUNDARY_CYPHER's semantics, for testing
    without a live Neo4j round trip."""
    if not title.lower().startswith(sec_num.lower()):
        return False
    if len(title) == len(sec_num):
        return True
    return not title.lower()[len(sec_num)].isalnum()


def test_item_7_does_not_match_item_7a():
    assert _prefix_boundary_ok("Item 7A. Quantitative and Qualitative...", "item 7") is False


def test_item_7_matches_real_item_7():
    assert _prefix_boundary_ok("Item 7. Management's discussion...", "item 7") is True


def test_item_9a_still_matches_its_own_letter_suffixed_title():
    assert _prefix_boundary_ok("Item 9A. Controls and Procedures.", "item 9a") is True


def test_chapter_1_does_not_match_chapter_10_through_19():
    for n in (10, 11, 15, 19):
        assert _prefix_boundary_ok(f"Chapter {n}. Some Title", "chapter 1") is False


def test_chapter_1_matches_real_chapter_1():
    assert _prefix_boundary_ok("Chapter 1. Units and Measurement", "chapter 1") is True


def test_exact_length_title_matches():
    assert _prefix_boundary_ok("Item 7", "item 7") is True


class _CaptureSession:
    """Records the Cypher text of every session.run() call, returns an
    empty result for each -- used to assert the boundary condition is
    actually wired into the rendered query, not just present as an
    unused module-level constant."""

    def __init__(self):
        self.queries: list[str] = []

    def run(self, cypher, **kwargs):
        self.queries.append(cypher)
        result = MagicMock()
        result.single.return_value = None
        return result


def test_boundary_condition_is_present_in_structural_subsections_query():
    strategy = SubsectionStrategy(MagicMock(), MagicMock(), MagicMock())
    strategy._document_resolver.resolve_document_for_query = lambda *a, **kw: ("doc1", "Doc 1")
    session = _CaptureSession()
    strategy._structural_subsections(session, "What does Item 7 discuss?", "item 7", "default")
    assert any(_PREFIX_BOUNDARY_CYPHER in q for q in session.queries)


def test_boundary_condition_is_present_in_multi_reference_query():
    strategy = SubsectionStrategy(MagicMock(), MagicMock(), MagicMock())
    session = _CaptureSession()
    strategy._structural_reference_excerpt(session, "item 7", "default", "doc1")
    assert any(_PREFIX_BOUNDARY_CYPHER in q for q in session.queries)
