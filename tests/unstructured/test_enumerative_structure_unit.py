"""tests/unstructured/test_enumerative_structure_unit.py — listing headings is a graph read.

"List every chapter heading in X" classifies as ENUMERATIVE, and the
enumerative shape never read the hierarchy: it fell through to prose
search and answered "this document does not cover chapter headings" while
outline() returned all 31 of them. One document in five passed, and only
because the hybrid search happened to land on a chunk that listed the
sections -- luck, not retrieval.
"""
from __future__ import annotations

from src.unstructured.retrieval.strategies.vector_first_hybrid import (
    _ASKS_FOR_STRUCTURE,
)


def test_requests_for_the_documents_own_units_are_recognised():
    for q in (
        'List every chapter heading in "arxiv_2608_16178".',
        "What are the sections of this report?",
        "Give me the appendices.",
        "List all subsections.",
        "What are the headings?",
    ):
        assert _ASKS_FOR_STRUCTURE.search(q), q


def test_listing_content_is_not_a_hierarchy_read():
    """The unit word is what makes it structural; the listing verb is not.

    "List the datasets used" is a content question and must keep going to
    the passages, or this fix would swallow half the enumerative shape.
    """
    for q in (
        "List the datasets used in the evaluation.",
        "List every author of this paper.",
        "What are the main contributions?",
        "List the baselines compared against.",
    ):
        assert not _ASKS_FOR_STRUCTURE.search(q), q
