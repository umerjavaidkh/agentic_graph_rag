"""tests/unstructured/test_underspecified_gate_unit.py — ask, don't guess.

Similarity search always returns a nearest neighbour. A question that names
no document and carries no topic ("What is the value?") therefore scoped to
whichever document happened to rank first and was answered with full
confidence -- 0/5 on the ambiguous shape, every one of them confident.

The property under test: the decision to answer must require the question to
have actually been placed, and that requirement must not fire on well-formed
questions.
"""
from __future__ import annotations

from src.unstructured.retrieval.strategies.vector_first_hybrid import (
    MIN_KEYWORDS_UNSCOPED,
)
from src.unstructured.retrieval.services.ranking import RankingService

_ranking = RankingService()


def _content_words(q: str) -> int:
    return len(_ranking._content_keywords_from_query(q))


def test_bare_questions_carry_too_few_content_words_to_place():
    """These are the two that were answered confidently from an arbitrary doc."""
    for q in ("What is the value?", "What is the rate?"):
        assert _content_words(q) < MIN_KEYWORDS_UNSCOPED, q


def test_well_formed_questions_are_not_caught_by_the_gate():
    """The gate is worthless if it also declines questions it can answer."""
    for q in (
        "What are the major supply chain risks in NIST SP 800-161r1?",
        "In IRS Publication 225, what is the standard mileage rate for 2025?",
        "What is the table of contents of the Go.Data annual report?",
    ):
        assert _content_words(q) >= MIN_KEYWORDS_UNSCOPED, q


def test_marker_chunk_is_returned_verbatim_not_paraphrased():
    """The clarification is the answer; sending it through the LLM only hedges it."""
    from src.unstructured.retrieval.graph import _generate_document_answer

    out = _generate_document_answer(
        "What is the value?",
        {},
        [{"id": "underspecified", "title": "Which document?", "text": "Name a document."}],
    )

    assert out["answer"] == "Name a document."
    assert out["low_confidence"] is True
    assert out["underspecified"] is True
