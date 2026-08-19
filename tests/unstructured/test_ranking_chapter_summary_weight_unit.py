"""tests/test_ranking_chapter_summary_weight_unit.py — _merge_and_rank's
chapter_summary_hits weighting.

Covers: chapter summaries are boosted for synthesis-shaped questions
("what does this document discuss") but only lightly weighted otherwise —
a short rollup paragraph shouldn't be able to outrank a specific on-topic
chunk for a normal fact-lookup question, but should be genuinely
competitive for a synthesis question, matching the whole point of this
feature (Microsoft GraphRAG-style "what's this about" coverage without a
graph-clustering algorithm).

Run with:
    python -m pytest tests/test_ranking_chapter_summary_weight_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


from src.unstructured.retrieval.services.ranking import RankingService


def _chapter_summary_hit(score: float = 0.9) -> dict:
    return {
        "id": "chapter_1",
        "title": "Item 1A. Risk Factors",
        "text": "This chapter discusses credit, market, and operational risk.",
        "score": score,
        "related": ["via:chapter_summary"],
    }


def _fulltext_hit(score: float = 0.6) -> dict:
    return {
        "id": "section_42",
        "title": "Credit Risk",
        "text": "Credit risk arises when a borrower fails to meet obligations.",
        "score": score,
        "node_label": "Section",
    }


def test_chapter_summary_wins_for_synthesis_question():
    ranking = RankingService()
    items = ranking._merge_and_rank(
        "What does this filing discuss overall?",
        vector_hits=[],
        fulltext_hits=[_fulltext_hit(score=0.6)],
        graph_hits=[],
        seed_scores={},
        limit=5,
        chapter_summary_hits=[_chapter_summary_hit(score=0.9)],
        chapter_summary_boost=True,
    )
    ids = [i["id"] for i in items]
    assert ids[0] == "chapter_1"


def test_chapter_summary_does_not_dominate_normal_question():
    ranking = RankingService()
    items = ranking._merge_and_rank(
        "What is the credit risk described in this section?",
        vector_hits=[],
        fulltext_hits=[_fulltext_hit(score=0.95)],
        graph_hits=[],
        seed_scores={},
        limit=5,
        chapter_summary_hits=[_chapter_summary_hit(score=0.9)],
        synthesis=False,
    )
    ids = [i["id"] for i in items]
    assert ids[0] == "section_42"


def test_no_chapter_summary_hits_is_a_no_op():
    ranking = RankingService()
    items = ranking._merge_and_rank(
        "What does this filing discuss overall?",
        vector_hits=[],
        fulltext_hits=[_fulltext_hit()],
        graph_hits=[],
        seed_scores={},
        limit=5,
        synthesis=True,
    )
    assert len(items) == 1
    assert items[0]["id"] == "section_42"


def test_chapter_summary_tagged_with_related_source():
    ranking = RankingService()
    items = ranking._merge_and_rank(
        "What does this filing discuss overall?",
        vector_hits=[],
        fulltext_hits=[],
        graph_hits=[],
        seed_scores={},
        limit=5,
        chapter_summary_hits=[_chapter_summary_hit()],
        chapter_summary_boost=True,
    )
    assert "via:chapter_summary" in items[0]["related"]
