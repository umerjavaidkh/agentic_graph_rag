r"""tests/test_box_strategy_regex_unit.py — BoxStrategy's title-match scoring.

Regression: rf"(?i)\\bbox\\s+{box_n}\\b" is a raw f-string with DOUBLED
backslashes — since raw strings don't collapse \\ to \, the compiled
pattern looks for a literal backslash character before "box"/before the
number, which never appears in real text. The check silently always
failed, so every candidate item got the default score (1.0) regardless of
whether its title/text actually named the requested box, and ranking
degraded to "longest text wins" via the secondary sort key — verified
live: asking about "Box 9" (a real but short section, its substance
mostly on a separate "Box 9 (continued)" node) returned "Box 8" content
instead, because Box 8's chunk happened to be longer.

Run with:
    python -m pytest tests/test_box_strategy_regex_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.retrieval.unstructured.strategies.box import BoxStrategy
from src.retrieval.unstructured.services.document_resolver import DocumentResolver
from src.retrieval.unstructured.services.formatter import ResponseFormatter
from src.retrieval.unstructured.services.graph_seeds import GraphSeedService
from src.retrieval.unstructured.executor import DocumentQueryExecutor
from src.retrieval.unstructured.services.ranking import RankingService


class _FakeSession:
    def __init__(self, rows):
        self.rows = rows

    def run(self, cypher, **kwargs):
        return self.rows


@pytest.fixture()
def box() -> BoxStrategy:
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    resolver.resolve_document_for_query = lambda *a, **kw: ("doc-1", "Doc Title")
    return BoxStrategy(resolver, ResponseFormatter(), DocumentQueryExecutor())


def test_exact_title_match_outranks_a_longer_unrelated_box(box):
    """The real-world failure case: a short, exact "Box 9" match must beat
    a longer "Box 8" chunk once title-match scoring actually works."""
    rows = [
        {"id": "n_box8", "title": "Box 8", "text": "Box 8\n\n" + ("filler " * 100)},
        {"id": "n_box9", "title": "Box 9", "text": "Box 9\n\nShort real content."},
    ]
    session = _FakeSession(rows)
    items = box._structural_box_content(session, "What is Box 9 about?", 9, "")
    assert items[0]["id"] == "n_box9"


def test_text_match_beats_no_match_when_titles_are_generic(box):
    rows = [
        {"id": "n_untitled", "title": "", "text": "General discussion with no box reference here " * 5},
        {"id": "n_mentions9", "title": "", "text": "Box 9 covers the PowerBI dashboard templates."},
    ]
    session = _FakeSession(rows)
    items = box._structural_box_content(session, "What is Box 9 about?", 9, "")
    assert items[0]["id"] == "n_mentions9"


def test_heading_synthesis_prefers_title_when_it_names_the_box(box):
    rows = [{"id": "n1", "title": "Box 3: Example", "text": "Box 3: Example\n\nSome content about box 3."}]
    session = _FakeSession(rows)
    items = box._structural_box_headings(session, "list all boxes", "")
    assert items[0]["title"] == "Box 3: Example"
