"""tests/test_query_intent_overview_unit.py — is_overview_question().

Covers the detector added to gate the chapter-summary rollup feature:
deliberately broader than is_synthesis_question (which is tuned for
compare/contrast/structural-map phrasing) so it catches the much more
common "what does this document/chapter discuss" phrasing that
_SYNTHESIS_RE doesn't match at all — confirmed live: "What does this
JPMorgan 10-K filing discuss overall?" returned zero chapter-summary
candidates before this detector existed, because is_synthesis_question()
alone never fired for it.

Run with:
    python -m pytest tests/test_query_intent_overview_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.retrieval.unstructured.query_intent import is_overview_question, is_synthesis_question


@pytest.mark.parametrize("query", [
    "What does this JPMorgan 10-K filing discuss overall?",
    "What is this document about?",
    "What does the filing discuss?",
    "Summarize this document for me.",
    "Give me an overview of this filing.",
    "What does this chapter cover?",
    "What is this 10-Q about?",
])
def test_matches_overview_phrasing(query):
    assert is_overview_question(query)


@pytest.mark.parametrize("query", [
    "What is JPMorgan's net income for Q4 2016?",
    "List all board members mentioned in the filing.",
    "What page discusses risk factors?",
    "How did net income change from 2015 to 2016?",
])
def test_does_not_match_specific_fact_questions(query):
    assert not is_overview_question(query)


def test_the_live_regression_case_is_covered_where_synthesis_alone_was_not():
    query = "What does this JPMorgan 10-K filing discuss overall?"
    assert is_overview_question(query)
    assert not is_synthesis_question(query)
