"""
tests/test_toc_intent_structural_phrasing_unit.py — TOC-intent classification
for structural-outline phrasing that doesn't say "contents"/"toc" literally.

Regression: "What sections does this document have?" and "List every
heading in order" fell through to the relevance-ranked hybrid strategy
instead of TocStrategy (the only strategy that walks the graph in actual
document order), producing incomplete/out-of-order answers even though
the underlying graph structure was correct. Root cause was two-fold:
is_toc_question()'s regex only matched literal "contents"/"toc" wording,
and include_in_outline_fallback() only included depth<=1 (Chapter-tier)
headings by default -- a flat document with no chapter tier (everything
at depth=2) had every heading silently dropped from the outline.

Run with:
    python -m pytest tests/test_toc_intent_structural_phrasing_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.retrieval.unstructured.query_intent import is_toc_question
from src.retrieval.unstructured.toc_retrieval import include_in_outline_fallback


@pytest.mark.parametrize("query", [
    "What sections does this document have?",
    "what sections are in this document",
    "List every heading in order.",
    "list all sections",
    "show me the headings",
    "what are the sections",
    "table of contents",
    "list all the contents",
])
def test_structural_outline_phrasing_is_toc_question(query):
    assert is_toc_question(query) is True


@pytest.mark.parametrize("query", [
    "what is the standard error formula",
    "what does section 3 say about revenue",
    "summarize the sections on liquidity",
])
def test_content_questions_mentioning_sections_are_not_toc_questions(query):
    assert is_toc_question(query) is False


def test_depth_2_section_included_in_outline_fallback():
    """A document with no chapter tier (everything a depth-2 Section) must
    still get its headings into the outline, not silently dropped."""
    assert include_in_outline_fallback("Worked Example 1", depth=2, label="Section") is True
    assert include_in_outline_fallback("Sample Mean Basics", depth=2, label="Section") is True


def test_chapter_tier_still_included():
    assert include_in_outline_fallback("Introduction", depth=1, label="Chapter") is True


def test_deep_unnumbered_subsection_still_excluded():
    """depth>=3, not numbered, not uppercase -- still filtered, unchanged
    behavior for genuinely nested subsections."""
    assert include_in_outline_fallback("a minor point", depth=3, label="Section") is False


def test_numbered_deep_subsection_still_included():
    assert include_in_outline_fallback("4.5.1 Detail", depth=4, label="Section") is True
