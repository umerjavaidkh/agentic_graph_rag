"""tests/test_chapter_summary_builder_unit.py — ChapterSummaryBuilder
(ingestion-time chapter rollup summaries).

Covers: gathering a Chapter's descendant Section text via CONTAINS edges
(skipping Page/Region descendants), capping context length, one LLM call
per chapter, writing the result to DKGNode.summary, and never failing
ingestion when the LLM call errors.

Run with:
    python -m pytest tests/test_chapter_summary_builder_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.models import DKGEdge, DKGNode, NodeType, RelType
from src.semantic.chapter_summary import ChapterSummaryBuilder


def _builder() -> ChapterSummaryBuilder:
    builder = ChapterSummaryBuilder.__new__(ChapterSummaryBuilder)
    builder.client = MagicMock()
    return builder


def _chat_response(text: str) -> MagicMock:
    return MagicMock(choices=[MagicMock(message=MagicMock(content=text))])


def _chapter(id_: str, title: str, order: int = 0) -> DKGNode:
    return DKGNode(id=id_, type=NodeType.CHAPTER, title=title, text=title, order=order, depth=1)


def _section(id_: str, title: str, text: str, order: int) -> DKGNode:
    return DKGNode(id=id_, type=NodeType.SECTION, title=title, text=text, order=order, depth=2)


def _page(id_: str, order: int) -> DKGNode:
    return DKGNode(id=id_, type=NodeType.PAGE, title=f"Page {order}", text="page body", order=order, depth=3)


def test_summarizes_chapter_from_its_sections(monkeypatch):
    builder = _builder()
    builder.client.chat_completion.return_value = _chat_response("This chapter covers risk factors.")

    chapter = _chapter("chapter_1", "Item 1A. Risk Factors")
    sections = [
        _section("section_1", "Credit Risk", "Credit risk arises from...", order=1),
        _section("section_2", "Market Risk", "Market risk arises from...", order=2),
    ]
    edges = [
        DKGEdge("chapter_1", "section_1", RelType.CONTAINS),
        DKGEdge("chapter_1", "section_2", RelType.CONTAINS),
    ]
    nodes = builder.build([chapter, *sections], edges)

    assert chapter.summary == "This chapter covers risk factors."
    builder.client.chat_completion.assert_called_once()
    _, kwargs = builder.client.chat_completion.call_args
    prompt = kwargs["messages"][0]["content"]
    assert "Credit Risk" in prompt
    assert "Market Risk" in prompt


def test_skips_page_descendants_but_follows_nested_sections(monkeypatch):
    builder = _builder()
    builder.client.chat_completion.return_value = _chat_response("summary")

    chapter = _chapter("chapter_1", "Item 7. MD&A")
    top_section = _section("section_1", "Overview", "overview text", order=1)
    nested_section = _section("section_1_1", "Liquidity", "liquidity text", order=1)
    page = _page("page_5", order=5)
    edges = [
        DKGEdge("chapter_1", "section_1", RelType.CONTAINS),
        DKGEdge("chapter_1", "page_5", RelType.CONTAINS),  # e.g. a page-axis edge, not a real child section
        DKGEdge("section_1", "section_1_1", RelType.CONTAINS),
    ]
    builder.build([chapter, top_section, nested_section, page], edges)

    _, kwargs = builder.client.chat_completion.call_args
    prompt = kwargs["messages"][0]["content"]
    assert "Overview" in prompt
    assert "Liquidity" in prompt
    assert "page body" not in prompt


def test_chapter_with_no_section_text_is_skipped_without_llm_call():
    builder = _builder()
    chapter = _chapter("chapter_1", "Empty Chapter")
    builder.build([chapter], [])
    builder.client.chat_completion.assert_not_called()
    assert chapter.summary is None


def test_llm_failure_is_swallowed_and_does_not_set_summary():
    builder = _builder()
    builder.client.chat_completion.side_effect = RuntimeError("rate limited")
    chapter = _chapter("chapter_1", "Item 1. Business")
    section = _section("section_1", "Overview", "overview text", order=1)
    edges = [DKGEdge("chapter_1", "section_1", RelType.CONTAINS)]

    builder.build([chapter, section], edges)  # must not raise

    assert chapter.summary is None


def test_no_client_returns_nodes_unchanged():
    builder = ChapterSummaryBuilder.__new__(ChapterSummaryBuilder)
    builder.client = None
    chapter = _chapter("chapter_1", "Item 1. Business")
    result = builder.build([chapter], [])
    assert result == [chapter]
    assert chapter.summary is None


def test_context_length_is_capped(monkeypatch):
    import src.semantic.chapter_summary as mod

    monkeypatch.setattr(mod, "CHAPTER_SUMMARY_MAX_CONTEXT_CHARS", 100)
    monkeypatch.setattr(mod, "CHAPTER_SUMMARY_SECTION_EXCERPT_CHARS", 50)

    builder = _builder()
    builder.client.chat_completion.return_value = _chat_response("summary")

    chapter = _chapter("chapter_1", "Item 1. Business")
    sections = [
        _section(f"section_{i}", f"Section {i}", "x" * 60, order=i) for i in range(10)
    ]
    edges = [DKGEdge("chapter_1", s.id, RelType.CONTAINS) for s in sections]
    builder.build([chapter, *sections], edges)

    _, kwargs = builder.client.chat_completion.call_args
    prompt = kwargs["messages"][0]["content"]
    # Only a couple of sections should fit under the 100-char cap.
    assert prompt.count("Section ") <= 3
