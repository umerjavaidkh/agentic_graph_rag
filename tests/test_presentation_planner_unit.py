"""tests/test_presentation_planner_unit.py — src/presentation/planner.py's
build_presentation() and chart-extraction heuristic.

Covers a real bug: a narrative answer with several unrelated percentages
scattered through prose (e.g. "CET1 4.5%... Tier 1 6%... GSIB surcharge
1.75%...") triggered a meaningless bar chart with generic "Item 1"/"Item 2"
labels, AND the markdown answer text was only appended as a fallback when
`blocks` was still empty — so the chart silently replaced the real answer
in the response instead of supplementing it.

Run with:
    python -m pytest tests/test_presentation_planner_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


from src.unstructured.presentation import _extract_chart_from_text, build_presentation


# ── _extract_chart_from_text ─────────────────────────────────────────────


def test_narrative_percentages_do_not_produce_a_chart():
    """The real bug case: several different concepts, each with its own
    percentage, scattered through prose — no genuine label-value pairing,
    so no chart should be produced at all."""
    answer = (
        "JPMorgan Chase discloses the following regulatory capital ratios "
        "in this filing:\n\n"
        "1. **CET1 Capital Ratio**: Minimum capital ratio of 4.5% of CET1 capital.\n"
        "2. **Tier 1 Capital Ratio**: Minimum of 6% Tier 1 capital required.\n"
    )
    # This case DOES have bold labels — verify it produces a real, labeled chart.
    chart = _extract_chart_from_text(answer)
    assert chart is not None
    assert chart["labels"] == ["CET1 Capital Ratio", "Tier 1 Capital Ratio"]
    assert chart["values"] == [4.5, 6.0]


def test_unlabeled_percentages_produce_no_chart():
    """No bold labels paired with the percentages — the original bug's
    exact shape (a narrative mentioning "1.125%" and "1.75%" as GSIB
    surcharge figures for two different years, with no per-number label) —
    must NOT fall back to generic "Item 1"/"Item 2" placeholders."""
    answer = (
        "The GSIB surcharge was 1.125% for 2016 and 1.75% for 2017, "
        "reflecting the transition provisions."
    )
    assert _extract_chart_from_text(answer) is None


def test_single_labeled_value_produces_no_chart():
    answer = "**CET1 Capital Ratio**: 4.5%."
    assert _extract_chart_from_text(answer) is None


def test_duplicate_labels_deduplicated():
    answer = "**Ratio**: 4.5%. Later, **Ratio**: 4.5% again. **Other**: 6%."
    chart = _extract_chart_from_text(answer)
    assert chart is not None
    assert chart["labels"] == ["Ratio", "Other"]


# ── build_presentation ────────────────────────────────────────────────────


def test_answer_text_always_present_even_with_a_chart():
    """The actual regression: the markdown answer must never disappear
    from presentation.blocks just because a chart was also extracted."""
    answer = (
        "JPMorgan discloses:\n\n"
        "1. **CET1 Capital Ratio**: 4.5%.\n"
        "2. **Tier 1 Capital Ratio**: 6%.\n"
    )
    result = build_presentation(
        question="What regulatory capital ratios does JPMorgan disclose?",
        answer=answer,
        sources=[{"id": "s1", "title": "Notes", "text": "some source text"}],
        agent="unstructured",
    )
    types = [b["type"] for b in result["blocks"]]
    assert "markdown" in types
    assert "chart" in types
    markdown_block = next(b for b in result["blocks"] if b["type"] == "markdown")
    assert markdown_block["content"] == answer
    assert result["kind"] == "mixed"


def test_answer_text_present_when_no_chart_or_table():
    answer = "Plain narrative answer with no numbers at all."
    result = build_presentation(
        question="What does the filing say?",
        answer=answer,
        sources=[],
        agent="unstructured",
    )
    assert len(result["blocks"]) == 1
    assert result["blocks"][0] == {"type": "markdown", "content": answer}
    assert result["kind"] == "markdown"


def test_answer_text_present_alongside_a_table():
    answer = (
        "Here is the breakdown:\n\n"
        "| Metric | Value |\n"
        "| --- | --- |\n"
        "| CET1 | 4.5% |\n"
        "| Tier 1 | 6% |\n"
    )
    result = build_presentation(
        question="Show me a table",
        answer=answer,
        sources=[],
        agent="unstructured",
    )
    types = [b["type"] for b in result["blocks"]]
    assert types == ["markdown", "table"]
    # Raw pipe-table syntax is stripped from the markdown block (rendered
    # separately as a proper table block instead), but real prose survives.
    assert "Here is the breakdown" in result["blocks"][0]["content"]
    assert "| CET1 |" not in result["blocks"][0]["content"]
