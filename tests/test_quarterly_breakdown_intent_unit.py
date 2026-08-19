"""
tests/test_quarterly_breakdown_intent_unit.py

Guards is_quarterly_breakdown_question — the detector that gates the
quarterly-financial-data retrieval boost. Real bug it protects: "How did
JPMorgan's quarterly net income break down across the year (Q1-Q4)?"
answered "This document does not cover the quarterly breakdown..." even
though the filing has an exact "Selected Quarterly Financial Data
(Unaudited)" table (Q1 $5,520M / Q2 $6,200M / Q3 $6,286M / Q4 $6,727M) —
retrieval never surfaced it because that table's wrapping section title is
a generic "Supplementary information", so vector-cosine ranking buried it
under narrative annual-summary sections that merely mention the same
metric name once. The detector must fire for quarter + metric questions
and must NOT fire on a bare quarter mention (no metric) or a bare metric
mention (no quarter).

Run with:
    python -m pytest tests/test_quarterly_breakdown_intent_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


from src.retrieval.unstructured.query_intent import is_quarterly_breakdown_question


@pytest.mark.parametrize(
    "query",
    [
        "How did JPMorgan's quarterly net income break down across the year (Q1-Q4)?",
        "What was net income for each quarter of 2016?",
        "Show quarterly total revenue by quarter.",
        "What was Q1-Q4 diluted EPS?",
        "What were earnings in the fourth quarter versus the first quarter?",
        "quarterly earnings breakdown",
    ],
)
def test_fires_for_quarterly_metric_questions(query):
    assert is_quarterly_breakdown_question(query) is True


@pytest.mark.parametrize(
    "query",
    [
        # Quarter mentioned, but no financial metric -- not this question shape.
        "What happened in the fourth quarter regarding hiring?",
        "What is the exact filing date of this Form 10-Q?",
        # Metric mentioned, but no quarter -- the firmwide-summary detector's job.
        "What was net income in 2024?",
        "What was total revenue for the year?",
        # Neither.
        "What does the whistleblowing policy say?",
    ],
)
def test_does_not_fire_without_both_quarter_and_metric(query):
    assert is_quarterly_breakdown_question(query) is False
