"""
tests/test_firmwide_financial_metric_intent_unit.py

Guards is_firmwide_financial_metric_question — the detector that gates the
firmwide-summary retrieval boost. Real bug it protects: a plain "What were
net earnings for 2025?" answered from a *business segment's* net earnings
(Global Banking & Markets, $13.81B) instead of the firm total ($17.18B),
because the long Executive Overview section loses vector-cosine ranking to
short segment tables. The detector must fire for firmwide metric questions
and must NOT fire once the question names a segment/region (there the
segment figure is exactly what's wanted).

Run with:
    python -m pytest tests/test_firmwide_financial_metric_intent_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.retrieval.unstructured.query_intent import (
    is_firmwide_financial_metric_question,
)


@pytest.mark.parametrize(
    "query",
    [
        "What were Goldman Sachs net earnings for 2025, and how did that compare to 2024?",
        "What was net income in 2024?",
        "What was diluted EPS and ROE for 2025?",
        "What was book value per common share?",
        "What were total net revenues for the year?",
        "What was the return on average common equity?",
        "What were total assets at year end?",
        "What was the effective income tax rate?",
    ],
)
def test_fires_for_firmwide_metric_questions(query):
    assert is_firmwide_financial_metric_question(query) is True


@pytest.mark.parametrize(
    "query",
    [
        # Explicitly segment/region-scoped — the segment figure is the answer.
        "What were net earnings for Global Banking & Markets in 2025?",
        "What was net income by segment?",
        "Show net revenues for Asset & Wealth Management.",
        "What were net earnings by geographic region?",
        "What net earnings did each operating segment report?",
        # Not a financial-metric question at all.
        "What does the whistleblowing policy say?",
        "Summarize the risk factors section.",
        "How many products do we have?",
    ],
)
def test_does_not_fire_for_segment_or_nonfinancial(query):
    assert is_firmwide_financial_metric_question(query) is False
