"""Detect ranking shape for Northwind / structured analytics queries."""
from __future__ import annotations

import re

_TOP_N = re.compile(r"\btop\s+(\d+)\b", re.I)
_SINGULAR_BEST = re.compile(
    r"\b(?:the\s+)?(?:one|1)\s+best\b"
    r"|\b(?:the\s+)?best\s+(?:product|seller|item)\b"
    r"|\bwhich\s+product\b.+\b(?:most|highest|best)\b"
    r"|\bwhat\s+product\b.+\b(?:most|highest|best)\b"
    r"|\b(?:highest|most)\s+(?:sales|revenue|profit)\b.+\bproduct\b"
    r"|\bproduct\b.+\b(?:highest|most)\s+(?:sales|revenue|profit)\b",
    re.I,
)


def is_singular_best_query(question: str) -> bool:
    """User wants one winner (best product), not a top-N leaderboard."""
    if _TOP_N.search(question):
        return False
    return bool(_SINGULAR_BEST.search(question))


def analytics_result_limit(question: str, default: int = 5) -> int:
    if is_singular_best_query(question):
        return 1
    m = _TOP_N.search(question)
    if m:
        return max(1, min(25, int(m.group(1))))
    return default
