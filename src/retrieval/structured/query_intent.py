"""Detect ranking shape and multistep-plan cues for structured analytics queries."""
from __future__ import annotations

import re

_TOP_N = re.compile(r"\btop\s+(\d+)\b", re.I)

# Nested / per-group analytics that are painful in one Cypher query → LLM multistep planner.
_TOP_PER_GROUP = re.compile(
    r"\btop\s+\d+\b.{0,120}\bper\s+(?:each\s+)?\w+"
    r"|\bper\s+(?:each\s+)?(?:country|countries|category|categories|region|regions|"
    r"customer|customers|group|groups|city|cities|supplier|suppliers|segment|segments)\b"
    r".{0,120}\btop\s+\d+\b"
    r"|\btop\s+\d+\b.{0,120}\b(?:for|within)\s+each\b"
    r"|\b(?:for|within)\s+each\b.{0,120}\btop\s+\d+\b",
    re.I | re.S,
)
_TOP_AMONG_TOP = re.compile(
    r"\b(?:among|amongst|from)\s+(?:the\s+)?top\s+\d+\b"
    r"|\btop\s+\d+\b.{0,80}\b(?:among|amongst|from)\s+(?:the\s+)?top\s+\d+\b"
    r"|\bthen\b.{0,80}\btop\s+\d+\b",
    re.I | re.S,
)
_SEQUENTIAL_TOPS = re.compile(
    r"\b(?:first|then|next|after that)\b.{0,100}\btop\s+\d+\b",
    re.I | re.S,
)
_SINGULAR_BEST = re.compile(
    r"\b(?:the\s+)?(?:one|1)\s+best\b"
    r"|\b(?:the\s+)?best\s+(?:product|seller|item)\b"
    r"|\bwhich\s+product\b.+\b(?:most|highest|best)\b"
    r"|\bwhat\s+product\b.+\b(?:most|highest|best)\b"
    r"|\b(?:highest|most)\s+(?:sales|revenue|profit)\b.+\bproduct\b"
    r"|\bproduct\b.+\b(?:highest|most)\s+(?:sales|revenue|profit)\b",
    re.I,
)

# Negation / exclusion filters on graph entities (category, product, supplier, …).
_NEGATION_ANALYTICS = re.compile(
    r"\b(?:never|not\s+in|without|excluding|except)\b.{0,100}\b(?:category|categories|"
    r"product|products|supplier|suppliers|customer|customers|order|orders|region|regions)\b"
    r"|\b(?:category|categories|product|products|supplier|suppliers)\b.{0,100}\b"
    r"(?:never|not\s+in|without|excluding|except)\b",
    re.I | re.S,
)

# Year (or date) combined with exclusion — common multi-constraint analytics shape.
_YEAR_WITH_NEGATION = re.compile(
    r"\b(?:19|20)\d{2}\b.+\b(?:never|not\s+in|without|excluding)\b"
    r"|\b(?:never|not\s+in|without|excluding)\b.+\b(?:19|20)\d{2}\b",
    re.I | re.S,
)

# Superlative ranking plus an extra filter clause (not already caught by top-N-per-group).
_RANK_WITH_EXTRA_FILTER = re.compile(
    r"\b(?:most|highest|greatest|largest|fewest|lowest)\b.+\b(?:but|that|which|who|where|"
    r"never|not\s+in|without|excluding|in\s+\d{4}|category|categories|supplier)\b",
    re.I | re.S,
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


def estimate_structured_synthesis_max_tokens(
    question: str,
    *,
    chunk_count: int = 0,
    default_max: int,
    long_max: int,
) -> int:
    """Long budget for nested top-N / per-group analytics and multistep result sets."""
    if likely_needs_multistep_plan(question) or chunk_count > 3:
        return long_max
    return default_max


def likely_needs_multistep_plan(question: str) -> bool:
    """
    Fast regex gate for the multistep LLM planner.

    Single-step questions (counts, one leaderboard, simple filters) should skip
    the extra planning round trip; nested top-N-per-group patterns should not.
    """
    q = (question or "").strip()
    if not q:
        return False
    if _TOP_PER_GROUP.search(q) or _TOP_AMONG_TOP.search(q) or _SEQUENTIAL_TOPS.search(q):
        return True
    if len(_TOP_N.findall(q)) >= 2:
        return True
    if _NEGATION_ANALYTICS.search(q) or _YEAR_WITH_NEGATION.search(q):
        return True
    if _RANK_WITH_EXTRA_FILTER.search(q):
        return True
    return False
