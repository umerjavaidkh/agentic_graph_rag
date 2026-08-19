"""Structured query clarification when a metric maps to several properties."""
from __future__ import annotations

import re
from typing import Any, Iterable, Optional

from ....shared.conversation.clarification import format_clarification_answer

# Words that mean "aggregate something" rather than naming the something.
_AGGREGATE_WORDS = frozenset({
    "average", "avg", "mean", "total", "sum", "median", "typical",
})

# Stopwords for picking the measured noun out of the question. Deliberately
# grammatical only -- no domain nouns, since the domain is whatever the
# connected graph happens to be. Words like "value", "amount" and "count" are
# likewise NOT listed:
# they read as filler but are extremely common column names (a Payment.value
# column made "average payment value" match nothing at all when they were),
# and excluding a real property name is the one mistake this list must not
# make.
_NON_METRIC_WORDS = frozenset({
    "what", "whats", "which", "how", "much", "many", "the", "a", "an", "of",
    "is", "are", "was", "were", "in", "on", "for", "by", "per", "across",
    "all", "each", "every", "and", "or", "to", "from", "with", "that", "this",
    "me", "give", "show", "tell", "database", "data", "graph", "dataset",
}) | _AGGREGATE_WORDS

_WORD_RE = re.compile(r"[a-z][a-z0-9]*")
# One option per candidate is unhelpful past a handful; asking someone to
# choose between ten near-identical columns is worse than picking one.
_MAX_OPTIONS = 4


def _tokens(text: str) -> list[str]:
    return _WORD_RE.findall((text or "").lower())


def _property_tokens(prop: str) -> set[str]:
    """`unit_price` / `unitPrice` / `UnitPrice` -> {"unit", "price"}."""
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", prop or "")
    return set(_WORD_RE.findall(spaced.lower().replace("_", " ")))


def _singular(word: str) -> str:
    """Crude depluralisation so "order items" matches an OrderItem label."""
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _query_words(query: str) -> set[str]:
    return {_singular(w) for w in _tokens(query)} - _NON_METRIC_WORDS


def _specificity(candidate: tuple[str, str], words: set[str]) -> int:
    """How much of the question this (label, property) pair accounts for.

    Counts label words as well as property words, which is what separates
    OrderItem.freight from Payment.value for a question that says "order
    items": both match one property word, only one matches the label too.
    """
    label, prop = candidate
    matched = {_singular(t) for t in _property_tokens(prop)} & words
    matched |= {_singular(t) for t in _property_tokens(label)} & words
    return len(matched)


def numeric_metric_candidates(
    query: str, numeric_properties: Iterable[tuple[str, str]]
) -> list[tuple[str, str]]:
    """Numeric (label, property) pairs the question's metric noun could mean.

    Matching is on the words in the property NAME against the words in the
    question, so it works for whatever the graph happens to hold. A property
    is a candidate only when the question actually names it -- so a question
    about "price" does not offer up every numeric column in the schema.
    """
    words = _query_words(query)
    if not words:
        return []
    hits = []
    for label, prop in numeric_properties:
        if {_singular(t) for t in _property_tokens(prop)} & words:
            hits.append((label, prop))
    return sorted(set(hits))


def needs_clarification(
    query: str, numeric_properties: Optional[Iterable[tuple[str, str]]] = None
) -> Optional[dict[str, Any]]:
    """Ask which property to measure, but only when the graph really is ambiguous.

    This used to be a keyword rule with three fixed options describing a
    schema that is not loaded any more -- it offered "unitPrice x quantity x
    (1 - discount)" against a graph with no such fields, so "average price of
    an order item" could not be answered at all: every option was
    uncomputable. Candidates now come from the live schema, so the question is
    only asked when two or more real numeric properties match what was asked,
    and every option offered can actually be computed.
    """
    if not (query or "").strip() or not numeric_properties:
        return None
    if not (set(_tokens(query)) & _AGGREGATE_WORDS):
        return None

    candidates = numeric_metric_candidates(query, numeric_properties)
    # One match is not ambiguity, it is the answer. Nothing matching means the
    # question names no property, which the generator handles better than a
    # menu would.
    if len(candidates) < 2:
        return None

    # Two matches are not ambiguity either if one of them explains more of the
    # question. "Total freight value across all order items" matches
    # OrderItem.freight AND Payment.value on a bare property-word overlap, but
    # only the first is also on the label the question names -- asking the user
    # to choose there is a worse answer than just answering. Ambiguity is a TIE
    # at the top, not merely more than one candidate.
    words = _query_words(query)
    ranked = sorted(candidates, key=lambda c: _specificity(c, words), reverse=True)
    best = _specificity(ranked[0], words)
    tied = [c for c in ranked if _specificity(c, words) == best]
    if len(tied) < 2:
        return None
    candidates = tied[:_MAX_OPTIONS]

    options = [
        {
            "id": f"{label}.{prop}",
            "label": f"{label}.{prop}",
            "detail": f"Aggregate {prop} on {label}",
            "aliases": [f"{label}.{prop}".lower(), prop.lower(), f"{label} {prop}".lower()],
        }
        for label, prop in candidates
    ]
    prompt = (
        "That could be measured from more than one field in this data. "
        "Which one do you mean?\n\n"
        f"Reply with 1-{len(options)}, or the field name."
    )
    return {
        "query": query,
        "strategy": "clarification",
        "mode": "needs_clarification",
        "original_question": query,
        "clarification_kind": "structured_metric_choice",
        "clarification_options": options,
        "chunks": [
            {
                "id": "clarification",
                "title": "Clarification",
                "text": format_clarification_answer(prompt, options),
                "score": 1.0,
                "related": [],
            }
        ],
        "total_available": 1,
    }
