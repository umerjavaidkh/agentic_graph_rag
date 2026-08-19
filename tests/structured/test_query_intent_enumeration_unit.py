"""
tests/test_query_intent_enumeration_unit.py — is_enumeration_question's
detection of natural "which X apply/discuss/cover Y" phrasing, not just
the original literal "list all/enumerate/name all/distinct" set.

Regression: found live -- "which worked examples apply it" (asking for
ALL matching worked-example sections, not the single best match) fell
through the original literal-phrase-only detector entirely. Since
full_hybrid.py's retrieve() only bumps fetch_limit (8 -> 18) when
enumeration is detected, the miss meant only 1 of 3 structurally-parallel
"Worked Example N" sections survived ranking into the final context, even
though the graph itself connects all three via SHARES_ENTITY.

Run with:
    python -m pytest tests/test_query_intent_enumeration_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path


from src.unstructured.retrieval.query_intent import is_enumeration_question


def test_original_literal_phrases_still_match():
    assert is_enumeration_question("List all the risk factors mentioned")
    assert is_enumeration_question("Please enumerate the board members")
    assert is_enumeration_question("Name all subsidiaries of the company")
    assert is_enumeration_question("What are the distinct product categories?")


def test_which_plural_noun_now_matches():
    # The motivating regression case, and its general shape.
    assert is_enumeration_question("What is the formula for standard error, and which worked examples apply it?")
    assert is_enumeration_question("Which sections discuss revenue recognition?")
    assert is_enumeration_question("Which pages mention the merger?")
    assert is_enumeration_question("Which items on the balance sheet increased?")


def test_what_are_all_matches():
    assert is_enumeration_question("What are all the assumptions used in this model?")


def test_all_the_plural_noun_matches():
    assert is_enumeration_question("Summarize all the examples of risk mitigation")
    assert is_enumeration_question("Cover all the sections about liquidity")


def test_singular_which_noun_does_not_match():
    # A genuine single-answer lookup -- must NOT be treated as enumeration,
    # or every "which X" question would over-fetch unnecessarily.
    assert not is_enumeration_question("Which section discusses the merger?")
    assert not is_enumeration_question("Which page has the signature?")


def test_unrelated_questions_do_not_match():
    assert not is_enumeration_question("What is the sample mean?")
    assert not is_enumeration_question("How does the standard error formula work?")
    assert not is_enumeration_question("Compare the sample mean and standard error")


def test_empty_and_none_query_do_not_match():
    assert not is_enumeration_question("")
    assert not is_enumeration_question(None)
