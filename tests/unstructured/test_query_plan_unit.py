"""tests/unstructured/test_query_plan_unit.py — shape routing.

Misrouting is the dominant cause of "works on some questions, fails on
others", so these pin the two rules that are not negotiable: an exhaustive
question never gets a top-k cut, and a graph address never goes to the
vector index.
"""
from __future__ import annotations

import pytest

from src.unstructured.retrieval.query_plan import Shape, classify


@pytest.mark.parametrize("q", [
    "What is the table of contents of Go.Data annual report?",
    "show me the contents page",
])
def test_table_of_contents_is_exhaustive_and_in_document_order(q):
    """Measured failure: 8 of 30 headings returned, out of order, confidently."""
    p = classify(q)
    assert p.shape is Shape.STRUCTURAL
    assert p.exhaustive and p.document_order
    assert not p.use_vectors


@pytest.mark.parametrize("q,addr", [
    ("What is Box 9 about in this report?", "Box 9"),
    ("What does Figure 1 show in this report?", "Figure 1"),
    ("What is discussed on page 1 of the introduction?", "page 1"),
    ("how does clause 8 differ from the earlier text", "clause 8"),
    ("what does section 4.2 say", "section 4.2"),
])
def test_an_address_never_goes_to_the_vector_index(q, addr):
    """"Section 4.2" is a graph address; its nearest neighbours are other addresses.

    Sharper on this corpus than in general: Page nodes are 1.0% embedded and
    Region nodes 0%, so vectors cannot see the units these questions name.
    """
    p = classify(q)
    assert p.shape is Shape.STRUCTURAL
    assert p.address == addr
    assert not p.use_vectors


@pytest.mark.parametrize("q", [
    "list every clause mentioning indemnity",
    "list all sections about risk",
    "enumerate the worked examples",
])
def test_exhaustive_questions_are_never_truncated(q):
    p = classify(q)
    assert p.exhaustive
    assert p.limit > 100


def test_content_filtered_enumeration_keeps_the_vector_channel():
    """"all sections that mention risk" is an enumeration, not a hierarchy request.

    The shared is_toc_question fires on "all sections", which would route it
    structural and switch vectors off -- on a query whose point is matching
    "mention risk" semantically.
    """
    p = classify("show me all sections that mention risk")
    assert p.exhaustive
    assert p.use_vectors


def test_thematic_gets_breadth():
    p = classify("What does this Go.Data annual report discuss overall?")
    assert p.shape is Shape.THEMATIC
    assert p.limit >= 16


def test_plain_lookup_stays_on_the_default_path():
    p = classify("what is the notice period")
    assert p.shape is Shape.FACTOID
    assert p.use_vectors and not p.exhaustive
