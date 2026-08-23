"""Model output is untrusted input, and it reaches a dict.update().

`dict.update(x)` accepts any iterable of pairs, so a string arrives as a
sequence of characters and raises "dictionary update sequence element #0
has length 1; 2 is required". That killed the Neo4j load for every document
in a 104-document run -- after parsing, embedding and summarising had all
succeeded -- and each job still reported success.
"""
import pytest

from src.unstructured.semantic.axis2 import _as_type_map


@pytest.mark.parametrize("bad", ["ORG", "", ["ORG"], [("a", "b")], 42, None, {"a": 1}, {1: "x"}])
def test_anything_that_is_not_a_string_map_becomes_empty(bad):
    """Empty is safe: the node keeps its entities, just untyped."""
    assert _as_type_map(bad) == {}


def test_a_string_does_not_explode_into_characters():
    """The exact failure. dict().update('ORG') raises; this must not."""
    out = _as_type_map("ORG")
    assert out == {}
    d = {}
    d.update(out)          # the operation that used to raise
    assert d == {}


def test_a_good_map_is_preserved():
    assert _as_type_map({"pfizer": "ORG", "2025": "DATE"}) == {"pfizer": "ORG", "2025": "DATE"}


def test_mixed_entries_keep_only_the_valid_ones():
    """A partly-malformed response should cost only the bad entries."""
    assert _as_type_map({"pfizer": "ORG", "bad": 7, 9: "ORG"}) == {"pfizer": "ORG"}
