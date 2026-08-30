"""
tests/shared/test_intent_shapes_multilingual_unit.py — question shape without
a language branch.

The 77 shape regexes decide what KIND of question this is -- a table of
contents, a page lookup, a comparison. They were written in English, so
an Arabic question matched none of them and always took the default
shape.

The fix unions each profile's patterns into the English one instead of
selecting between them, which needs no language argument at all: a
pattern written in Arabic script cannot match ASCII, and an English one
cannot match Arabic. Threading a language through all 77 checks would
have been the branch the design forbids, wearing a parameter as a
disguise.

That safety rests on the normalizer being a strict no-op on ASCII, which
is asserted here rather than assumed.

Run with:
    python -m pytest tests/shared/test_intent_shapes_multilingual_unit.py -v
"""
from __future__ import annotations

import pytest

from src.shared.language import intent_alternations, normalize_arabic
from src.unstructured.retrieval.query_intent import (
    is_enumeration_question,
    is_overview_question,
    is_synthesis_question,
    is_toc_question,
)


@pytest.mark.parametrize(
    "query,shape,expected",
    [
        # English must be exactly as it was.
        ("table of contents", is_toc_question, True),
        ("list all the sections", is_toc_question, True),
        ("what is revenue", is_toc_question, False),
        ("list all examples", is_enumeration_question, True),
        ("compare A and B", is_synthesis_question, True),
        ("what is the revenue", is_synthesis_question, False),
        ("what does this document cover", is_overview_question, True),
        # Arabic now reaches the same shapes.
        ("جدول المحتويات", is_toc_question, True),
        ("ما هي فصول هذا المستند؟", is_toc_question, True),
        ("ما هي الفصول في هذا الكتاب؟", is_toc_question, True),
        ("اذكر جميع الأقسام", is_enumeration_question, True),
        ("قارن بين العراق والسعودية", is_synthesis_question, True),
        ("لخص هذا المستند", is_overview_question, True),
        # ...and a specific Arabic question is still not a shape query.
        ("ما هي نسبة الفائدة؟", is_toc_question, False),
    ],
)
def test_shape_detection_across_scripts(query, shape, expected):
    assert shape(query) is expected


def test_the_definite_article_is_optional_in_arabic():
    """Arabic drops it freely in questions; a pattern that requires it
    matches only half the idiom. Found by running the patterns, not by
    reading them."""
    assert is_toc_question("ما هي فصول هذا المستند؟")
    assert is_toc_question("ما هي الفصول في هذا الكتاب؟")


def test_normalization_is_a_no_op_on_ascii():
    """The whole no-branch design depends on this.

    Shape accessors normalize every query before matching. If that
    normalizer touched ASCII, every English question would be silently
    rewritten before its own patterns saw it.
    """
    for text in (
        "What are the table of contents?",
        "page 7 of the PDF",
        "NIST SP 800-207",
        "compare A and B",
        "",
    ):
        assert normalize_arabic(text) == text


def test_a_profile_with_no_patterns_contributes_nothing():
    assert intent_alternations("no_such_shape") == []


def test_english_supplies_no_alternations_of_its_own():
    """English's patterns live in query_intent.py, not in its profile --
    the profile is the seam for languages that are NOT the default."""
    from src.shared.language import ENGLISH

    assert not ENGLISH.intent_patterns
