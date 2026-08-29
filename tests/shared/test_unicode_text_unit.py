"""
tests/shared/test_unicode_text_unit.py — script-independent tokenization.

The point of this module is a *negative* property: widening the ASCII
letter classes must not move English. So the first test here is not a
list of examples, it is a differential run of every pattern that was
replaced against the replacement, over generated ASCII text. If a class
were widened one character too far, English behaviour would drift
somewhere nobody thought to write an example for, and only this catches
it.

Run with:
    python -m pytest tests/shared/test_unicode_text_unit.py -v
"""
from __future__ import annotations

import random
import re
import string

from src.shared.unicode_text import (
    fold,
    letters,
    opens_sentence,
    sentence_split,
    squash_punctuation,
    words,
)

# Each entry is (name, the ASCII expression that used to be in the tree,
# the call that replaced it). The left-hand sides are copied verbatim from
# the call sites this module absorbed.
_REPLACEMENTS = [
    (
        "term_stats corpus vocabulary",
        lambda s: re.findall(r"[a-z][a-z0-9-]{2,}", s.lower()),
        lambda s: words(s, min_length=3, hyphens=True),
    ),
    (
        "clarification query tokens",
        lambda s: re.findall(r"[a-z][a-z0-9]*", s.lower()),
        lambda s: words(s),
    ),
    (
        "graph citation content words",
        lambda s: re.findall(r"[a-z0-9]+", s.lower()),
        lambda s: words(s, numeric=True),
    ),
    (
        "visual_retrieval phrase words",
        lambda s: re.findall(r"[a-z0-9]{3,}", s.lower()),
        lambda s: words(s, min_length=3, numeric=True),
    ),
    (
        "vector_first_hybrid alpha tokens",
        lambda s: re.findall(r"[a-z]+", s.lower()),
        lambda s: letters(s),
    ),
    (
        "ranking hyphenated query terms",
        lambda s: re.findall(r"[a-z]+(?:-[a-z]+)+", s.lower()),
        lambda s: [w for w in letters(s, hyphens=True) if "-" in w],
    ),
    (
        "document_resolver title squash",
        lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip(),
        squash_punctuation,
    ),
    (
        "graph sentence split",
        lambda s: re.split(r"(?<![0-9].)(?<=[.!?])\s+(?=[A-Z])", s),
        sentence_split,
    ),
]

# Punctuation, digits, both cases, and the whitespace a PDF actually
# emits. Spaces are weighted up so sentence boundaries occur often enough
# to exercise the split.
_ASCII = string.ascii_letters + string.digits + " _-.:!?'()|/\t\n" + "   "

_FIXED = [
    "",
    " ",
    "Item 7A. Management's Discussion",
    "unit_price",
    "unitPrice",
    "case-control study",
    "See 2.4. Interoperability now.",
    "U.S. Downstream",
    "IRS Publication 225",
    "One thing. Two things.",
    "ends with a hyphen- and more",
    "3.5. 4. numbers everywhere 10K",
]


def _ascii_corpus(n: int = 20000) -> list[str]:
    rnd = random.Random(20260829)
    generated = [
        "".join(rnd.choice(_ASCII) for _ in range(rnd.randint(0, 40)))
        for _ in range(n)
    ]
    return _FIXED + generated


def test_every_replaced_ascii_pattern_is_reproduced_exactly():
    """The English suite is unchanged by construction, and this is the proof.

    Widening `[a-z]` to "any letter" is only safe if the new class is a
    strict superset that adds nothing inside ASCII. Underscore is the
    trap: `\\w` contains it and `[a-z0-9]` does not, so a naive widening
    would silently join `unit_price` into one token everywhere in the
    corpus.
    """
    corpus = _ascii_corpus()
    for name, ascii_impl, unicode_impl in _REPLACEMENTS:
        mismatches = [s for s in corpus if ascii_impl(s) != unicode_impl(s)]
        assert not mismatches, (
            f"{name} moved on ASCII input, e.g. {mismatches[0]!r}: "
            f"{ascii_impl(mismatches[0])!r} != {unicode_impl(mismatches[0])!r}"
        )


def test_underscore_still_separates_words():
    """Guarded explicitly because `\\w` would have swallowed it."""
    assert words("unit_price") == ["unit", "price"]


def test_an_accent_no_longer_splits_a_word():
    """The silent half of the bug: fragments that look like real tokens.

    `Prévisions` used to tokenize to `['visions']` -- a genuine English
    word, produced out of French text, which then counted as corpus
    vocabulary and matched unrelated English prose.
    """
    assert words("Résultats Prévisions") == ["résultats", "prévisions"]


def test_arabic_text_produces_tokens_at_all():
    assert words("النتائج المالية") == ["النتائج", "المالية"]


def test_ascii_words_are_still_lowercased():
    assert words("Revenue Growth") == ["revenue", "growth"]


def test_letters_stop_at_a_digit_but_words_do_not():
    """The two are separate functions because the callers disagree.

    Prose matching wants `IRS225` to be neither `irs` nor a match for it;
    quantity matching wants the digits kept.
    """
    assert letters("IRS225") == ["irs"]
    assert words("IRS225") == ["irs225"]


def test_min_length_drops_short_tokens():
    assert words("a bc def ghij", min_length=3) == ["def", "ghij"]


def test_hyphenated_terms_survive_as_one_token():
    assert [w for w in letters("case-control study", hyphens=True) if "-" in w] == [
        "case-control"
    ]


def test_sentence_split_still_refuses_to_break_a_section_number():
    """The regression this guard was added for.

    Splitting "2.4. Interoperability" attributed the tail of the sentence
    to a different page, so a claim cited a page that did not support it.
    """
    assert sentence_split("See 2.4. Interoperability here. Next one.") == [
        "See 2.4. Interoperability here.",
        "Next one.",
    ]


def test_sentence_split_works_in_a_script_with_no_case():
    """Arabic could never satisfy the old `(?=[A-Z])` lookahead.

    An Arabic answer therefore split into exactly one sentence, and every
    claim in it came back with no citation.
    """
    assert sentence_split("هذا نص. وهذا نص آخر.") == ["هذا نص.", "وهذا نص آخر."]


def test_sentence_split_does_not_break_on_a_lowercase_continuation():
    assert sentence_split("e.g. this continues") == ["e.g. this continues"]


def test_opens_sentence_across_scripts():
    assert opens_sentence("A")
    assert opens_sentence("École")  # uppercase outside ASCII
    assert opens_sentence("النتائج")  # caseless: treated as a sentence start
    assert not opens_sentence("a")
    assert not opens_sentence("é")
    assert not opens_sentence("9")
    assert not opens_sentence("")


def test_fold_normalizes_a_ligature():
    """Why `fold` is applied to queries and not yet to stored text.

    NFKC rewrites English content -- this very case -- so applying it at
    ingest is a re-parse of the corpus and a re-measurement, not a
    tokenization fix.
    """
    assert fold("ﬁle") == "file"
    assert fold("") == ""


def test_squash_punctuation_collapses_every_separator_alike():
    assert squash_punctuation("Foo_Bar-Baz 10K") == "foo bar baz 10k"
