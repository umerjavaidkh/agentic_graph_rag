"""
tests/shared/test_language_scoping_unit.py — how a document gets its language,
and how that language scopes a query.

The rule under test, from docs/DESIGN_language_independence.md:

  * English is the default -- what a document is when nothing else claims it.
  * Any other configured language present in enough quantity wins.
  * One language per document, stamped onto every node of it.
  * The scope predicate compiles to "true" while one language is live, so
    English is unchanged by construction rather than by testing.

Run with:
    python -m pytest tests/shared/test_language_scoping_unit.py -v
"""
from __future__ import annotations

import pytest

from src.shared import language as language_mod
from src.shared.language import (
    ARABIC,
    ENGLISH,
    LanguageProfile,
    configured_languages,
    detect_language,
    get_profile,
    script_shares,
)
from src.shared.neo4j.tenancy import language_filter

_ENGLISH_TEXT = (
    "Revenue increased by 12 percent in the fourth quarter, driven by higher "
    "realisations across every segment in which the company operates."
)
_ARABIC_TEXT = "النتائج المالية للربع الرابع من العام الحالي مقارنة بالعام الماضي"


@pytest.fixture
def two_languages(monkeypatch):
    """A deployment with Arabic switched on."""
    monkeypatch.setattr(language_mod, "ENABLED_LANGUAGES", ("en", "ar"))


# ── detection ────────────────────────────────────────────────────────────


def test_an_english_document_is_the_default_language():
    assert detect_language(_ENGLISH_TEXT) == "en"


def test_an_arabic_document_is_arabic():
    assert detect_language(_ARABIC_TEXT) == "ar"


def test_a_bilingual_document_goes_to_the_non_default_language():
    """The rule, stated directly: Arabic alongside English is Arabic."""
    assert detect_language(f"{_ENGLISH_TEXT} {_ARABIC_TEXT}") == "ar"


def test_a_stray_glyph_does_not_move_an_english_document():
    """Why the rule is a share and not a presence test.

    OCR on a scanned page emits stray glyphs routinely. Under a presence
    test one of them would move a 300-page English filing into the Arabic
    corpus, where no English query would ever reach it again.
    """
    assert detect_language(_ENGLISH_TEXT + " ا") == "en"


def test_the_threshold_is_what_decides_a_marginal_document():
    """A share just under the bar stays English; just over it does not."""
    mostly_english = _ENGLISH_TEXT * 20 + " " + _ARABIC_TEXT
    assert detect_language(mostly_english, threshold=0.5) == "en"
    assert detect_language(mostly_english, threshold=0.001) == "ar"


def test_empty_and_scriptless_text_take_the_default():
    for text in ("", "   ", "12345", "%$#@!"):
        assert detect_language(text) == "en", text


def test_digits_and_punctuation_do_not_dilute_the_share():
    """Counting them would make a document's language depend on how many
    tables it contains, which is not a property of its language."""
    bare = script_shares(_ARABIC_TEXT)
    with_table = script_shares(_ARABIC_TEXT + " 1,234.56 | 7,890.12 | 3.4% |")
    assert bare["ar"] == pytest.approx(with_table["ar"])


def test_detection_is_deterministic_across_reingest():
    """A document must not be able to change corpora by being re-ingested."""
    text = f"{_ENGLISH_TEXT} {_ARABIC_TEXT}"
    assert len({detect_language(text) for _ in range(10)}) == 1


# ── profiles ─────────────────────────────────────────────────────────────


def test_an_unknown_code_falls_back_rather_than_raising():
    """A code from a future build must not be able to break retrieval."""
    assert get_profile("zz").code == "en"
    assert get_profile(None).code == "en"
    assert get_profile("").code == "en"


def test_a_code_is_normalised_before_lookup():
    assert get_profile("  AR ").code == "ar"


def test_the_default_profile_is_a_no_op():
    """Registering a language and filling in nothing must behave as today."""
    assert ENGLISH.normalize("Revenue") == "Revenue"
    assert ENGLISH.scripts == ()
    assert ENGLISH.structural_terms == frozenset()


def test_a_new_language_is_registered_not_coded(two_languages, monkeypatch):
    """Adding a language must be data, never a branch.

    Nothing in the module tests for Arabic; it is a precedence over
    whatever profiles are registered. A third language proves that.
    """
    greek = LanguageProfile(code="el", name="Greek", scripts=((0x0370, 0x03FF),))
    monkeypatch.setitem(language_mod._PROFILES, "el", greek)
    monkeypatch.setattr(language_mod, "ENABLED_LANGUAGES", ("en", "ar", "el"))

    assert detect_language("Ολοκληρωμένα αποτελέσματα του τριμήνου") == "el"
    assert configured_languages() == ["en", "ar", "el"]


# ── configuration and the scope predicate ────────────────────────────────


def test_only_one_language_is_live_by_default():
    """Registering Arabic adds it to the catalogue; enabling it is separate.

    If these were the same thing, merging Arabic support would switch
    scoping on everywhere at once -- against a corpus where no document
    has a `language` property yet, so every query would scope to nothing.
    """
    assert ARABIC.code in language_mod._PROFILES
    assert configured_languages() == ["en"]


def test_the_predicate_compiles_away_with_one_language():
    """This is what "English is byte-identical by construction" means.

    With one language live there is no filter to get wrong -- Neo4j folds
    the literal away, and all 20 scope call sites can splice it today.
    """
    assert language_filter() == "true"
    assert language_filter("s", "$lang") == "true"


def test_the_predicate_scopes_once_a_second_language_is_enabled(two_languages):
    assert language_filter() == "n.language = $language"
    assert language_filter("s", "$lang") == "s.language = $lang"


def test_the_default_language_is_always_listed_first(two_languages):
    assert configured_languages()[0] == "en"


def test_an_enabled_language_with_no_profile_is_ignored(monkeypatch):
    """Config naming a language nothing implements must not scope to nothing."""
    monkeypatch.setattr(language_mod, "ENABLED_LANGUAGES", ("en", "xx"))
    assert configured_languages() == ["en"]
    assert language_filter() == "true"
