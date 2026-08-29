"""
tests/shared/test_answer_language_directive_unit.py — Phase 4.

The model already answered Arabic questions in Arabic, by disposition.
That is not a guarantee: nothing asked it to, so a model swap could turn
it into English answers about Arabic documents with no code change and
no failing test.

The directive makes it explicit, and carries the harder half with it --
citations must stay verbatim in the source language, because a
translated quotation cannot be checked against the page it cites and
every deterministic eval in eval/ depends on exact spans.

Two things this file guards that are easy to get wrong:

  * it must reach prompts that produce PROSE and nothing else.
    document_verify emits strict JSON; telling it to "answer in the
    language of the question" invites Arabic prose where a parser
    expects `{"valid": true}`.
  * it must be absent while one language is configured, so a
    single-language deployment gets byte-identical prompts. A prompt
    change is not free even when it is logically a no-op -- the model is
    not a pure function of intent.

Run with:
    python -m pytest tests/shared/test_answer_language_directive_unit.py -v
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.shared.config import prompts as prompts_mod
from src.shared.config.prompts import _ANSWER_PROMPTS, language_directive

REPO = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO / "src" / "prompts"


@pytest.fixture
def one_language(monkeypatch):
    import src.shared.language as language_mod

    monkeypatch.setattr(language_mod, "ENABLED_LANGUAGES", ("en",))


@pytest.fixture
def two_languages(monkeypatch):
    import src.shared.language as language_mod

    monkeypatch.setattr(language_mod, "ENABLED_LANGUAGES", ("en", "ar"))


def test_no_directive_while_one_language_is_configured(one_language):
    """English prompts stay byte-identical under a single-language deploy."""
    assert language_directive() == ""


def test_the_directive_appears_once_two_languages_are_live(two_languages):
    directive = language_directive()
    assert "same language as the question" in directive
    assert "VERBATIM" in directive


def test_the_directive_forbids_translating_citations(two_languages):
    """The half that matters more than answering in the right language."""
    directive = language_directive().lower()
    assert "never translate" in directive
    assert "cited" in directive or "cites" in directive


def test_it_actually_reaches_a_rendered_answer_prompt(two_languages):
    """Rendered, not asserted about.

    The first version of this test checked that a name was in a set,
    which would have passed just as happily if load_prompt never appended
    anything at all.
    """
    prompts_mod.clear_prompt_cache()
    rendered = prompts_mod.load_prompt(
        "document_default", question="ما هي الهويات؟", context="..."
    )
    assert "same language as the question" in rendered
    assert rendered.count("same language as the question") == 1


def test_it_does_not_reach_a_rendered_json_prompt(two_languages):
    prompts_mod.clear_prompt_cache()
    rendered = prompts_mod.load_prompt(
        "document_verify", question="q", answer="a", chunks="c"
    )
    assert "same language as the question" not in rendered
    assert '"valid": true' in rendered


def test_a_rendered_answer_prompt_is_unchanged_under_one_language(one_language):
    """The byte-identical guarantee, on real rendered text."""
    prompts_mod.clear_prompt_cache()
    rendered = prompts_mod.load_prompt("document_default", question="q", context="c")
    raw = (PROMPTS_DIR / "document_default.txt").read_text(encoding="utf-8")
    assert rendered == raw.format(question="q", context="c")


def test_the_json_verifier_never_gets_it():
    """It emits `{"valid": true}`. Prose instructions do not belong there."""
    assert "document_verify" not in _ANSWER_PROMPTS
    assert "structured_verify" not in _ANSWER_PROMPTS


def test_routing_and_cypher_prompts_never_get_it():
    """route_query emits a tool name; text2cypher emits Cypher. A language
    instruction there is noise at best, and at worst asks for Arabic
    identifiers inside a query."""
    for name in ("route_query", "structured_text2cypher", "structured_multistep_plan"):
        assert name not in _ANSWER_PROMPTS


def test_structured_prompts_are_excluded():
    """The business graph has no language dimension."""
    assert not any(n.startswith("structured_") for n in _ANSWER_PROMPTS)


def test_every_listed_prompt_actually_exists():
    """A name that does not resolve to a file is a directive that silently
    reaches nothing."""
    missing = [n for n in _ANSWER_PROMPTS if not (PROMPTS_DIR / f"{n}.txt").exists()]
    assert not missing, f"listed prompts with no file: {missing}"


def test_every_prose_document_prompt_is_covered():
    """The list must not drift as prompts are added.

    document_verify is the one deliberate exclusion: it emits JSON.
    """
    on_disk = {p.stem for p in PROMPTS_DIR.glob("document_*.txt")}
    uncovered = on_disk - _ANSWER_PROMPTS - {"document_verify"}
    assert not uncovered, (
        "document prompts producing prose that never get the language "
        f"directive: {sorted(uncovered)}"
    )
