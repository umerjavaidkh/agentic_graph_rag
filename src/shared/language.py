"""language.py — which language a document is in, and what that language needs.

Two jobs, kept in one file because they are the same decision seen from
either end: `detect_language` answers it for a document at ingest, and
`get_profile` answers "so what does that language need" for everything
downstream.

The rule, from `docs/DESIGN_language_independence.md`:

  * English is the default -- what a document is when no other profile
    claims it. That is why every existing document backfills to `en`
    without being examined.
  * Any other registered language present in enough quantity wins over
    the default. A document laid out in both Arabic and English is
    Arabic.
  * One language per document, stamped onto every node in it.

"In enough quantity" is a *share* of the document's letters, never a
presence test. A presence test would move a 300-page English filing into
the Arabic corpus on a single stray glyph, and OCR on scanned pages
produces stray glyphs routinely. The threshold is a setting because the
right value has to be measured against real bilingual documents -- it is
not knowable in advance, and a constant here would be a number nobody
could defend.

Nothing in this file tests for Arabic. It is a precedence over registered
profiles, so adding a third language is registering a profile and
changing no logic -- the same plug-and-play shape as the retrieval
strategies, and the reason the design forbids `if lang == "ar"` in
retrieval.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .config.settings import (
    DEFAULT_LANGUAGE,
    ENABLED_LANGUAGES,
    LANGUAGE_SHARE_THRESHOLD,
)
from .unicode_text import letters


@dataclass(frozen=True)
class LanguageProfile:
    """Everything that differs between languages, in one registrable object.

    Empty `scripts` marks the default profile: it is not detected, it is
    what text is when nothing else claims it. Every other field defaults
    to today's English behaviour, so registering a language and filling in
    nothing is a language that behaves exactly as the system does now --
    which is what makes the English profile provably a no-op.
    """

    code: str
    name: str
    # Inclusive Unicode code-point ranges whose letters count as this
    # language. Ranges rather than a script-name lookup because Python
    # ships no script database, and the alternative is a dependency for
    # something this file needs four numbers to express.
    scripts: tuple[tuple[int, int], ...] = ()
    # Applied to text before matching. English needs nothing; Arabic
    # collapses alef variants, teh marbuta, tashkeel and tatweel here in
    # Phase 2 -- in the profile rather than in the shared tokenizer, so
    # no other language pays for it.
    normalize: Callable[[str], str] = lambda text: text
    # Words that mark structure ("chapter", "section", "table"). Empty
    # means "use the existing English constants", which is what every
    # call site does today.
    structural_terms: frozenset[str] = field(default_factory=frozenset)


ENGLISH = LanguageProfile(code="en", name="English")

ARABIC = LanguageProfile(
    code="ar",
    name="Arabic",
    scripts=(
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    ),
)

_PROFILES: dict[str, LanguageProfile] = {}


def register(profile: LanguageProfile) -> None:
    _PROFILES[profile.code] = profile


register(ENGLISH)
register(ARABIC)


def get_profile(code: Optional[str]) -> LanguageProfile:
    """The profile for `code`, falling back to the default language.

    Falls back rather than raising on purpose. An unknown language code
    on a request is a request that should be answered in the default
    language, not a 500 -- and a `language` property written by an older
    build must not be able to break retrieval for a document.
    """
    return _PROFILES.get((code or "").lower().strip(), _PROFILES[DEFAULT_LANGUAGE])


def configured_languages() -> list[str]:
    """Language codes live in this deployment, default first.

    Deliberately not "every registered profile". Registering Arabic adds
    it to the catalogue; naming it in ENABLED_LANGUAGES turns it on. If
    this read the catalogue instead, merging Arabic support would switch
    scoping on for every deployment at once -- against a corpus where no
    document has a `language` property yet, so every query would scope to
    nothing.

    `language_filter` reads this to decide whether scoping is needed at
    all: with one language live there is nothing to separate, so the
    predicate compiles away and English is untouched by construction.
    """
    live = [c for c in ENABLED_LANGUAGES if c in _PROFILES]
    rest = sorted(c for c in live if c != DEFAULT_LANGUAGE)
    return [DEFAULT_LANGUAGE, *rest]


def script_shares(text: str) -> dict[str, float]:
    """Share of the text's letters belonging to each non-default language.

    Letters only. Digits, punctuation and whitespace are script-neutral --
    counting them would dilute a genuinely Arabic document in proportion
    to how many tables it contains, which is not a property of its
    language.
    """
    counted = 0
    hits: dict[str, int] = {}
    for word in letters(text):
        for ch in word:
            counted += 1
            point = ord(ch)
            for profile in _PROFILES.values():
                if any(lo <= point <= hi for lo, hi in profile.scripts):
                    hits[profile.code] = hits.get(profile.code, 0) + 1
                    break
    if not counted:
        return {}
    return {code: n / counted for code, n in hits.items()}


def detect_language(text: str, *, threshold: Optional[float] = None) -> str:
    """The one language code for a document's text.

    The largest non-default share wins if it clears the threshold, so a
    bilingual document goes to the non-default language exactly as the
    design says. Ties break by code for determinism: re-ingesting a
    document must not be able to move it between corpora.
    """
    bar = LANGUAGE_SHARE_THRESHOLD if threshold is None else threshold
    shares = script_shares(text)
    clearing = sorted(
        ((share, code) for code, share in shares.items() if share >= bar),
        key=lambda pair: (-pair[0], pair[1]),
    )
    return clearing[0][1] if clearing else DEFAULT_LANGUAGE
