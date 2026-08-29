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


def _identity(text: str) -> str:
    """The default normalizer: text is already in the space it matches in.

    Returning the same object matters -- callers write a `match_text`
    property only when normalization actually changed something, so an
    identity normalizer costs no storage and leaves matching provably
    byte-identical.
    """
    return text


# Arabic writes the same word several ways, and a reader does not
# distinguish them. Hamza carriers vary by orthographic convention
# (أحمد / احمد), teh marbuta and heh are interchanged in much informal
# and OCR'd text, tashkeel is optional and usually absent from what
# someone types, and tatweel is a typographic stretch with no meaning
# at all.
#
# Measured on the ingested corpus before this existed: the query
# "الهويات" matched 0 nodes and the stored form "الهويّات" matched 38 --
# the same word, one shadda apart.
_ARABIC_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",   # alef with hamza / madda
    "ى": "ي", "ئ": "ي",                        # alef maqsura, yeh with hamza
    "ة": "ه",                                  # teh marbuta
    "ؤ": "و",                                  # waw with hamza
    "ـ": "",                                   # tatweel: pure typography
})

# Tashkeel and the Quranic annotation marks: combining characters that
# carry pronunciation, not identity. Removed rather than folded, and
# removed LAST so the table above still sees the letters it expects.
_ARABIC_MARKS = dict.fromkeys(
    list(range(0x064B, 0x0653))   # fathatan..sukun
    + list(range(0x0653, 0x0660))  # maddah, hamza above/below, superscript alef
    + list(range(0x0670, 0x0671))  # superscript alef
    + list(range(0x06D6, 0x06ED))  # Quranic annotation
)


def normalize_arabic(text: str) -> str:
    """Collapse the spellings of a word that a reader treats as one.

    Applied to the MATCHING key on both sides -- the query and a derived
    `match_text` -- and never to stored text. Citations have to stay
    byte-identical to the PDF, and every deterministic eval in eval/
    depends on exact spans.
    """
    if not text:
        return text
    return text.translate(_ARABIC_FOLD).translate(_ARABIC_MARKS)


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
    normalize: Callable[[str], str] = _identity
    # Words that mark structure ("chapter", "section", "table"). Empty
    # means "use the existing English constants", which is what every
    # call site does today.
    structural_terms: frozenset[str] = field(default_factory=frozenset)
    # Extra alternations for the question-SHAPE regexes, keyed by shape.
    # Written in this profile's normalized form, because the matcher
    # normalizes the query first.
    #
    # Unioned into the English patterns rather than selected between: a
    # pattern written in one script cannot match text in another, so
    # there is no language to branch on and no way for one language's
    # patterns to fire on another's text. That is the design's rule --
    # language is data, never a branch -- holding without a parameter.
    intent_patterns: dict = field(default_factory=dict)


ENGLISH = LanguageProfile(code="en", name="English")

# Question shapes in Arabic, in NORMALIZED form (teh marbuta folded to
# heh, hamza carriers to bare alef) because the matcher normalizes the
# query before testing. Written out rather than transliterated from the
# English list: "list all the sections" and "اذكر جميع الاقسام" are the
# same shape but not the same words, and a translated regex would match
# neither idiom well.
_ARABIC_INTENT = {
    # The definite article is optional throughout: Arabic drops it freely
    # in questions ("ما هي فصول هذا المستند" alongside "ما هي الفصول"),
    # and a pattern that requires it matches only half the idiom.
    "toc": (
        r"فهرس|جدول\s+المحتويات|المحتويات|"
        r"(?:قايمه|اذكر|اسرد|ما)\s+(?:\S+\s+){0,2}?(?:ال)?(?:فصول|اقسام|عناوين)|"
        r"ما\s+هي\s+(?:ال)?(?:فصول|اقسام|عناوين)"
    ),
    "page": r"صفحه\s*\d+|الصفحه\s*(?:رقم\s*)?\d+|محتوى\s+الصفحه",
    "enumeration": r"اذكر\s+(?:كل|جميع)|عدد\s+(?:كل|جميع)|اسرد|ما\s+هي\s+(?:كل|جميع)|قايمه\s+ب",
    "synthesis": r"قارن|قارن\s+بين|العلاقه\s+بين|كيف\s+يرتبط|الفرق\s+بين",
    "overview": r"نظره\s+عامه|ملخص|لخص|عن\s+ماذا\s+يتحدث|عم\s+يتحدث|ما\s+موضوع",
}


ARABIC = LanguageProfile(
    code="ar",
    name="Arabic",
    normalize=normalize_arabic,
    intent_patterns=_ARABIC_INTENT,
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


def derive_match_text(search_text: Optional[str], language: Optional[str]) -> Optional[str]:
    """The normalized key for `search_text`, or None when there is no work.

    None on purpose rather than a copy. English's normalizer is the
    identity, so a copy would double the stored text of the entire English
    corpus to say nothing -- and callers match on
    `coalesce(n.match_text, n.search_text)`, which falls through to exactly
    the behaviour English had before this existed. Byte-identical by
    construction, and free.

    A language whose normalizer does change the text (Arabic) gets the
    property, and matching happens in that normalized space on both sides.
    """
    if not search_text:
        return None
    normalized = get_profile(language).normalize(search_text)
    return normalized if normalized != search_text else None


def intent_alternations(shape: str) -> list[str]:
    """Every registered profile's extra patterns for one question shape.

    Every profile, not the request's language. A shape regex is asking
    "what kind of question is this", and the scripts do not overlap: an
    Arabic alternation cannot match ASCII and an English one cannot match
    Arabic. Unioning them is therefore free of cross-language false
    positives AND removes the need to thread a language into all 77
    shape checks -- which would have been the branch the design forbids,
    spelled as a parameter.
    """
    out = []
    for profile in _PROFILES.values():
        pattern = (profile.intent_patterns or {}).get(shape)
        if pattern:
            out.append(pattern)
    return out


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
