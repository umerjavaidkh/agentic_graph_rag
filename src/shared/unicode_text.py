"""unicode_text.py — word and sentence handling that is not tied to ASCII.

Twenty-four regexes in this repository spell "a letter" as `[a-z]` or
`[A-Z]`. Most of them are right to: they parse Cypher identifiers, thread
ids and provider error strings, all of which really are ASCII, and
widening those would only let malformed input through. The ones that
tokenize *document and query text* are a different matter, and they fail
in two different ways::

    Résultats   -> ['sultats']     the accent splits the word
    Prévisions  -> ['visions']     a real English word, out of French text
    النتائج      -> []              nothing matches at all

Arabic fails cleanly and visibly. Accented Latin fails silently: the
fragments are plausible tokens that go on to pollute corpus frequency
statistics and match unrelated English text, and nothing about the output
looks broken. The second is the worse bug, and it is live today.

The classes below are exact supersets of the ASCII ones they replace, so
text made only of ASCII tokenizes identically -- the English suite is
unchanged by construction rather than by testing.

Case is handled in Python rather than in the pattern. `re` has no
`\\p{Lu}`, and `str.isupper()` already means the right thing in every
script: True for `É`, False for `é`, and False for Arabic, which has no
case at all. A caseless script therefore takes the "not lowercase" branch
instead of matching nothing, which is what `[A-Z]` did to it.

Not here, deliberately: Arabic normalization (alef variants, teh marbuta,
tashkeel, tatweel) belongs to the `LanguageProfile` in Phase 2, not to a
shared helper every language would pay for. Combining marks are not word
characters to Python, so vocalized Arabic still splits on its diacritics
until that normalizer strips them.
"""
from __future__ import annotations

import re
import unicodedata

# A letter in any script: a word character that is neither a digit nor the
# underscore. `\w` admits both, which is why it cannot be used directly.
LETTER = _LETTER = r"[^\W\d_]"
# Letter or digit in any script -- `\w` minus the underscore. The
# underscore stays out so `unit_price` still tokenizes to two words, which
# is exactly what the `[a-z0-9]` classes being replaced here did.
_ALNUM = r"[^\W_]"

_WORD = re.compile(rf"{_LETTER}{_ALNUM}*")
_WORD_HYPHENATED = re.compile(rf"{_LETTER}(?:{_ALNUM}|-)*")
_TOKEN = re.compile(rf"{_ALNUM}+")
_TOKEN_HYPHENATED = re.compile(rf"(?:{_ALNUM}|-)+")
_LETTERS = re.compile(rf"{_LETTER}+")
_LETTERS_HYPHENATED = re.compile(rf"{_LETTER}+(?:-{_LETTER}+)*")

# Everything that is not a letter or a digit, including the underscore.
# Over lowercased text this is precisely `[^a-z0-9]`.
_NOT_ALNUM = re.compile(r"[\W_]+")

# A digit before the period is a section number, not a sentence end:
# splitting on "2.4. Interoperability" produced a fragment ending at
# '"2.4.' and attributed the rest to a different page.
_SENTENCE_BOUNDARY = re.compile(r"(?<![0-9].)(?<=[.!?])\s+")


def words(
    text: str,
    *,
    min_length: int = 1,
    hyphens: bool = False,
    numeric: bool = False,
) -> list[str]:
    """Lowercased word tokens, in whatever script the text is written in.

    `min_length` filters after matching rather than inside the pattern.
    The two are equivalent -- a token shorter than the bound is dropped
    either way -- and the filter says what it means.

    `hyphens` keeps `case-control` as one token; `numeric` allows a token
    that is all digits, which callers matching quantities need and
    callers building vocabulary do not.
    """
    if numeric:
        pattern = _TOKEN_HYPHENATED if hyphens else _TOKEN
    else:
        pattern = _WORD_HYPHENATED if hyphens else _WORD
    found = pattern.findall((text or "").lower())
    if min_length <= 1:
        return found
    return [w for w in found if len(w) >= min_length]


def letters(text: str, *, min_length: int = 1, hyphens: bool = False) -> list[str]:
    """Lowercased runs of letters only, in any script.

    Distinct from `words` because a digit *ends* a token here rather than
    continuing it: the callers are matching prose against prose, where
    `IRS225` is not the word `irs`. `hyphens` joins letter runs across a
    hyphen (`case-control`) and never keeps a leading or trailing one.
    """
    pattern = _LETTERS_HYPHENATED if hyphens else _LETTERS
    found = pattern.findall((text or "").lower())
    if min_length <= 1:
        return found
    return [w for w in found if len(w) >= min_length]


def opens_sentence(text: str) -> bool:
    """Whether `text` begins the way a new sentence does.

    True for an uppercase letter in any script, and for a letter in a
    script that has no case at all -- Arabic, Hebrew, CJK -- where
    "must be uppercase" would otherwise reject every sentence there is.
    False for a lowercase letter, which is the discrimination the callers
    actually want, and False for a digit, matching the `[A-Z]` this
    replaces.
    """
    ch = (text or "")[:1]
    if not ch or not ch.isalpha():
        return False
    return ch.isupper() or not ch.islower()


def sentence_split(text: str) -> list[str]:
    """Split on sentence ends without requiring a case distinction.

    The ASCII version demanded an uppercase letter after the space, which
    Arabic can never supply, so an Arabic answer split into exactly one
    sentence and every claim in it came back uncited. `opens_sentence`
    keeps the uppercase requirement wherever the script has one and drops
    it where it does not.
    """
    src = text or ""
    out: list[str] = []
    start = 0
    for m in _SENTENCE_BOUNDARY.finditer(src):
        if not opens_sentence(src[m.end():m.end() + 1]):
            continue
        out.append(src[start:m.start()])
        start = m.end()
    out.append(src[start:])
    return out


def squash_punctuation(text: str) -> str:
    """Lowercase, with every run of non-alphanumerics collapsed to a space.

    Filenames use underscores, display titles use spaces and filing ids
    use hyphens; none of that changes which document is meant.
    """
    return _NOT_ALNUM.sub(" ", (text or "").lower()).strip()


def fold(text: str) -> str:
    """NFKC-normalized text, for comparison rather than for storage.

    Applied to *queries* only. The same normalization over stored text
    would rewrite English content -- the ligature `ﬁ` becomes `fi` -- and
    that is a re-ingestion and a re-measurement, not a Phase 0 change. It
    is deferred to Phase 2, where Arabic documents arrive and the corpus
    has to be rebuilt anyway.
    """
    return unicodedata.normalize("NFKC", text) if text else text
