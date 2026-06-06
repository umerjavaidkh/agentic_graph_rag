"""Text normalization helpers for document retrieval."""
from __future__ import annotations

import re

_BROKEN_URL_RE = re.compile(r"https?://[^\s\)\]>\"']+(?:\s+[^\s\)\]>\"']+)+", re.I)


def normalize_broken_urls(text: str) -> str:
    """Fix PDF line-breaks/spaces inside URLs."""

    def _fix(match: re.Match) -> str:
        return re.sub(r"\s+", "", match.group(0))

    return _BROKEN_URL_RE.sub(_fix, text or "")


def extract_urls(text: str) -> list[str]:
    normalized = normalize_broken_urls(text)
    return list(dict.fromkeys(re.findall(r"https?://[^\s\)\]>\"']+", normalized)))


def query_anchor_terms(query: str) -> list[str]:
    """Proper names and dotted tokens from the user question (corpus-agnostic)."""
    terms: list[str] = []
    seen: set[str] = set()

    def _add(raw: str) -> None:
        for variant in (
            raw,
            raw.replace(".", ""),
            raw.replace(".", " "),
            raw.replace("-", " "),
        ):
            tl = variant.lower().strip()
            if tl and tl not in seen and len(tl) >= 2:
                seen.add(tl)
                terms.append(tl)

    for m in re.finditer(r"\b[A-Za-z][\w]*(?:\.[\w]+)+\b", query or ""):
        _add(m.group(0))

    for m in re.finditer(r"\b[A-Z][A-Z0-9]{2,}\b", query or ""):
        _add(m.group(0))

    for m in re.finditer(r"\b[A-Z][a-z][A-Za-z0-9]{2,}\b", query or ""):
        _add(m.group(0))

    return terms[:10]


_query_anchor_terms = query_anchor_terms
_extract_urls = extract_urls
