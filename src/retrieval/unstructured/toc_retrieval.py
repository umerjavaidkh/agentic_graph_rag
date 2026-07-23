"""Table-of-contents retrieval helpers (document-agnostic)."""
from __future__ import annotations

import re
from typing import Any, Optional

_TOC_PAGE_HEADING_RE = re.compile(
    # "Form 10-K Index" / "10-K Index" is a real, standard alternate TOC
    # heading used across many actual SEC filings (banks in particular),
    # not specific to any one filer — without it, a page using this
    # convention instead of "Table of Contents" got zero heading credit.
    r"(?:^|\n)\s*(?:table\s+of\s+contents?|contents|(?:form\s+)?10-?k\s+index)\s*(?:\n|$)",
    re.I | re.M,
)
_TOC_LINE_RE = re.compile(
    r"(?:\.{2,}|…+|\s\.\s)\s*\d{1,4}\s*$"  # dotted leaders + page number
    r"|\s+\d{1,4}\s*$",
    re.I,
)
# A line that is *only* a page number — arabic or roman (TOCs often put the
# page number on its own line under each entry title).
_PAGE_NUMBER_LINE_RE = re.compile(r"^(?:\d{1,4}|[ivxlcdm]{1,6})$", re.I)
_ENTRY_TITLE_RE = re.compile(r"[A-Za-z]")
_TOC_SECTION_TITLE_RE = re.compile(
    r"^table\s+of\s+contents?\.?$|^contents$",
    re.I,
)
_TOC_OUTLINE_SKIP_RE = re.compile(
    r"^(?:box\s+\d|figure\s*\d|table\s*\d|page\s+\d+)\b",
    re.I,
)
_NUMBERED_OUTLINE_RE = re.compile(r"^\d+(?:\.\d+)*\.?\s+\S")


def score_page_text_as_toc(text: str) -> float:
    """Higher = more likely a dedicated TOC page (not body content).

    Many SEC filings print "Table of Contents" as a small running header on
    every page that follows the real TOC, not just on the TOC page itself
    (e.g. Tesla's 10-K: the actual Item/page-number listing is on one page
    with no such heading at all -- its first line is just the column header
    "Page" -- while the very next page, pure "Forward-Looking Statements"
    prose with zero entries, carries the "Table of Contents" running header
    and used to win on that heading match alone). The heading phrase is
    therefore corroborating evidence at best, not proof by itself -- it's
    scaled by how much real entry/page-number structure the page's own body
    actually has, so a heading with no supporting structure contributes
    close to nothing, while a strong entry-pair ratio can win the page on
    its own even with zero heading match (verified: this flips the Tesla
    case correctly -- real TOC page 0.379->0.528, decoy page 0.500->0.050
    -- while every previously-correct document's real TOC page scores the
    same or higher and every negative-control page scores the same or
    lower).
    """
    body = (text or "").strip()
    if len(body) < 40:
        return 0.0
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if len(lines) < 3:
        return 0.0

    # Same-line page numbers / dotted leaders.
    toc_lines = sum(1 for ln in lines if _TOC_LINE_RE.search(ln))

    # Title line immediately followed by a standalone page-number line
    # (arabic or roman) — the common multi-line TOC layout.
    entry_pairs = 0
    for i in range(len(lines) - 1):
        if _PAGE_NUMBER_LINE_RE.match(lines[i + 1]) and _ENTRY_TITLE_RE.search(lines[i]):
            entry_pairs += 1

    signal = toc_lines + entry_pairs
    ratio = signal / max(1, len(lines))

    score = 0.0
    if _TOC_PAGE_HEADING_RE.search(body[:600]):
        score += 0.45 * min(1.0, ratio * 6)
    score += min(0.7, ratio * 1.6)
    # TOC pages are usually short lists, not long prose.
    if len(lines) <= 120 and len(body) < 14000:
        score += 0.05
    return min(1.0, score)


def format_toc_chunk(
    *,
    body: str,
    doc_title: str,
    source: str,
    pdf_page: Optional[int] = None,
    document_page: Optional[str] = None,
) -> dict[str, Any]:
    header = [f"Document: {doc_title}", source, ""]
    if pdf_page is not None or document_page:
        loc = []
        if document_page:
            loc.append(f"printed page {document_page}")
        if pdf_page is not None:
            loc.append(f"PDF page {pdf_page}")
        header.insert(2, f"Location: {', '.join(loc)}")
    text = "\n".join(header) + (body or "").strip()
    return {
        "id": "structural_toc",
        "title": f"Table of contents — {doc_title}",
        "text": text,
        "score": 1.0,
        "related": ["via:toc_page" if "page" in source.lower() else "via:toc_outline"],
        "pdf_page": pdf_page,
        "document_page": document_page,
    }


def format_outline_chunk(
    entries: list[str],
    *,
    doc_title: str,
) -> dict[str, Any]:
    lines = [
        f"Document: {doc_title}",
        "Table of contents (inferred from chapter/section headings — no TOC page found):",
        "",
    ]
    for i, title in enumerate(entries, 1):
        lines.append(f"{i}. {title}")
    return {
        "id": "structural_toc",
        "title": f"Table of contents — {doc_title}",
        "text": "\n".join(lines),
        "score": 0.95,
        "related": ["via:toc_outline"],
    }


def include_in_outline_fallback(title: str, depth: int, label: str) -> bool:
    t = (title or "").strip()
    if not t or len(t) < 2:
        return False
    if t.lower() in ("contents", "content"):
        return False
    if _TOC_OUTLINE_SKIP_RE.match(t):
        return False
    if label == "Chapter":
        return True
    if depth <= 1:
        return True
    if _NUMBERED_OUTLINE_RE.match(t):
        return True
    if t.isupper() and len(t) < 100 and not _TOC_OUTLINE_SKIP_RE.match(t):
        return True
    return False


def section_title_is_toc(title: str) -> bool:
    return bool(_TOC_SECTION_TITLE_RE.match((title or "").strip()))
