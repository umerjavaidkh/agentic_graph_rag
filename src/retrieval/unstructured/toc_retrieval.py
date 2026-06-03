"""Table-of-contents retrieval helpers (document-agnostic)."""
from __future__ import annotations

import re
from typing import Any, Optional

_TOC_PAGE_HEADING_RE = re.compile(
    r"(?:^|\n)\s*(?:table\s+of\s+contents?|contents)\s*(?:\n|$)",
    re.I | re.M,
)
_TOC_LINE_RE = re.compile(
    r"(?:\.{2,}|…+|\s\.\s)\s*\d{1,4}\s*$"  # dotted leaders + page number
    r"|\s+\d{1,4}\s*$",
    re.I,
)
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
    """Higher = more likely a dedicated TOC page (not body content)."""
    body = (text or "").strip()
    if len(body) < 40:
        return 0.0
    score = 0.0
    head = body[:600]
    if _TOC_PAGE_HEADING_RE.search(head):
        score += 0.45
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if len(lines) < 3:
        return score
    toc_lines = sum(1 for ln in lines if _TOC_LINE_RE.search(ln))
    ratio = toc_lines / len(lines)
    score += min(0.55, ratio * 0.9)
    # TOC pages are usually short lists, not long prose.
    if len(lines) <= 80 and len(body) < 12000:
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
