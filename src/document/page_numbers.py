"""
PDF page index vs document (printed) page labels on the page itself.

document_page is stored as text (e.g. "43", "iii", "A") so roman numerals and
annex letters are preserved. User-facing page queries prefer document_page when set.
"""
from __future__ import annotations

import re
from typing import Optional

from ..models import DKGNode, NodeType

# Footer / header line patterns (order matters — specific first)
_PAGE_LABEL_PATTERNS = [
    re.compile(r"(?:^|\s)(?:page|p\.?|pg\.?)\s*[:.]?\s*([a-zA-Z0-9ivxlcdm\-]+)\s*$", re.I),
    re.compile(r"^\s*([ivxlcdm]+)\s*$", re.I),
    re.compile(r"^\s*([a-z])\s*$", re.I),
    re.compile(r"^\s*(\d{1,4})\s*$"),
    re.compile(r"[-|]\s*(\d{1,4})\s*[-|]\s*$"),
]

_ROMAN_VALUES = {
    "i": 1, "ii": 2, "iii": 3, "iv": 4, "v": 5, "vi": 6, "vii": 7, "viii": 8,
    "ix": 9, "x": 10, "xi": 11, "xii": 12, "xiii": 13, "xiv": 14, "xv": 15,
    "xvi": 16, "xvii": 17, "xviii": 18, "xix": 19, "xx": 20,
}


def detect_document_page_label(page_text: str) -> Optional[str]:
    """
  Read likely printed page number from bottom/top lines of extracted page text.
  Returns label as shown on PDF (string): '43', 'iii', 'A', etc.
    """
    if not page_text or not page_text.strip():
        return None

    lines = [ln.strip() for ln in page_text.splitlines() if ln.strip()]
    if not lines:
        return None

    candidates = lines[-4:] + lines[:2]
    seen: set[str] = set()
    for line in candidates:
        if line in seen:
            continue
        seen.add(line)
        label = _label_from_line(line)
        if label:
            return label
    return None


def _label_from_line(line: str) -> Optional[str]:
    for pat in _PAGE_LABEL_PATTERNS:
        m = pat.search(line.strip())
        if m:
            raw = m.group(1).strip()
            if raw and len(raw) <= 12:
                return raw
    return None


def document_page_matches_pdf(document_page: Optional[str], pdf_page: int) -> bool:
    if not document_page:
        return False
    if document_page.isdigit() and int(document_page) == pdf_page:
        return True
    return False


def build_page_tags(pdf_page: int, document_page: Optional[str]) -> list[str]:
    tags = [f"pdf:{pdf_page}", f"pdf-page:{pdf_page}"]
    if document_page:
        norm = document_page.strip()
        tags.append(f"doc:{norm}")
        tags.append(f"document-page:{norm}")
        if norm.isdigit():
            tags.append(f"doc-num:{norm}")
        low = norm.lower()
        if low in _ROMAN_VALUES:
            tags.append(f"doc-roman:{low}")
            tags.append(f"doc-num:{_ROMAN_VALUES[low]}")
    if document_page and document_page_matches_pdf(document_page, pdf_page):
        tags.append("pdf-doc-same")
    else:
        if document_page:
            tags.append("pdf-doc-differ")
    return list(dict.fromkeys(tags))


def enrich_page_nodes(
    page_nodes: list[DKGNode],
    section_nodes: list[DKGNode],
) -> None:
    """Set pdf_page, document_page, page_tags; fill sparse page text from parent section."""
    section_by_pdf: dict[int, DKGNode] = {}
    for sec in section_nodes:
        for pno in range(sec.page_start or 1, (sec.page_end or sec.page_start or 1) + 1):
            prev = section_by_pdf.get(pno)
            if prev is None or sec.depth >= prev.depth:
                section_by_pdf[pno] = sec

    for page in page_nodes:
        pdf_page = page.page_start or page.order
        page.pdf_page = pdf_page
        page.order = pdf_page
        page.page_start = pdf_page
        page.page_end = pdf_page

        raw_text = (page.text or "").strip()
        doc_label = detect_document_page_label(raw_text)
        page.document_page = doc_label

        if not raw_text or len(raw_text) < 40:
            page.text = _fill_missing_page_text(page, section_by_pdf.get(pdf_page))

        page.page_tags = build_page_tags(pdf_page, page.document_page)

        doc_disp = page.document_page or str(pdf_page)
        if page.document_page and not document_page_matches_pdf(page.document_page, pdf_page):
            page.title = f"Page {doc_disp} (PDF {pdf_page})"
        else:
            page.title = f"Page {pdf_page}"


def _fill_missing_page_text(page: DKGNode, section: Optional[DKGNode]) -> str:
    """
    When a page has little/no extracted text, attach context from the spanning section.
    """
    existing = (page.text or "").strip()
    if existing and len(existing) >= 40:
        return existing

    parts: list[str] = []
    if section:
        parts.append(f"[Section context — {section.title}]")
        sec_body = (section.text or "").strip()
        if sec_body:
            snippet = sec_body[:2000]
            if len(sec_body) > 2000:
                snippet += "\n…"
            parts.append(snippet)
        else:
            parts.append("(Section has no body text in graph.)")
    else:
        parts.append("[No section mapped to this PDF page.]")

    if existing:
        parts.append(f"[Sparse page extract]\n{existing}")

    parts.append(
        "[Note: This PDF page had little extractable text; use visual_content if present "
        "or ask about the printed page number shown in the page title.]"
    )
    return "\n\n".join(parts)


def parse_page_number_from_query(query: str) -> tuple[Optional[int], Optional[str]]:
    """
    Extract page reference from user question.
    Returns (pdf_page_int_or_none, document_page_label_or_none).
    Prefers explicit 'pdf page N' vs bare 'page N' (treated as document page label).
    """
    q = query.lower()
    pdf_m = re.search(r"\bpdf\s+(?:\w+\s+){0,3}page\s+(\d+)\b", q, re.I)
    if pdf_m:
        return int(pdf_m.group(1)), None

    pdf_m2 = re.search(r"\bpdf\s+page\s+(\d+)\b", q, re.I)
    if pdf_m2:
        return int(pdf_m2.group(1)), None

    pdf_m3 = re.search(r"\(\s*pdf\s+page\s+(\d+)\s*\)", q, re.I)
    if pdf_m3:
        return int(pdf_m3.group(1)), None

    page_m = re.search(
        r"\b(?:page|p\.?|pg\.?)\s+([a-zA-Z0-9ivxlcdm\-]+)\b",
        query,
        re.I,
    )
    if page_m:
        label = page_m.group(1).strip()
        if not is_valid_document_page_label(label):
            return None, None
        if label.isdigit():
            return None, label
        return None, label

    return None, None


# Words that follow "page X" but are not printed page labels (e.g. "pdf page data")
_PAGE_LABEL_STOPWORDS = frozenset({
    "data", "content", "text", "image", "info", "information", "details",
    "number", "num", "index", "sheet", "pdf", "form", "whole", "fetch",
    "from", "where", "with", "that", "this", "about", "showing", "shows",
})


def is_valid_document_page_label(label: Optional[str]) -> bool:
    if not label or not str(label).strip():
        return False
    low = str(label).strip().lower()
    if low in _PAGE_LABEL_STOPWORDS:
        return False
    if len(low) > 12:
        return False
    return True
