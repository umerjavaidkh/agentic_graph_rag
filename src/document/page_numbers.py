"""
PDF page index vs document (printed) page labels on the page itself.

document_page is stored as text (e.g. "43", "iii", "A") so roman numerals and
annex letters are preserved. User-facing page queries prefer document_page when set.
"""
from __future__ import annotations

import re
from typing import Optional

from ..models import DKGNode, NodeType

# Footer / header line patterns, in priority order. Each is either "strong"
# (self-describing — the text itself signals "this is a page number", via
# a page/p./pg. keyword or an "N / M" divisor) or "weak" (a bare digit,
# roman numeral, or letter with no such signal). See
# detect_document_page_label for why they use different search windows.
#
# The "p."/"pg." abbreviations require a mandatory period: an optional
# period let a bare "p" match the start of any unrelated word ending a
# candidate line — e.g. a repeated running header "COMPLIANCE POLICY" has
# " P" (space then "P") immediately before "OLICY", which the old
# optional-period version greedily captured whole as if "OLICY" were a
# page label.
_STRONG_PAGE_LABEL_PATTERNS = [
    re.compile(r"(?:^|\s)(?:page|p\.|pg\.)\s*[:.]?\s*([a-zA-Z0-9ivxlcdm\-]+)\s*$", re.I),
    # "6 / 12" / "6 of 12" — a page-N-of-M footer with no leading word,
    # common enough (short reports, policies) to warrant its own pattern
    # rather than relying on the word-prefixed one above.
    re.compile(r"^\s*(\d{1,4})\s*(?:/|of)\s*\d{1,4}\s*$", re.I),
]
_WEAK_PAGE_LABEL_PATTERNS = [
    re.compile(r"^\s*([ivxlcdm]+)\s*$", re.I),
    re.compile(r"^\s*([a-z])\s*$", re.I),
    re.compile(r"^\s*(\d{1,4})\s*$"),
    re.compile(r"[-|]\s*(\d{1,4})\s*[-|]\s*$"),
]
_PAGE_LABEL_PATTERNS = _STRONG_PAGE_LABEL_PATTERNS + _WEAK_PAGE_LABEL_PATTERNS

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

    # Strong patterns (self-describing — "page"/"p."/"pg." keyword, or an
    # "N / M" divisor) get a wider window: some layouts put the printed
    # label a few lines after a multi-line repeated running header (title
    # line + document-name line, then the "N / M" label, then body) — a
    # narrow window misses it. Weak patterns (a bare digit, roman numeral,
    # or letter, with no self-describing signal) keep the original narrow
    # edge window — widening it too would pull in unrelated bare numbers
    # from body content that only coincidentally sits near the top/bottom
    # (confirmed: a financial table's "2016" fiscal-year column header
    # landed inside a widened window and got mistaken for a page label).
    wide = list(dict.fromkeys(lines[:6] + lines[-6:]))
    narrow = list(dict.fromkeys(lines[-4:] + lines[:2]))

    for pat in _STRONG_PAGE_LABEL_PATTERNS:
        for line in wide:
            m = pat.search(line)
            if not m:
                continue
            raw = m.group(1).strip()
            if raw and len(raw) <= 12:
                return raw

    for pat in _WEAK_PAGE_LABEL_PATTERNS:
        for line in narrow:
            m = pat.search(line)
            if not m:
                continue
            raw = m.group(1).strip()
            if raw and len(raw) <= 12:
                return raw
    return None


_ROMAN_NUMERALS = {v: k for k, v in _ROMAN_VALUES.items()}

# How far to look for a page that agrees on the offset. Front matter runs are
# short (this report has four roman pages), so a wide window would let the
# main arabic run vouch for a roman one; three is enough to bracket a single
# bad page while staying inside its own run.
_OFFSET_WINDOW = 3


def label_to_number(label: Optional[str]) -> tuple[Optional[int], Optional[str]]:
    """(value, numbering system) for a printed label, or (None, None).

    The system matters as much as the value: a document restarts numbering at
    1 when it moves from roman front matter to arabic body, so "iv" and "4"
    are the same number in different runs and must not vouch for each other.
    """
    if not label:
        return None, None
    text = str(label).strip().lower()
    if text.isdigit():
        return int(text), "arabic"
    if text in _ROMAN_VALUES:
        return _ROMAN_VALUES[text], "roman"
    return None, None


def _format_label(value: int, system: str) -> str:
    if system == "roman" and value in _ROMAN_NUMERALS:
        return _ROMAN_NUMERALS[value]
    return str(value)


def reconcile_page_labels(pages: list[tuple[int, Optional[str]]]) -> dict[int, Optional[str]]:
    """Repair printed page labels using the sequence the document establishes.

    detect_document_page_label reads one page at a time, so anything sitting
    alone on a line near an edge can pass for a page number. In one 52-page
    report that produced three wrong labels out of 52: a decorative drop-cap
    "A" on PDF 9 and "N" on PDF 37 (both matching the bare-letter pattern),
    and the cover's "2021" on PDF 2 (a year matching the bare-digit pattern).
    A citation to "page A" cannot be navigated to and cannot be checked.

    Read across the document instead, the offset between printed label and
    PDF index is constant within a numbering run -- here iv..vii at offset 0
    and 1..44 at offset -7. So:

      * a label whose offset no nearby same-system page shares is not
        believable, and is dropped rather than shown;
      * a page with no usable label that sits BETWEEN two pages agreeing on
        an offset takes that offset -- which is what turns "page A" into
        "page 2".

    Only interpolates between agreeing anchors, never past the ends of a run:
    a document's last page is often deliberately unnumbered, and inventing a
    label there would be a fabricated citation of exactly the kind this whole
    effort exists to prevent.

    `pages` is [(pdf_page, detected_label)]; returns {pdf_page: label or None}.
    """
    parsed: dict[int, tuple[Optional[int], Optional[str]]] = {
        pdf: label_to_number(label) for pdf, label in pages
    }
    offsets: dict[int, tuple[int, str]] = {
        pdf: (value - pdf, system)
        for pdf, (value, system) in parsed.items()
        if value is not None and system is not None
    }

    # Keep only labels an independent neighbour agrees with.
    trusted: dict[int, tuple[int, str]] = {}
    for pdf, (offset, system) in offsets.items():
        for other, (other_offset, other_system) in offsets.items():
            if other == pdf:
                continue
            if (
                abs(other - pdf) <= _OFFSET_WINDOW
                and other_system == system
                and other_offset == offset
            ):
                trusted[pdf] = (offset, system)
                break

    resolved: dict[int, Optional[str]] = {}
    anchors = sorted(trusted)
    for pdf, original in pages:
        if pdf in trusted:
            offset, system = trusted[pdf]
            resolved[pdf] = _format_label(pdf + offset, system)
            continue
        before = [a for a in anchors if a < pdf]
        after = [a for a in anchors if a > pdf]
        if before and after:
            low, high = trusted[before[-1]], trusted[after[0]]
            if low == high:  # same offset AND same system on both sides
                offset, system = low
                resolved[pdf] = _format_label(pdf + offset, system)
                continue
        # Untrusted and unbracketed: report no printed label rather than one
        # that cannot be relied on.
        resolved[pdf] = None
    return resolved


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

    # Detect first, reconcile across the whole document, then apply. Applying
    # each page's own detection directly is what let a drop-cap letter and a
    # cover year stand as page numbers -- nothing checked them against the
    # sequence the rest of the document establishes.
    detected: list[tuple[int, Optional[str]]] = []
    for page in page_nodes:
        pdf_page = page.page_start or page.order
        detected.append((pdf_page, detect_document_page_label((page.text or "").strip())))
    reconciled = reconcile_page_labels(detected)

    for page in page_nodes:
        pdf_page = page.page_start or page.order
        page.pdf_page = pdf_page
        page.order = pdf_page
        page.page_start = pdf_page
        page.page_end = pdf_page

        raw_text = (page.text or "").strip()
        page.document_page = reconciled.get(pdf_page)

        if not raw_text or len(raw_text) < 40:
            page.text = _fill_missing_page_text(page, section_by_pdf.get(pdf_page))

        page.page_tags = build_page_tags(pdf_page, page.document_page)

        doc_disp = page.document_page or str(pdf_page)
        if page.document_page and not document_page_matches_pdf(page.document_page, pdf_page):
            page.title = f"Page {doc_disp} (PDF {pdf_page})"
        else:
            page.title = f"Page {pdf_page}"


def propagate_document_pages(page_nodes: list[DKGNode], other_nodes: list[DKGNode]) -> int:
    """Copy each page's printed label onto everything that starts on it.

    enrich_page_nodes labels Page nodes only, but a citation can land on a
    Region (a figure caption) or a Section just as easily, and those carried
    no printed label at all -- so a Box answer showed "printed page 23" while
    a Figure answer on the very next page showed nothing.

    Keyed on the node's first page, since that is where a reader is being
    sent. Nodes with no page, or on a page whose own label was rejected as
    unreliable, are left alone rather than given a guess.

    Returns how many nodes were labelled, for the ingest log.
    """
    labels = {
        p.pdf_page or p.page_start: p.document_page
        for p in page_nodes
        if p.document_page and (p.pdf_page or p.page_start)
    }
    if not labels:
        return 0
    labelled = 0
    for node in other_nodes:
        if getattr(node, "document_page", None):
            continue
        start = node.page_start or node.pdf_page
        label = labels.get(start)
        if label:
            node.document_page = label
            labelled += 1
    return labelled


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

    # "page 32 of pdf" / "page 32 in the pdf" → PDF file index, not printed page label
    page_of_pdf = re.search(
        r"\bpage\s+(\d+)\s+(?:of|in|from)\s+(?:the\s+)?pdf\b",
        q,
        re.I,
    )
    if page_of_pdf:
        return int(page_of_pdf.group(1)), None

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
            # "page 32 ... pdf" anywhere → treat as PDF page 32 (common phrasing)
            if re.search(r"\bpdf\b", q):
                return int(label), None
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
