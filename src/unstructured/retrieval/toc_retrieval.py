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
    # Ratio alone rewards a short page with one coincidental hit exactly as
    # much as a long page with dozens of genuine hits -- a 3-line title
    # page where a sentence happens to end in a year ("...December 31,
    # 2025") gets the same 33% ratio as one real entry out of three lines,
    # which is enough noise to beat a real TOC's 23-entry signal once it's
    # diluted across a long page. Scale by how much ABSOLUTE signal there
    # is, not just its share of the page, so a couple of coincidental
    # matches can't outweigh overwhelming genuine structure.
    confidence = min(1.0, signal / 3.0)

    score = 0.0
    if _TOC_PAGE_HEADING_RE.search(body[:600]):
        score += 0.45 * min(1.0, ratio * 6) * confidence
    score += min(0.7, ratio * 1.6) * confidence
    # TOC pages are usually short lists, not long prose.
    if len(lines) <= 120 and len(body) < 14000:
        score += 0.05
    return min(1.0, score)


TOC_PAGE_SCORE_FLOOR = 0.42


# Words that describe the REQUEST rather than the subject. "What is the table
# of contents of the financial statements?" is about financial statements; the
# rest is how the question was phrased, and matching on it would anchor every
# TOC question to whichever page says "contents" loudest.
_TOC_REQUEST_WORDS = frozenset(
    "table tables contents content index list show tell give what which where "
    "sections section chapter chapters part parts document report pdf file the "
    "for from about please me my this that these those does have has are was".split()
)


def toc_subject_terms(query: str) -> list[str]:
    """The words that say WHICH part of a document a TOC question is about.

    Numbers are kept and kept attached: "Item 8" has to stay distinguishable
    from "Item 80", and a bare "8" is worthless on its own. Everything that
    merely phrases the request is dropped, so a question naming no subject
    yields nothing and the document-level TOC stands.
    """
    tokens = re.findall(r"[A-Za-z]+|\d+", query or "")
    out: list[str] = []
    for i, tok in enumerate(tokens):
        low = tok.lower()
        if low.isdigit():
            # attach a number to the word before it: "item 8" -> "item 8"
            if out and not out[-1][-1].isdigit():
                out[-1] = "%s %s" % (out[-1], low)
            continue
        if low in _TOC_REQUEST_WORDS or len(low) < 3:
            continue
        out.append(low)
    return out


def score_page_for_subject(text: str, terms: list[str]) -> float:
    """How much of the question's subject a page actually contains.

    Fraction of subject terms present, so a long page is not rewarded for
    length. Used to locate the part of a document a TOC question is about
    WITHOUT relying on extracted headings -- on a filing whose heading
    detection produces "Preamble", "Smith Street" and "Dry", the headings
    cannot anchor anything, but the body text still says "Item 8".
    """
    if not terms:
        return 0.0
    low = (text or "").lower()
    if not low.strip():
        return 0.0
    return sum(1 for t in terms if t in low) / len(terms)


def pick_subject_page(
    pages: list[tuple[Optional[int], str]], terms: list[str], *, min_score: float = 0.6
) -> Optional[int]:
    """Page the question is about, or None when it names no subject.

    Below `min_score` the question is general ("what is the table of
    contents?") and must keep the document-level answer rather than being
    sent somewhere arbitrary. Ties go to the EARLIEST page: a subject is
    introduced before it is referred to again.
    """
    best_page, best, best_hits = None, 0.0, 0
    for page, text in pages:
        if page is None:
            continue
        score = score_page_for_subject(text, terms)
        if score < best:
            continue
        # Coverage first, then how often the subject is actually discussed.
        # A phrase like "financial statements" appears once on a cover page and
        # dozens of times in the section about it; taking the earliest page
        # with full coverage anchored on the cover and sent every question back
        # to the front of the document.
        low = (text or "").lower()
        hits = sum(low.count(t) for t in terms)
        if score > best or hits > best_hits:
            best_page, best, best_hits = page, score, hits
    return best_page if best >= min_score else None


# A chapter's table of contents lists a handful of entries; a document's lists
# dozens. The score rewards density of "title .... page" lines, so the two are
# not on the same scale, and one floor cannot serve both: at 0.42 the Chevron
# 10-K admits its front TOC (0.57) and its exhibit index (1.00) while dropping
# the six chapter TOCs between them, which score 0.30-0.41.
#
# The floor exists to keep prose out. When a question anchors on a place in the
# document, that anchor does the disambiguating -- a false positive far from it
# cannot win -- so the bar can come down to admit local TOCs. Unanchored
# questions keep the strict floor, because nothing else is holding them back.
TOC_PAGE_SCORE_FLOOR_ANCHORED = 0.28


def stitch_toc_run(
    scored_pages: list[tuple[dict, float]], near: Optional[int] = None
) -> Optional[dict]:
    """The whole table of contents, not just its best-scoring page.

    A TOC routinely runs over two or three pages, and only one page could
    ever be returned before this -- so a question about the contents of a
    document silently got a third of the answer, with nothing to indicate
    the rest existed.

    Pages above the floor that are ADJACENT to the best one are joined, and
    the walk stops at the first gap. Contiguity is what separates one TOC
    from another: a chapter-wise TOC deeper in the book is a different run,
    and merging it into the document-level one would misrepresent both. When
    several runs exist the FIRST is returned -- that is the document-level
    TOC -- unless `near` points at one, and the run's earliest page is
    reported as its location.
    """
    floor = TOC_PAGE_SCORE_FLOOR_ANCHORED if near is not None else TOC_PAGE_SCORE_FLOOR
    by_key = {
        int(page.get("sort_key") or 0): page
        for page, score in scored_pages
        if score > floor
    }
    if not by_key:
        return None

    runs: list[list[int]] = []
    for key in sorted(by_key):
        if runs and key == runs[-1][-1] + 1:
            runs[-1].append(key)
        else:
            runs.append([key])

    # Unscoped questions get the document-level TOC, which is the first run.
    # A chapter's own TOC is only returned when the question points at it
    # (`near`), because picking a deeper one for a general question would
    # answer a question nobody asked.
    if near is None:
        run = runs[0]
    else:
        # A table of contents precedes the material it indexes, so a run that
        # starts at or before the anchor is preferred; distance only decides
        # between those. Without this a section beginning on 165 would take
        # the TOC on 168 -- which indexes the section AFTER it.
        preceding = [r for r in runs if r[0] <= near]
        candidates = preceding or runs
        run = min(candidates, key=lambda r: abs(r[0] - near))
    first = by_key[run[0]]
    return {
        **first,
        "text": "\n\n".join((by_key[k].get("text") or "").strip() for k in run),
        "page_count": len(run),
    }


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
    # depth<=1 only ever matched Chapter-tier nodes (Document=0, Chapter=1,
    # Section=2+ per Axis1StructuralBuilder's convention) -- a document
    # with no chapter tier at all (common for the heuristic/no-TOC parse
    # path; flat SEC filings and short documents alike) has every heading
    # sitting at depth 2, so this silently dropped all of them from the
    # outline instead of falling back to the numbered/uppercase checks
    # below. depth<=2 includes top-level Sections too; deeper numbered
    # subsections (3+) still rely on the checks below, unchanged.
    if depth <= 2:
        return True
    if _NUMBERED_OUTLINE_RE.match(t):
        return True
    if t.isupper() and len(t) < 100 and not _TOC_OUTLINE_SKIP_RE.match(t):
        return True
    return False


def section_title_is_toc(title: str) -> bool:
    return bool(_TOC_SECTION_TITLE_RE.match((title or "").strip()))
