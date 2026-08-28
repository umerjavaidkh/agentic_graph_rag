"""region_description.py — say what a table is, so it can be found by asking.

A table arrives from the parser as a pipe grid and was stored, embedded
and titled from exactly that:

    title:       "|  | Interest Expenses . . . . . . . 45<br>Bond Premium"
    search_text: "| Publication | 1099-DIV<br>1099-INT Interest Income |\\n|---|---|..."

Both problems are visible in that one line. The title is the grid's first
row, because it was taken as the first line of the text. And the text a
reader would match against -- "the table of tiers", "which table lists
interest expenses" -- is nowhere in it: what is embedded is punctuation,
`<br>` markers and column padding.

So these nodes were fully embedded and still effectively unsearchable,
which is why every recall miss was a Region. The fix belongs at ingest
rather than in retrieval: no query rewriting can recover a caption that
was never stored.

Deliberately deterministic -- no model call. There are 12,818 of these,
descriptions must be rebuildable on every ingest, and a caption plus the
column names is most of what a question actually matches on. What is
written here is only what the page already says.
"""
from __future__ import annotations

import re

# "Table 3.2. Tiers", "Figure 1: CSF Core", "Fig. 4 -- Overview", "Box 9 ..."
_CAPTION = re.compile(
    r"^\s*(table|figure|fig\.?|box|exhibit)\s*"
    r"(\d+(?:[.\-]\d+)*)\s*[.:\-–—]?\s+(\S.{2,150}?)\s*$",
    re.I | re.M,
)
_KIND_WORDS = {
    "table": {"table", "exhibit", "box"},
    "figure": {"figure", "fig", "fig.", "exhibit"},
}

_MAX_COLUMNS_LISTED = 8
_MAX_VALUES_LISTED = 6
_MAX_DESCRIPTION_CHARS = 900


def captions_on_page(page_text: str, kind: str) -> list[str]:
    """Caption lines on this page, in reading order, for one kind of region.

    Captions sit in the page's prose, not inside the region the parser cut
    out, so they have to be recovered from around it.
    """
    wanted = _KIND_WORDS.get(kind, set())
    out: list[str] = []
    for m in _CAPTION.finditer(page_text or ""):
        word = m.group(1).lower().rstrip(".")
        if word not in wanted and f"{word}." not in wanted:
            continue
        rest = m.group(3).strip()
        if not _looks_like_caption(rest):
            continue
        label = m.group(1).strip().rstrip(".").title()
        out.append(f"{label} {m.group(2)}. {rest}")
    return out


def _looks_like_caption(rest: str) -> bool:
    """Whether the text after "Table 3." is a caption or just the page's prose.

    Documents refer to their own tables in body text ("...as shown in Table
    1-1. 4. There are changes in the tax law that af-"), and taking the rest
    of that line produced titles like the one above. A caption is a short
    noun phrase; these three checks reject what the body text does and a
    caption does not.
    """
    if len(rest) > 120:
        return False
    if rest.endswith("-"):
        return False  # hyphenated line break: mid-sentence, not a title
    if re.match(r"^\d+[.)]", rest):
        return False  # list numbering picked up after the table's number
    return bool(re.match(r"^[\"\u201c(A-Z]", rest))


def parse_grid(text: str) -> tuple[list[str], list[list[str]]]:
    """Header cells and body rows of a markdown pipe grid, if it is one."""
    rows: list[list[str]] = []
    for line in (text or "").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.replace("<br>", " ").strip() for c in line.strip("|").split("|")]
        if all(set(c) <= set("-: ") for c in cells if c):
            continue  # markdown separator row
        rows.append(cells)
    if not rows:
        return [], []
    header = [c for c in rows[0]]
    return header, rows[1:]


def describe(kind: str, text: str, caption: str = "", *, index: int = 0,
             page: int = 0) -> str:
    """One paragraph a person could have written about this region.

    Shaped around what questions actually contain: the caption, the column
    names, and a few real values -- "which table lists the tiers", "the
    table with Nodes and Server columns", "the row for Tier 3".
    """
    label = "Table" if kind == "table" else "Figure"
    parts: list[str] = []
    if caption:
        parts.append(caption if caption.endswith(".") else caption + ".")
    else:
        parts.append(f"{label} {index} on page {page}." if index else f"{label} on page {page}.")

    header, body = parse_grid(text)
    named = [h for h in header if h]
    if named:
        cols = ", ".join(named[:_MAX_COLUMNS_LISTED])
        more = "" if len(named) <= _MAX_COLUMNS_LISTED else f", and {len(named) - _MAX_COLUMNS_LISTED} more"
        parts.append(
            f"{label} with {len(header)} columns ({cols}{more}) and {len(body)} rows."
        )
        first = [r[0] for r in body if r and r[0]][:_MAX_VALUES_LISTED]
        if first:
            parts.append(f"Entries under {named[0]}: " + "; ".join(first) + ".")
    elif kind == "table" and body:
        parts.append(f"{label} with {len(body)} rows.")

    # No grid was parsed. Parsers disagree about regions -- rtldoc emits
    # markdown pipe grids, LightPdfParser emits the region's prose, and a
    # figure never has a grid at all -- so whatever text there is has to be
    # carried through. Dropping it here would replace a table's contents
    # with the sentence "Table 3 on page 12.", which is strictly worse than
    # the raw grid this exists to improve on.
    if not named:
        snippet = " ".join((text or "").split())[:400]
        if snippet and snippet.lower() not in (caption or "").lower():
            parts.append(snippet)

    return " ".join(parts)[:_MAX_DESCRIPTION_CHARS].strip()


def region_title(kind: str, caption: str, index: int, page: int) -> str:
    """The caption when the page gave one, never the grid's first row."""
    if caption:
        return caption[:200]
    label = "Table" if kind == "table" else "Figure"
    return f"{label} {index} (PDF page {page})"
