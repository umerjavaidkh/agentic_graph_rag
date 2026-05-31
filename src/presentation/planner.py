"""
Build a dynamic presentation payload from answer + sources + retrieval meta.

Block types: markdown, table, chart, image — extensible via BLOCK_BUILDERS.
"""
from __future__ import annotations

import re
from typing import Any, Callable, Optional

from ..assets.page_images import resolve_image_url
from ..document.page_numbers import parse_page_number_from_query
from .structured_planner import build_structured_presentation
from ..document.page_vision import compact_visual_content
from ..unstructured.visual_retrieval import wants_page_text

# ── Intent detectors (extensible) ─────────────────────────────

_IMAGE_QUERY = re.compile(
    r"\b(?:show|display|see|fetch|get)\s+(?:the\s+)?(?:image|picture|photo|figure|page|pdf)\b|"
    r"\b(?:show\s+all|all|every|list|each)\s+(?:\w+\s+){0,3}"
    r"(?:images?|figures?|figs?\.?|photos?|pictures?|visuals?)\b|"
    r"\b(?:image|picture|photo|screenshot|figure)\s+(?:of|from|on)\b|"
    r"\bshow\s+page\b|\bsee\s+page\b|\bdisplay\s+page\b|"
    r"\bwhole\s+page\b|\bfull\s+page\b|\bentire\s+page\b|"
    r"\bpdf\s+page\s+\d+"
    r"|\b(?:logo|icon)\s+(?:only|image)\b|\b(?:only|just)\s+(?:the\s+)?(?:logo|icon)\b",
    re.I,
)
_TEXT_ONLY = re.compile(
    r"\b(?:text\s+only|only\s+text|no\s+image|without\s+image|don'?t\s+show\s+image)\b",
    re.I,
)
_PERCENT_PATTERN = re.compile(
    r"(\d+(?:\.\d+)?)\s*%|(\d+(?:\.\d+)?)\s*percent",
    re.I,
)
_PIPE_TABLE_ROW = re.compile(r"^\s*\|.+\|\s*$")


def wants_page_image(question: str) -> bool:
    if _TEXT_ONLY.search(question):
        return False
    return bool(_IMAGE_QUERY.search(question))


def _extract_markdown_tables(text: str) -> list[dict]:
    """Parse GitHub-style pipe tables from answer text."""
    tables: list[dict] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        if not _PIPE_TABLE_ROW.match(lines[i]):
            i += 1
            continue
        block: list[str] = []
        while i < len(lines) and _PIPE_TABLE_ROW.match(lines[i]):
            block.append(lines[i])
            i += 1
        if len(block) < 2:
            continue
        rows_raw = [[c.strip() for c in ln.strip("|").split("|")] for ln in block]
        sep_idx = None
        for j, row in enumerate(rows_raw):
            if all(re.match(r"^:?-+:?$", c) for c in row if c):
                sep_idx = j
                break
        if sep_idx == 0 and len(rows_raw) > 2:
            headers = rows_raw[0]
            data_rows = [r for k, r in enumerate(rows_raw) if k != 0 and k != sep_idx]
        else:
            headers = rows_raw[0]
            data_rows = rows_raw[1:]
        if headers and data_rows:
            tables.append({"headers": headers, "rows": data_rows})
    return tables


def _extract_chart_from_text(text: str) -> Optional[dict]:
    matches = _PERCENT_PATTERN.findall(text)
    values: list[float] = []
    for a, b in matches:
        raw = a or b
        if raw:
            values.append(float(raw))
    if len(values) < 2:
        return None
    labels = [f"Item {i + 1}" for i in range(len(values))]
    return {
        "chartType": "bar",
        "labels": labels[:12],
        "values": values[:12],
        "title": "Values from answer",
    }


def _image_blocks_from_sources(
    sources: list[dict],
    question: str,
    force: bool = False,
    retrieved_context: Optional[dict] = None,
    query_type: Optional[str] = None,
) -> list[dict]:
    blocks: list[dict] = []
    want = force or wants_page_image(question)
    if not want:
        return blocks

    ctx = retrieved_context or {}
    mode = ctx.get("mode") or ""
    page_text_mode = mode == "page_text"
    pin_pdf = ctx.get("pdf_page")
    list_all_visuals = bool(
        re.search(
            r"\b(?:show\s+all|all|every|list|each)\s+(?:\w+\s+){0,3}"
            r"(?:images?|figures?|figs?\.?|photos?|pictures?|visuals?)\b",
            question,
            re.I,
        )
    )
    single_page_image = (
        ctx.get("single_visual")
        or (
            mode != "page_visual_list"
            and not list_all_visuals
            and (mode == "page_lookup" or query_type == "page")
            and wants_page_image(question)
            and not page_text_mode
            and not wants_page_text(question)
        )
    )
    if pin_pdf is None and single_page_image:
        pin_pdf, _ = parse_page_number_from_query(question)

    seen_keys: set[str] = set()
    seen_urls: set[str] = set()
    for src in sources:
        if pin_pdf is not None:
            sp = src.get("pdf_page")
            if sp is not None and int(sp) != int(pin_pdf):
                continue
        key = src.get("image_key")
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        url = src.get("image_url") or resolve_image_url(key)
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        title = src.get("title") or "Page"
        doc_p = src.get("document_page")
        pdf_p = src.get("pdf_page")
        kind = src.get("region_kind")
        caption = title
        if kind:
            caption = f"{title} ({kind})"
        if doc_p and str(doc_p) != str(pdf_p):
            caption = f"{caption} — printed {doc_p}, PDF {pdf_p}"
        elif pdf_p:
            caption = f"{caption} — PDF page {pdf_p}"
        blocks.append({
            "type": "image",
            "url": url,
            "alt": caption,
            "caption": caption,
        })
        if single_page_image:
            break
    return blocks


def _short_visual_blurb(src: dict) -> str:
    """One-line summary for list UI — not the full vision dump."""
    visual = compact_visual_content((src.get("visual_content") or "").strip())
    blob = visual.lower()
    if "logo" in blob or "brand" in blob or "emblem" in blob:
        return "Logo / brand mark"
    if "diagram" in blob or "flowchart" in blob:
        return "Diagram"
    if "chart" in blob or "graph" in blob:
        return "Chart"
    if "table" in blob:
        return "Table"
    kind = (src.get("region_kind") or "figure").strip()
    if kind:
        return kind.replace("_", " ").title()
    first = visual.split("\n", 1)[0].strip()
    if len(first) > 100:
        first = first[:97] + "…"
    return first or "Figure"


def _page_visual_list_blocks(
    sources: list[dict],
    ctx: dict,
) -> list[dict]:
    """Interleave compact captions with one image per figure (no duplicate wall of text)."""
    blocks: list[dict] = []
    pdf_p = ctx.get("pdf_page")
    doc_p = ctx.get("document_page")
    items = [s for s in sources if s.get("image_key")]
    if not items:
        return blocks

    header = f"**Visuals on PDF page {pdf_p}** ({len(items)} found)"
    if doc_p and pdf_p and str(doc_p) != str(pdf_p):
        header += f"\n\n_Printed page **{doc_p}**._"
    blocks.append({"type": "markdown", "content": header})

    seen_keys: set[str] = set()
    for i, src in enumerate(items, 1):
        key = src.get("image_key")
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)
        url = resolve_image_url(key)
        if not url:
            continue
        title = src.get("title") or f"Figure {i}"
        blurb = _short_visual_blurb(src)
        blocks.append({
            "type": "markdown",
            "content": f"\n{i}. **{title}** — _{blurb}_",
        })
        doc_page = src.get("document_page")
        pdf_page = src.get("pdf_page") or pdf_p
        kind = src.get("region_kind")
        caption = title
        if kind:
            caption = f"{title} ({kind})"
        if doc_page and str(doc_page) != str(pdf_page):
            caption = f"{caption} — printed {doc_page}, PDF {pdf_page}"
        elif pdf_page:
            caption = f"{caption} — PDF page {pdf_page}"
        blocks.append({
            "type": "image",
            "url": url,
            "alt": caption,
            "caption": caption,
        })
    return blocks


def _table_blocks_from_answer(answer: str) -> list[dict]:
    blocks: list[dict] = []
    for idx, tbl in enumerate(_extract_markdown_tables(answer)):
        blocks.append({
            "type": "table",
            "title": f"Table {idx + 1}" if idx else None,
            "headers": tbl["headers"],
            "rows": tbl["rows"],
        })
    return blocks


def _chart_blocks_from_answer(answer: str) -> list[dict]:
    chart = _extract_chart_from_text(answer)
    if not chart:
        return []
    return [{"type": "chart", **chart}]


def _markdown_block(answer: str, tables_found: bool) -> dict:
    text = answer
    if tables_found:
        lines = []
        for ln in answer.splitlines():
            if _PIPE_TABLE_ROW.match(ln):
                continue
            if ln.strip().startswith("|---"):
                continue
            lines.append(ln)
        text = "\n".join(lines).strip()
    return {"type": "markdown", "content": text or answer}


def _has_tabular_sources(sources: list[dict]) -> bool:
    return sum(1 for s in sources if isinstance(s.get("raw"), dict)) >= 2


def build_presentation(
    question: str,
    answer: str,
    sources: list[dict],
    retrieved_context: Optional[dict] = None,
    query_type: Optional[str] = None,
    agent: Optional[str] = None,
) -> dict:
    """
    Returns { kind, blocks } for the chat UI.
    """
    if agent in ("structured",) or _has_tabular_sources(sources):
        structured = build_structured_presentation(question, answer, sources)
        if structured:
            return structured

    ctx = retrieved_context or {}
    mode = ctx.get("mode") or ""
    blocks: list[dict] = []

    if mode == "page_visual_list":
        blocks = _page_visual_list_blocks(sources, ctx)
        kinds = {b["type"] for b in blocks}
        return {
            "kind": "mixed" if len(kinds) > 1 else (next(iter(kinds)) if kinds else "plain"),
            "blocks": blocks,
        }

    text_only = bool(_TEXT_ONLY.search(question)) or wants_page_text(question)
    page_text_mode = mode == "page_text"
    visual_query_type = query_type in ("page", "visual_scene", "figure_caption")
    visual_mode = mode in (
        "unified_visual", "page_lookup", "page_visual_list",
        "visual_scene", "caption_figure",
    )
    wants_visual = wants_page_image(question)
    force_image = not text_only and not page_text_mode and wants_visual

    image_blocks = _image_blocks_from_sources(
        sources,
        question,
        force=force_image,
        retrieved_context=ctx,
        query_type=query_type,
    )
    blocks.extend(image_blocks)

    table_blocks = _table_blocks_from_answer(answer)
    blocks.extend(table_blocks)

    # Avoid generic "percent from prose" charts for structured/tabular answers.
    if agent != "structured" and not _has_tabular_sources(sources):
        chart_blocks = _chart_blocks_from_answer(answer)
        if chart_blocks and not table_blocks:
            blocks.extend(chart_blocks)

    blocks.append(_markdown_block(answer, bool(table_blocks)))

    kinds = {b["type"] for b in blocks}
    if len(kinds) > 1:
        kind = "mixed"
    elif kinds:
        kind = next(iter(kinds))
    else:
        kind = "plain"

    return {"kind": kind, "blocks": blocks}
