"""
Universal visual retrieval — one pipeline for page number, caption, scene, and list-all.

Gathers Page + Region candidates, scores by dynamic phrases (vision text weighted
higher than printed caption text), optional embedding rerank, returns image-bearing chunks.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..document.page_numbers import (
    is_valid_document_page_label,
    parse_page_number_from_query,
)
from ..document.page_vision import compact_visual_content
from ..document.patterns import TABLE_REF_PATTERN

# Boilerplate stripped from queries before phrase extraction
_QUERY_TAIL = re.compile(
    r"\s*(?:search|find|show|fetch|get|locate|display|see)\s+"
    r"(?:for\s+)?(?:that\s+)?(?:the\s+)?(?:figure|fig\.?|table|image|photo|page).*",
    re.I,
)
_DOC_REF = re.compile(
    r"\s*(?:in|from)\s+(?:the\s+)?(?:go\.?data|godata|document)?\s*pdf\.?",
    re.I,
)
_LIST_ALL = re.compile(r"\b(all|every|each|list|show\s+all)\b", re.I)
_VISUAL_KIND = re.compile(
    r"\b(figures?|figs?\.?|tables?|charts?|diagrams?|images?|photos?|pictures?)\b",
    re.I,
)
_WANTS_IMAGE = re.compile(
    r"\b(?:show|display|see|fetch|get)\s+(?:the\s+)?(?:image|picture|photo|figure|page|pdf)\b|"
    r"\b(?:image|picture|photo|screenshot)\s+(?:of|from|on)\b|"
    r"\bwhole\s+page\b|\bfull\s+page\b|\bentire\s+page\b|"
    r"\bpdf\s+page\s+\d+\b|\bpage\s+image\b",
    re.I,
)
_VISUAL_FOCUS_TERM = re.compile(
    r"\b(logos?|icons?|emblems?|brand\s*marks?|badges?|seals?)\b",
    re.I,
)
_ONLY_VISUAL = re.compile(r"\b(?:only|just)\b", re.I)
_DIAGRAM_HINTS = re.compile(
    r"\b(diagram|flowchart|overview\s+diagram|boxes?/nodes?|arrows?)\b",
    re.I,
)

_GENERIC_TERMS = frozenset({
    "ministry", "health", "social", "public", "workers", "assistance", "center",
    "centre", "office", "department", "services", "service", "national", "data",
    "document", "report", "annual", "figure", "table", "image", "page", "pdf",
})


@dataclass
class VisualIntent:
    """Parsed intent for any visual / page / figure query."""

    query: str
    pdf_page: Optional[int] = None
    document_page: Optional[str] = None
    list_all: bool = False
    wants_image: bool = False
    kind_filter: Optional[str] = None  # figure | table
    phrases: list[str] = field(default_factory=list)
    terms: list[str] = field(default_factory=list)
    visual_focus: list[str] = field(default_factory=list)  # e.g. logo
    single_visual: bool = False  # return one crop, not a page listing


def wants_page_text(query: str) -> bool:
    """User wants readable page text, not a figure crop or image."""
    q = query.lower()
    if re.search(r"\b(?:text\s+only|only\s+text|no\s+image|without\s+image)\b", q):
        return True
    if re.search(r"\b(?:image|picture|photo|screenshot|figure\s+from)\b", q) and not re.search(
        r"\b(?:all|full|entire)\s+(?:the\s+)?text\b|\btext\s+(?:from|on|of)\b", q
    ):
        return False
    return bool(
        re.search(
            r"\b(?:all|full|entire|complete)\s+(?:the\s+)?(?:text|content|words)\b"
            r"|\b(?:text|content|words)\s+(?:from|on|of)\s+(?:the\s+)?(?:pdf\s+)?page\b"
            r"|\bgive\s+me\s+(?:all\s+)?(?:the\s+)?text\b"
            r"|\bwhat\s+(?:does|is)\s+(?:pdf\s+)?page\s+\d+\s+say\b",
            q,
        )
    )


def is_strict_page_lookup(intent: VisualIntent) -> bool:
    """
    User asked for a specific page (e.g. "image from page 32") — not a document-wide caption search.
    """
    if wants_page_text(intent.query):
        return False
    if intent.list_all and not intent.single_visual:
        return False
    if intent.pdf_page is None and not intent.document_page:
        return False
    for phrase in intent.phrases:
        if len(phrase) >= 50:
            return False
    return True


def normalize_visual_page_intent(intent: VisualIntent) -> None:
    """
    Bare 'page N' in image/figure queries means PDF page index, not printed label.

    Must run after pdf_page / document_page are set on the intent (including from
    retrieve_node), otherwise multi-document clarification never sees pdf_page=N.
    """
    if intent.pdf_page is not None or not intent.document_page:
        return
    label = str(intent.document_page).strip()
    if not label.isdigit():
        return
    q = intent.query.lower()
    if not (intent.wants_image or intent.list_all or _VISUAL_KIND.search(q)):
        return
    intent.pdf_page = int(label)
    intent.document_page = None


def parse_visual_intent(
    query: str,
    extract_terms: Optional[Callable[[str], list[str]]] = None,
) -> VisualIntent:
    pdf_p, doc_p = parse_page_number_from_query(query)
    if doc_p and not is_valid_document_page_label(doc_p):
        doc_p = None

    q = query.lower()
    list_all = bool(_LIST_ALL.search(q) and _VISUAL_KIND.search(q))

    kind_filter = None
    if re.search(r"\b(figures?|figs?\.?|photos?|pictures?|images?)\b", q, re.I):
        kind_filter = "figure"
    elif re.search(r"\btables?\b", q, re.I):
        kind_filter = "table"

    wants_image = bool(_WANTS_IMAGE.search(q)) or list_all

    phrases, terms = extract_search_phrases(query)
    if extract_terms:
        for t in extract_terms(query):
            if t not in terms and len(t) > 2:
                terms.append(t)

    visual_focus = extract_visual_focus_terms(query)
    single_visual = bool(visual_focus) and (
        bool(_ONLY_VISUAL.search(q))
        or bool(re.search(r"\b(?:logo|icon)\s+(?:only|image)\b", q, re.I))
        or bool(re.search(r"\b(?:only|just)\s+(?:the\s+)?(?:logo|icon)\b", q, re.I))
    )
    if single_visual:
        list_all = False
        wants_image = True
        for ft in visual_focus:
            if ft not in terms:
                terms.append(ft)

    intent = VisualIntent(
        query=query,
        pdf_page=pdf_p,
        document_page=doc_p,
        list_all=list_all,
        wants_image=wants_image,
        kind_filter=kind_filter,
        phrases=phrases,
        terms=terms,
        visual_focus=visual_focus,
        single_visual=single_visual,
    )
    normalize_visual_page_intent(intent)
    return intent


def extract_visual_focus_terms(query: str) -> list[str]:
    """Logo/icon/emblem when user asks for a specific visual element only."""
    seen: list[str] = []
    for m in _VISUAL_FOCUS_TERM.finditer(query):
        raw = m.group(1).lower().replace(" ", "")
        norm = raw.rstrip("s") if raw.endswith("s") and len(raw) > 4 else raw
        if norm == "brandmark":
            norm = "logo"
        if norm not in seen:
            seen.append(norm)
    return seen


def extract_search_phrases(query: str) -> tuple[list[str], list[str]]:
    """Dynamic phrases (substring) + terms (Neo4j needles) from any query."""
    core = _QUERY_TAIL.sub("", query).strip()
    core = _DOC_REF.sub("", core).strip()
    core = re.sub(r"\s+", " ", core)

    phrases: list[str] = []
    if len(core) >= 20:
        phrases.append(core.lower()[:220])

    words = [
        w
        for w in re.findall(r"[a-z0-9]{3,}", core.lower())
        if w not in _GENERIC_TERMS and len(w) > 2
    ]
    # Multi-word phrases (longest first)
    for n in (5, 4, 3):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i : i + n])
            if len(phrase) >= 14 and phrase not in phrases:
                phrases.append(phrase)

    for ref in TABLE_REF_PATTERN.findall(query):
        phrases.append(f"table {ref.lower()}")

    terms = list(dict.fromkeys(words))[:14]
    for p in phrases:
        for w in p.split():
            if w not in terms and len(w) > 3:
                terms.append(w)

    return list(dict.fromkeys(phrases))[:12], terms[:14]


def node_blobs(row: dict) -> dict[str, str]:
    visual = compact_visual_content((row.get("visual_content") or "").strip())
    text = (row.get("text") or "").strip()
    title = (row.get("title") or "").strip()
    tags = row.get("region_tags") or []
    tag_s = " ".join(tags) if isinstance(tags, list) else str(tags)
    return {
        "visual": visual.lower(),
        "text": text.lower(),
        "title": title.lower(),
        "tags": tag_s.lower(),
        "combined": f"{visual} {text} {title} {tag_s}".strip(),
    }


def score_visual_candidate(
    intent: VisualIntent,
    row: dict,
    *,
    node_label: str = "Page",
) -> float:
    """
    Higher weight on visual_content (what the photo shows) than text (printed caption).
    """
    blobs = node_blobs(row)
    score = 0.0

    pdf_p = row.get("pdf_page") or row.get("doc_order")
    if intent.pdf_page is not None and pdf_p == intent.pdf_page:
        score += 1000.0
    if intent.document_page and str(row.get("document_page", "")).lower() == intent.document_page.lower():
        score += 800.0

    for phrase in intent.phrases:
        if not phrase:
            continue
        if phrase in blobs["visual"]:
            score += 60.0 + min(len(phrase), 80) * 0.5
        elif phrase in blobs["text"]:
            score += 25.0
        elif phrase in blobs["combined"]:
            score += 12.0

    for term in intent.terms:
        if term in blobs["visual"]:
            score += 10.0
        elif term in blobs["text"]:
            score += 4.0
        elif term in blobs["tags"]:
            score += 6.0

    if intent.wants_image and row.get("image_key"):
        score += 40.0
    if node_label == "Region" and row.get("region_kind") == intent.kind_filter:
        score += 25.0
    if intent.kind_filter and row.get("region_kind") == intent.kind_filter:
        score += 15.0

    for term in intent.visual_focus:
        t = term.lower()
        if t in blobs["visual"]:
            score += 120.0
        elif t in blobs["tags"]:
            score += 70.0
        elif t in blobs["title"]:
            score += 45.0
        elif t in blobs["text"]:
            score += 18.0
        if t in ("logo", "icon", "emblem") and _DIAGRAM_HINTS.search(blobs["visual"]):
            if t not in blobs["visual"] and t not in blobs["tags"]:
                score -= 55.0

    return score


def best_region_for_visual_focus(
    regions: list[dict],
    intent: VisualIntent,
    page_visual: Optional[str] = None,
) -> Optional[dict]:
    """Pick the region crop that best matches logo/icon-only requests."""
    if not intent.visual_focus:
        return None
    scored: list[tuple[float, dict]] = []
    for reg in regions:
        if not reg.get("image_key"):
            continue
        row = dict(reg)
        if not row.get("visual_content") and page_visual:
            row["visual_content"] = page_visual
        s = score_visual_candidate(intent, row, node_label="Region")
        scored.append((s, reg))
    if not scored:
        return None
    scored.sort(key=lambda x: (-x[0], x[1].get("doc_order", 0)))
    return scored[0][1]


def display_text_for_chunk(row: dict) -> str:
    """Text for UI/LLM without duplicating embedded vision wrapper."""
    visual = compact_visual_content((row.get("visual_content") or "").strip())
    raw = (row.get("text") or "").strip()
    if raw.startswith("[Visual page content"):
        if "[Extracted text]" in raw:
            raw = raw.split("[Extracted text]", 1)[-1].strip()
        else:
            raw = ""
    raw = re.sub(r"\[No section mapped to this PDF page\.\]", "", raw).strip()
    raw = re.sub(
        r"\[Note: This PDF page had little extractable text[^\]]*\]",
        "",
        raw,
        flags=re.I,
    ).strip()
    parts = []
    if visual:
        parts.append(visual)
    if raw and raw not in parts:
        parts.append(raw)
    return "\n\n".join(parts).strip() or (row.get("title") or "")
