from langgraph.graph import StateGraph, END
from .state import ESGState
from ..document.page_numbers import (
    is_valid_document_page_label,
    parse_page_number_from_query,
)
from ..document.page_vision import compact_visual_content
from .retriever import DocumentRAGRetriever
from .visual_retrieval import display_text_for_chunk
import re
from typing import List, Optional
from ..config.prompts import load_prompt
from ..config.settings import CHAT_MODEL
from ..model_providers.factory import get_model_provider

retriever = DocumentRAGRetriever()
provider = get_model_provider()

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
MIN_CONFIDENCE_SCORE = 0.30   # below this → low confidence warning to LLM
MIN_CHUNKS_REQUIRED  = 1      # below this → no_context branch

TOC_PHRASES = {
    "table of contents", "full table of contents", "list all sections",
    "document structure", "show all sections", "all sections",
    "list sections", "what sections", "what chapters", "what topics",
    "overview of", "structure of",
    "table of contents form",  # common typo for "from"
}

STRUCTURAL_PHRASES = {
    "what is in chapter", "what is in section", "contents of chapter",
    "contents of section", "show chapter", "show section",
    "summarize chapter", "summarize section",
    "subsection", "sub-section", "sub section", "sections under",
    "headings under", "under the section", "list the sections",
    "with headings", "give them with headings", "children of",
    "nested under", "what are the sub",
    "explain section", "what does section",
}

_NUMBERED_SECTION = re.compile(r"\b\d+(?:\.\d+)+\.?\s+[a-zA-Z]", re.I)
# e.g. "6 IMPLEMENTATION AT STRATEC" (single-level heading without 6.1)
_TOP_LEVEL_SECTION = re.compile(r"\b\d+\s+[A-Za-z]", re.I)
_SECTION_CONTENT = re.compile(
    r"\b(?:what\s+(?:can\s+you\s+)?tell\s+me\s+about|tell\s+me\s+about|explain|describe|summarize)\b",
    re.I,
)

_OVERVIEW_ABOUT = re.compile(
    r"\bwhat\s+(?:is|'?s)\s+.+\s+about\b"
    r"|\b(?:explain|describe|summarize|summary\s+of)\b.+\bdocument\b"
    r"|\bdocument\s+is\s+about\b"
    r"|\bpurpose\s+of\s+(?:the\s+)?(?:go\.?data\s+)?document\b"
    r"|\b(?:\d{1,2}\s+points?|in\s+\d{1,2}\s+points)\b",
    re.I,
)


def is_document_overview_query(question: str) -> bool:
    """Whole-document summary (e.g. what is Go.Data about, explain in 10 points)."""
    q = question.lower()
    if _OVERVIEW_ABOUT.search(question):
        return True
    if re.search(r"\bgo\.?data\b|godata\b", q):
        return any(
            w in q
            for w in (
                "about", "explain", "summary", "overview", "points",
                "purpose", "what is", "what's", "describe",
            )
        )
    return False


def _requested_point_count(question: str) -> int:
    m = re.search(r"\b(\d{1,2})\s+points?\b", question, re.I)
    if m:
        return max(3, min(15, int(m.group(1))))
    m = re.search(r"\bin\s+(\d{1,2})\s+points?\b", question, re.I)
    if m:
        return max(3, min(15, int(m.group(1))))
    return 10


# ─────────────────────────────────────────
# QUERY CLASSIFIER
# ─────────────────────────────────────────
def classify_query(question: str) -> str:
    """
    Returns: 'toc' | 'structural' | 'page' | 'semantic'
    Uses substring matching — no LLM cost.
    """
    q = question.lower().strip()

    # TOC: contains any TOC phrase anywhere in question
    if any(phrase in q for phrase in TOC_PHRASES):
        return "toc"

    if is_document_overview_query(question):
        return "overview"

    # Structural: asking about a specific chapter/section by name or number (e.g. 2.2. Title)
    if any(phrase in q for phrase in STRUCTURAL_PHRASES):
        return "structural"
    if _NUMBERED_SECTION.search(question):
        return "structural"

    # Page # before bare-section heuristics ("page 1 Stratec" is not section "1 STRATEC")
    pdf_p, doc_p = parse_page_number_from_query(question)
    if pdf_p is not None or (doc_p is not None and is_valid_document_page_label(doc_p)):
        return "page"

    if _TOP_LEVEL_SECTION.search(question) and _SECTION_CONTENT.search(question):
        return "section_content"
    if _SECTION_CONTENT.search(question) and len(question.strip()) < 200:
        return "section_content"
    # Bare section title: "6 IMPLEMENTATION AT STRATEC ?" (not "page 6 …")
    if (
        _TOP_LEVEL_SECTION.search(question)
        and len(question.strip()) < 120
        and not re.search(r"\b(?:pdf\s+)?(?:page|p\.?|pg\.?)\s+\d", q, re.I)
    ):
        return "section_content"

    if _is_figure_caption_query(question):
        return "figure_caption"

    if _is_visual_scene_query(question):
        return "visual_scene"

    if re.search(r"\b(?:page|p\.|pg\.)\s+[a-z0-9ivxlcdm\-]+\b", q, re.I):
        m = re.search(r"\b(?:page|p\.|pg\.)\s+([a-z0-9ivxlcdm\-]+)\b", q, re.I)
        if m and is_valid_document_page_label(m.group(1)):
            return "page"

    return "semantic"


def _is_figure_caption_query(question: str) -> bool:
    """Long caption + figure/image search (not a page number)."""
    q = re.sub(
        r"\s*(?:search|find|show|locate)\s+(?:for\s+)?(?:that\s+)?"
        r"(?:figure|fig\.?|table|image).*",
        "",
        question,
        flags=re.I,
    ).strip()
    if len(q) < 30:
        return False
    return bool(re.search(r"\b(figure|fig\.?|image|photo|picture)\b", question, re.I))


def _is_visual_scene_query(question: str) -> bool:
    """Describe-a-picture / find page by scene (no page number)."""
    q = question.lower()
    pdf_p, doc_p = parse_page_number_from_query(question)
    if pdf_p is not None or (doc_p and is_valid_document_page_label(doc_p)):
        return False
    scene_hints = (
        "lady", "woman", "man", "person", "people", "holding", "phone",
        "screenshot", "photo", "picture", "image where", "image of", "shows",
        "showing", "illustration of", "diagram with",
    )
    if not any(h in q for h in scene_hints):
        return False
    return bool(
        re.search(r"\b(image|picture|photo|screenshot|page)\b", q)
        or "holding" in q
    )


# ─────────────────────────────────────────
# NODES
# ─────────────────────────────────────────
def retrieve_node(state: ESGState):
    question    = state["question"]
    user_context = state.get("user_context")
    focus_id = state.get("focus_section_id")
    document_id = state.get("document_id")

    if focus_id:
        context = retriever.section_detail_retrieve(
            focus_id,
            user_context=user_context,
            parent_section_id=state.get("parent_section_id"),
        )
        return {
            "retrieved_context": context,
            "keywords":          [],
            "sources":           context["chunks"],
            "query_type":        "section_detail",
        }

    query_type  = classify_query(question)

    if query_type == "toc":
        context  = retriever.get_table_of_contents(
            query=question, user_context=user_context, document_id=document_id
        )
        keywords = ["toc"]

    elif query_type == "overview":
        context = retriever.document_overview_retrieve(
            query=question,
            limit=12,
            user_context=user_context,
        )
        keywords = ["overview"]

    elif query_type == "structural":
        context = retriever.structural_retrieve(
            query=question,
            limit=12,
            user_context=user_context,
            document_id=document_id,
        )
        keywords = ["structural"]

    elif query_type == "section_content":
        context = retriever.section_content_retrieve(
            query=question,
            limit=8,
            user_context=user_context,
            document_id=document_id,
        )
        keywords = ["section_content"]

    elif query_type == "page":
        pdf_p, doc_p = parse_page_number_from_query(question)
        context = retriever.page_lookup_retrieve(
            query=question,
            pdf_page=pdf_p,
            document_page=doc_p,
            user_context=user_context,
            document_id=document_id,
        )
        keywords = ["page"]

    elif query_type in ("visual_scene", "figure_caption"):
        context = retriever.unified_visual_retrieve(
            query=question,
            limit=5,
            user_context=user_context,
        )
        keywords = ["visual"]

    else:
        # Broad fetch → rerank → top chunks for LLM
        context = retriever.hybrid_retrieve(
            query=question,
            limit=8,
            user_context=user_context,
        )
        keywords = _extract_keywords(question)

    return {
        "retrieved_context": context,
        "keywords":          keywords,
        "sources":           context["chunks"],
        "query_type":        query_type,
    }


def _format_page_text_answer(retrieved: dict, chunks: list) -> str | None:
    if retrieved.get("mode") != "page_text" or not chunks:
        return None
    c = chunks[0]
    pdf_p = retrieved.get("pdf_page") or c.get("pdf_page")
    text = (c.get("text") or "").strip()
    if not text or text == "(No text extracted for this page.)":
        return (
            f"No extractable text was found on PDF page **{pdf_p}**. "
            "The page may be image-only; try asking for the image instead."
        )
    header = f"**Text from PDF page {pdf_p}**"
    doc_p = c.get("document_page")
    if doc_p and str(doc_p) != str(pdf_p):
        header += f" (printed page **{doc_p}**)"
    return f"{header}\n\n{text}"


def _format_unified_visual_answer(retrieved: dict, chunks: list) -> str | None:
    mode = retrieved.get("mode") or ""
    if mode not in (
        "unified_visual", "page_lookup", "caption_figure",
        "visual_scene", "page_visual_list",
    ):
        return None
    if not chunks:
        return (
            "No matching page or figure found. "
            "Try a **PDF page number** (e.g. `pdf page 15`) or re-ingest with vision enabled."
        )

    if mode == "page_visual_list":
        return _format_page_visual_list_answer(retrieved, chunks)

    c = chunks[0]
    pdf_p = retrieved.get("pdf_page") or c.get("pdf_page") or c.get("doc_order")
    focus = retrieved.get("visual_focus") or []
    if retrieved.get("single_visual") and focus:
        label = focus[0].title()
        lines = [f"**{label}**", ""]
        if pdf_p:
            lines.append(f"_PDF page **{pdf_p}**._")
            lines.append("")
        if c.get("image_url") or c.get("image_key"):
            lines.append("_Logo/icon crop below._")
        blob = display_text_for_chunk(c).lower()
        if any(f.lower() in blob for f in focus):
            lines.append("_Matched using vision text for this crop._")
        else:
            lines.append(
                "_No separate logo region is indexed on this page; "
                "showing the only figure crop available._"
            )
        return "\n".join(lines).strip()

    lines = [f"**{c.get('title', 'Page')}**", ""]
    if pdf_p:
        lines.append(f"_PDF page **{pdf_p}**._")
        lines.append("")
    if c.get("image_url") or c.get("image_key"):
        src = "region crop" if c.get("image_source") == "region" else "page image"
        lines.append(f"_Showing {src} below._")
        lines.append("")
    body = display_text_for_chunk(c)
    if body:
        lines.append(body[:4000])
    return "\n".join(lines).strip()


def _format_page_visual_list_answer(retrieved: dict, chunks: list) -> str | None:
    if retrieved.get("mode") != "page_visual_list" or not chunks:
        return None

    kind = retrieved.get("list_kind") or "visual"
    label = "Figures" if kind == "figure" else "Tables" if kind == "table" else "Visuals"
    pdf_p = retrieved.get("pdf_page") or chunks[0].get("pdf_page")
    doc_p = retrieved.get("document_page") or chunks[0].get("document_page")

    lines = [f"**{label} on PDF page {pdf_p}** ({len(chunks)} found)", ""]
    if doc_p and str(doc_p) != str(pdf_p):
        lines.append(f"_Printed page **{doc_p}**._")
        lines.append("")

    for i, c in enumerate(chunks, 1):
        title = c.get("title") or f"{label[:-1]} {i}"
        lines.append(f"{i}. **{title}**")
        visual = compact_visual_content((c.get("visual_content") or "").strip())
        blob = visual.lower()
        if "logo" in blob or "brand" in blob:
            lines.append("   _Logo / brand mark._")
        elif "diagram" in blob or "flowchart" in blob:
            lines.append("   _Diagram._")
        elif visual:
            first = visual.split("\n", 1)[0].strip()
            if len(first) > 120:
                first = first[:117] + "…"
            lines.append(f"   _{first}_")
        if c.get("image_url") or c.get("image_key"):
            lines.append("   _(see image in the panel)_")
        lines.append("")

    return "\n".join(lines).strip()


def _uses_visual_prompt(retrieved: dict, chunks: list) -> bool:
    mode = retrieved.get("mode") or ""
    if "table" in mode or "visual" in mode:
        return True
    for c in chunks:
        types = c.get("match_types") or []
        if any(t in types for t in ("visual_page", "visual", "visual_region", "table_match", "region_match")):
            return True
        if "[Visual page content" in (c.get("text") or ""):
            return True
    return False


def _passthrough_visual_content(
    chunks: list, question: str, retrieved: Optional[dict] = None
) -> str | None:
    """
    For table/chart/diagram questions, return stored vision text directly when present
    so the answer is not shortened or rewritten by the chat model.
    """
    if retrieved and retrieved.get("mode") in (
        "unified_visual", "caption_figure", "visual_scene",
        "page_visual_list", "page_lookup",
    ):
        return None

    q = question.lower()
    if not any(
        w in q
        for w in (
            "table", "chart", "diagram", "figure", "graph", "map",
            "draw", "recreate", "create", "show", "visual", "shape",
        )
    ):
        return None

    caption = _is_figure_caption_query(question)
    anchors: list[str] = []
    if caption:
        cap_text = re.sub(
            r"\s*(?:search|find|show|locate)\s+(?:for\s+)?(?:that\s+)?"
            r"(?:figure|fig\.?|table|image).*",
            "",
            question,
            flags=re.I,
        ).lower()
        for phrase in ("guatemala", "respiratory", "windhoek", "namibia"):
            if phrase in cap_text:
                anchors.append(phrase)

    for c in chunks:
        text = c.get("text") or ""
        marker = "[Visual page content"
        if marker not in text:
            continue
        blob = ((c.get("visual_content") or "") + text).lower()
        if anchors and not any(a in blob for a in anchors):
            continue
        idx = text.index(marker)
        body = text[idx:]
        title = c.get("title", "Document page")
        return (
            f"**{title}** — from page vision (tables, charts, diagrams, shapes):\n\n"
            f"{body}\n\n"
            "_Verify critical values against the original PDF if needed._"
        )
    return None


def _format_subsection_listing(retrieved_context: dict) -> str | None:
    """
    Build answer from graph retrieval when parent + descendants were found.
    Avoids LLM wrongly claiming 'no subsections' when headings are separate chunks.
    """
    if retrieved_context.get("mode") != "subsection_tree":
        return None

    chunks = retrieved_context.get("chunks") or []
    parent_title = retrieved_context.get("parent_title") or "the section"

    children = [
        c for c in chunks
        if "descendant_of_match" in (c.get("match_types") or [])
    ]
    if not children and len(chunks) > 1:
        children = chunks[1:]

    if not children:
        return (
            f'The section "{parent_title}" has no nested subsections '
            "in the knowledge graph."
        )

    lines = [f'Subsections under "{parent_title}":', ""]
    for i, c in enumerate(children, 1):
        lines.append(f"{i}. {c['title']}")
    return "\n".join(lines)


def _format_toc_listing(
    chunks: list,
    question: str = "",
    retrieved: Optional[dict] = None,
) -> str | None:
    """Full TOC from graph — avoids LLM truncation at ~16 items."""
    if not chunks or chunks[0].get("id") == "access_denied":
        return None
    title = "Table of Contents"
    doc_title = (retrieved or {}).get("document_title")
    hint = (retrieved or {}).get("document_hint")
    if doc_title:
        title = f"Table of Contents — {doc_title}"
    elif hint:
        title = f"Table of Contents — {hint.strip().title()}"
    elif question and re.search(r"\b(?:from|form)\b", question, re.I):
        anchor = re.search(
            r"\b(?:from|form)\s+(?:the\s+)?(.+?)(?:\s+document)?(?:\s+of\s+|\?|$)",
            question,
            re.I,
        )
        if anchor:
            raw = anchor.group(1).strip()
            raw = re.sub(
                r"^(?:all\s+)?(?:the\s+)?(?:list\s+of\s+)?(?:all\s+)?"
                r"(?:table\s+of\s+contents?|toc)\s*",
                "",
                raw,
                flags=re.I,
            ).strip()
            if raw and raw.lower() not in ("document", "pdf"):
                title = f"Table of Contents — {raw.title()}"
    lines = [title, ""]
    for i, c in enumerate(chunks, 1):
        entry = c.get("title") or "Section"
        page = c.get("page")
        if page is None and c.get("text"):
            pm = re.search(r"\(Page\s+(\d+)\)", c.get("text", ""))
            if pm:
                page = pm.group(1)
        lines.append(f"{i}. {entry}" + (f" (Page {page})" if page else ""))
    return "\n".join(lines)


def _section_content_context(retrieved: dict, chunks: list) -> str:
    """Build LLM context: subsections first; parent heading may be empty."""
    parent_title = retrieved.get("parent_title") or ""
    subs = [
        c for c in chunks
        if (c.get("text") or "").strip()
        and c.get("title") != parent_title
        and len((c.get("text") or "").strip()) > len((c.get("title") or "")) + 15
    ]
    if not subs:
        subs = [c for c in chunks if (c.get("text") or "").strip()]
    heading = parent_title or (chunks[0].get("title") if chunks else "Section")
    parts = [
        f'Section: "{heading}"',
        "Note: the main heading may have no body text; use the subsections below.",
        "",
    ]
    for c in subs:
        parts.append(f"### {c.get('title', 'Subsection')}\n{c.get('text', '')}")
    return "\n\n".join(parts)


def _format_section_content_answer(retrieved: dict, chunks: list) -> str | None:
    """Deterministic summary when subsections carry the real content."""
    mode = retrieved.get("mode")
    if mode not in ("section_content", "section_content_fallback"):
        return None

    parent_title = retrieved.get("parent_title") or ""
    subs = [
        c for c in chunks
        if c.get("title") != parent_title and len((c.get("text") or "").strip()) > 60
    ]
    if not subs:
        return None

    heading = parent_title or subs[0].get("title", "Section")
    lines = [f"**{heading}** covers the following:", ""]
    for c in subs:
        text = (c.get("text") or "").strip()
        title = (c.get("title") or "").strip()
        paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        body_paras = [
            p for p in paras
            if p.lower() != title.lower() and not p.lower().startswith(title.lower() + "\n")
            and len(p) > 40
        ]
        lead = body_paras[0] if body_paras else (paras[-1] if paras else text)
        lead = lead.replace("\n", " ").strip()
        if len(lead) > 480:
            lead = lead[:477] + "..."
        lines.append(f"- **{title}**: {lead}")
    return "\n".join(lines)


def generate_node(state: ESGState):
    chunks     = state["retrieved_context"]["chunks"]
    query_type = state.get("query_type", "semantic")
    retrieved  = state["retrieved_context"]

    if retrieved.get("mode") == "needs_clarification":
        msg = retrieved.get("clarification_message") or (
            "I need one more detail before I can answer. Please pick an option from the list."
        )
        return {"answer": msg, "low_confidence": False}

    if not chunks:
        if retrieved.get("mode") == "table_of_contents" and retrieved.get("document_not_found"):
            hint = retrieved.get("document_hint") or "that document"
            lines = [
                f'No document matching **"{hint}"** is in the knowledge graph.',
                "",
                "The table of contents was **not** taken from another document.",
            ]
            avail = retrieved.get("available_documents") or []
            if avail:
                lines.append("")
                lines.append("Documents with ingested sections:")
                for name in avail:
                    lines.append(f"- {name}")
            lines.append("")
            lines.append("Upload the missing PDF at **/upload** and wait until ingestion shows **Done**.")
            return {"answer": "\n".join(lines), "low_confidence": False}
        return {"answer": "I couldn't find relevant information in the document knowledge graph."}

    if state.get("query_type") == "toc" or retrieved.get("mode") == "table_of_contents":
        toc = _format_toc_listing(chunks, state.get("question", ""), retrieved=retrieved)
        if toc:
            return {"answer": toc, "low_confidence": False}

    if retrieved.get("mode") != "section_detail":
        listing = _format_subsection_listing(retrieved)
        if listing is not None:
            return {"answer": listing, "low_confidence": False}

    section_answer = _format_section_content_answer(retrieved, chunks)
    if section_answer is not None:
        return {"answer": section_answer, "low_confidence": False}

    visual_list = _format_page_visual_list_answer(retrieved, chunks)
    if visual_list is not None:
        return {"answer": visual_list, "low_confidence": False}

    page_text_answer = _format_page_text_answer(retrieved, chunks)
    if page_text_answer is not None:
        return {"answer": page_text_answer, "low_confidence": False}

    visual_answer = _format_unified_visual_answer(retrieved, chunks)
    if visual_answer is not None:
        return {"answer": visual_answer, "low_confidence": False}

    visual_answer = _passthrough_visual_content(
        chunks, state["question"], retrieved=retrieved
    )
    if visual_answer is not None:
        return {"answer": visual_answer, "low_confidence": False}

    visual_prompt_mode = _uses_visual_prompt(retrieved, chunks)

    # ── Confidence check (semantic queries only) ──────────────
    low_confidence = False
    if query_type == "semantic":
        top_score = max((c.get("score", 0) for c in chunks), default=0)
        if top_score < MIN_CONFIDENCE_SCORE:
            low_confidence = True

    if query_type == "structural" and retrieved.get("parent_title"):
        context_parts = [f"Parent section: {retrieved['parent_title']}", "Subsections:"]
        for i, c in enumerate(
            [c for c in chunks if c.get("title") != retrieved.get("parent_title")],
            1,
        ):
            context_parts.append(f"{i}. {c['title']}\n{c['text'][:1500]}")
        context_text = "\n\n".join(context_parts)
    elif query_type == "section_content" or retrieved.get("mode") == "section_content":
        context_text = _section_content_context(retrieved, chunks)
    else:
        context_parts = []
        for c in chunks:
            header = f"[Section: {c['title']}]"
            context_parts.append(f"{header}\n{c['text']}")
        context_text = "\n\n".join(context_parts)

    # ── System prompt adapts to query type + confidence ───────
    if query_type == "toc":
        system_prompt = load_prompt("document_toc", context=context_text, question=state["question"])
    elif query_type == "page":
        system_prompt = load_prompt("document_page", context=context_text, question=state["question"])
    elif visual_prompt_mode:
        system_prompt = load_prompt("document_visual", context=context_text, question=state["question"])
    elif query_type == "structural" or retrieved.get("mode") == "subsection_tree":
        system_prompt = load_prompt("document_structural", context=context_text, question=state["question"])
    elif query_type == "section_content" or retrieved.get("mode") == "section_content":
        system_prompt = load_prompt("document_section_content", context=context_text, question=state["question"])
    elif query_type == "section_detail":
        system_prompt = load_prompt("document_default", context=context_text, question=state["question"])
    elif query_type == "overview":
        n_pts = _requested_point_count(state["question"])
        q_user = state["question"]
        if not re.search(r"\b\d{1,2}\s+points?\b", q_user, re.I):
            q_user = f"{q_user.strip()} (Respond with exactly {n_pts} bullet points.)"
        system_prompt = load_prompt(
            "document_overview", context=context_text, question=q_user
        )
    elif low_confidence:
        system_prompt = load_prompt("document_low_confidence", context=context_text, question=state["question"])
    else:
        system_prompt = load_prompt("document_default", context=context_text, question=state["question"])

    response = provider.chat_completion(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"{state['question']}"},
        ],
        temperature=0.1,
    )

    return {
        "answer":         response.choices[0].message.content,
        "low_confidence": low_confidence,
    }


def should_continue(state: ESGState):
    # Always run generate_node so empty retrieval still returns a user-visible message.
    return "generate"


# ─────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────
def _extract_keywords(question: str) -> List[str]:
    stopwords = {
        "what", "are", "the", "that", "this", "with", "for", "and", "or",
        "in", "of", "to", "a", "an", "is", "it", "section", "document",
        "from", "do", "does", "did", "can", "could", "would", "should",
        "will", "have", "has", "had", "be", "been", "being", "members",
        "ask", "themselves", "their", "they", "we", "our", "us", "me", "my", "i",
    }
    cleaned = re.sub(r'[^a-z\s]', '', question.lower())
    return [w for w in cleaned.split() if w not in stopwords and len(w) > 2][:5]


# ─────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────
workflow = StateGraph(ESGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_conditional_edges(
    "retrieve",
    should_continue,
    {"generate": "generate"},
)
workflow.add_edge("generate", END)

# NO MEMORY — each query is independent
esg_agent = workflow.compile()