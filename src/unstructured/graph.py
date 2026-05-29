from langgraph.graph import StateGraph, END
from .state import ESGState
from ..document.page_numbers import parse_page_number_from_query
from .retriever import ESGComplianceRetriever
import re
from typing import List
from ..config.prompts import load_prompt
from ..config.settings import CHAT_MODEL
from ..model_providers.factory import get_model_provider

retriever = ESGComplianceRetriever()
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
}

STRUCTURAL_PHRASES = {
    "what is in chapter", "what is in section", "contents of chapter",
    "contents of section", "show chapter", "show section",
    "summarize chapter", "summarize section",
    "subsection", "sub-section", "sub section", "sections under",
    "headings under", "under the section", "list the sections",
    "with headings", "give them with headings", "children of",
    "nested under", "what are the sub",
}


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

    # Structural: asking about a specific chapter/section by name
    if any(phrase in q for phrase in STRUCTURAL_PHRASES):
        return "structural"

    pdf_p, doc_p = parse_page_number_from_query(question)
    if pdf_p is not None or doc_p is not None:
        return "page"

    if re.search(r"\b(?:page|p\.|pg\.)\s+[a-z0-9ivxlcdm\-]+\b", q, re.I):
        return "page"

    return "semantic"


# ─────────────────────────────────────────
# NODES
# ─────────────────────────────────────────
def retrieve_node(state: ESGState):
    question    = state["question"]
    query_type  = classify_query(question)
    user_context = state.get("user_context")

    if query_type == "toc":
        context  = retriever.get_all_sections(user_context=user_context)
        keywords = ["toc"]

    elif query_type == "structural":
        context = retriever.structural_retrieve(
            query=question,
            limit=12,
            user_context=user_context,
        )
        keywords = ["structural"]

    elif query_type == "page":
        pdf_p, doc_p = parse_page_number_from_query(question)
        context = retriever.page_lookup_retrieve(
            query=question,
            pdf_page=pdf_p,
            document_page=doc_p,
            user_context=user_context,
        )
        keywords = ["page"]

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


def _format_page_lookup_answer(retrieved: dict, chunks: list) -> str | None:
    if retrieved.get("mode") != "page_lookup" or not chunks:
        if retrieved.get("mode") == "page_lookup" and not chunks:
            return "No page found for that page number in the knowledge graph."
        return None

    c = chunks[0]
    meta = retrieved
    pdf_p = meta.get("pdf_page") or c.get("pdf_page")
    doc_p = meta.get("document_page") or c.get("document_page")
    lines = [f"**{c.get('title', 'Page')}**", ""]
    if doc_p and str(doc_p) != str(pdf_p):
        lines.append(
            f"_Printed page **{doc_p}** (PDF sheet **{pdf_p}**). "
            "Answers use the printed page label when they differ._"
        )
        lines.append("")
    elif doc_p:
        lines.append(f"_PDF page and printed page label: **{doc_p}**._")
        lines.append("")

    body = (c.get("text") or "").strip()
    if body:
        lines.append("### Text")
        lines.append(body)
        lines.append("")

    visual = c.get("visual_content") or ""
    if not visual and "[Visual page content" in body:
        visual = body
    elif visual:
        lines.append("### Visual description (tables, charts, diagrams)")
        lines.append(visual)

    if len(lines) <= 3:
        lines.append("_No text or visual description stored for this page. Re-ingest with vision enabled._")

    return "\n".join(lines)


def _uses_visual_prompt(retrieved: dict, chunks: list) -> bool:
    mode = retrieved.get("mode") or ""
    if "table" in mode or "visual" in mode:
        return True
    for c in chunks:
        types = c.get("match_types") or []
        if any(t in types for t in ("visual_page", "visual", "table_match")):
            return True
        if "[Visual page content" in (c.get("text") or ""):
            return True
    return False


def _passthrough_visual_content(chunks: list, question: str) -> str | None:
    """
    For table/chart/diagram questions, return stored vision text directly when present
    so the answer is not shortened or rewritten by the chat model.
    """
    q = question.lower()
    if not any(
        w in q
        for w in (
            "table", "chart", "diagram", "figure", "graph", "map",
            "draw", "recreate", "create", "show", "visual", "shape",
        )
    ):
        return None

    for c in chunks:
        text = c.get("text") or ""
        marker = "[Visual page content"
        if marker not in text:
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


def generate_node(state: ESGState):
    chunks     = state["retrieved_context"]["chunks"]
    query_type = state.get("query_type", "semantic")
    retrieved  = state["retrieved_context"]

    if not chunks:
        return {"answer": "I couldn't find relevant information in the document knowledge graph."}

    listing = _format_subsection_listing(retrieved)
    if listing is not None:
        return {"answer": listing, "low_confidence": False}

    page_answer = _format_page_lookup_answer(retrieved, chunks)
    if page_answer is not None:
        return {"answer": page_answer, "low_confidence": False}

    visual_answer = _passthrough_visual_content(chunks, state["question"])
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
    elif query_type == "structural":
        system_prompt = load_prompt("document_structural", context=context_text, question=state["question"])
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
    chunks = state["retrieved_context"]["chunks"]
    if len(chunks) < MIN_CHUNKS_REQUIRED:
        return "no_context"
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
    {"no_context": END, "generate": "generate"},
)
workflow.add_edge("generate", END)

# NO MEMORY — each query is independent
esg_agent = workflow.compile()