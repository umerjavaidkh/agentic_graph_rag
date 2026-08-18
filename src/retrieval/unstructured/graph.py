"""
retrieval/unstructured/graph.py — Neo4j Graph RAG agent.

Vector seed + graph expansion + LLM synthesis.
"""

from langgraph.graph import END, StateGraph

import re

from ...routing import document_agent_structured_guard, has_document_cue, structured_entity_summary
from .retriever import (
    DocumentRAGRetriever,
    is_page_question,
    is_synthesis_question,
    is_toc_question,
    is_visual_page_question,
)
from ...config.prompts import load_prompt
from ...config.settings import (
    CHAT_MODEL,
    DOCUMENT_SYNTHESIS_CONTEXT_MAX_CHARS,
    DOCUMENT_SYNTHESIS_LONG_MAX_TOKENS,
    DOCUMENT_SYNTHESIS_MAX_TOKENS,
    RETRIEVAL_FINAL_LIMIT,
)
from ...model_providers.factory import get_chat_provider
from ...telemetry import pipeline_step
from .state import ESGState
from .verification import compute_confidence

retriever = DocumentRAGRetriever()
provider = get_chat_provider()

_STRUCTURED_MISROUTE = re.compile(
    r"not in the document corpus|use structured data access",
    re.I,
)

# Retrieval modes where chunks are already the answer (TOC, page, box, subsection).
_STRUCTURAL_FAST_MODES = frozenset({
    "structural_toc",
    "structural_page",
    "structural_page_visual",
    "page_visual_list",
    "structural_box_list",
    "structural_box_content",
    "subsection_tree",
    "section_detail",
    "structural_filing_date",
    "needs_clarification",
})


def _build_fast_unstructured_answer(chunks: list[dict]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        title = (chunk.get("title") or "").strip()
        if title and title.lower() not in text.lower()[:80]:
            parts.append(f"**{title}**\n{text}")
        else:
            parts.append(text)
    return "\n\n".join(parts).strip()


def _fix_misrouted_structured_answer(answer: str, question: str) -> str:
    """LLM sometimes mis-applies the structured-data redirect on document
    questions."""
    if not _STRUCTURED_MISROUTE.search(answer or ""):
        return (answer or "").strip()
    if not has_document_cue(question):
        return (answer or "").strip()
    return (
        # Names no dataset: this text reaches the user, and the structured
        # graph is whatever schema happens to be loaded (it was Northwind,
        # it is now an e-commerce dataset, it will be a customer's own).
        # Naming one made the reply wrong for every other deployment.
        "This is a document question (ingested PDF content), not the structured business database. "
        "I searched the ingested document sections but could not find the exact figure or detail you asked for. "
        "Try rephrasing with a section number or page reference if you have one."
    )


def retrieve_node(state: ESGState):
    question = state["question"]
    user_context = state.get("user_context")
    document_id_hint = state.get("document_id") or ""

    limit = max(RETRIEVAL_FINAL_LIMIT, 12) if is_synthesis_question(question) else RETRIEVAL_FINAL_LIMIT
    with pipeline_step("document.graph.retrieve", limit=limit):
        context = retriever.hybrid_retrieve(
            query=question,
            limit=limit,
            user_context=user_context,
            document_id_hint=document_id_hint,
        )
    strategy = context.get("strategy", "graph_rag")
    return {
        "retrieved_context": context,
        "keywords": [],
        "sources": context.get("chunks", []),
        "query_type": strategy,
    }


_CITATION_STOPWORDS = frozenset(
    "the a an and or of to in for on with is are was were be been this that these "
    "those it its as at by from not no which who whom whose what when where how "
    "section covers cover document report page pages".split()
)


def _grounded_sources(answer: str, chunks: list[dict]) -> list[dict]:
    """Keep the chunks the answer actually draws on.

    Every retrieved chunk was being cited, not just the ones used, so a reader
    was pointed at roughly four times the material that bore on the answer --
    measured at 0.26 mean page precision against 88% correct section naming.
    A citation nobody can practically check is close to no citation at all,
    which is the whole reason citation is tracked separately from answer text.

    Lexical overlap rather than a second LLM call: it costs nothing per query
    and is deterministic, so the same answer always cites the same sources.

    Fails OPEN. If nothing clears the bar the original list is returned
    unchanged -- dropping a source a reader wanted is worse than showing one
    they did not need, and the source viewer must never be left with nothing
    to open.
    """
    if not answer or len(chunks) <= 1:
        return chunks

    def words(text: str) -> set[str]:
        return {
            w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
            if len(w) > 3 and w not in _CITATION_STOPWORDS
        }

    answer_words = words(answer)
    if not answer_words:
        return chunks

    scored = []
    for c in chunks:
        overlap = len(answer_words & words(c.get("text") or c.get("title") or ""))
        scored.append((overlap, c))

    best = max(o for o, _ in scored)
    if best == 0:
        return chunks
    # Half the best chunk's overlap: keeps genuine supporting passages while
    # dropping the long tail that merely came back from the same retrieval.
    keep = [c for o, c in scored if o >= max(1, best // 2)]
    return keep or chunks


def generate_node(state: ESGState):
    question = state["question"]
    retrieved = state.get("retrieved_context", {}) or {}
    chunks = retrieved.get("chunks", []) or []

    with pipeline_step(
        "document.graph.generate",
        mode=retrieved.get("mode"),
        chunks=len(chunks),
    ):
        return _generate_document_answer(
            question,
            retrieved,
            chunks,
            user_context=state.get("user_context"),
            skip_structured_guard=bool(state.get("skip_structured_guard")),
        )


def _generate_document_answer(
    question: str,
    retrieved: dict,
    chunks: list,
    *,
    user_context=None,
    skip_structured_guard: bool = False,
) -> dict:
    denied = next((c for c in chunks if c.get("id") == "access_denied"), None)
    if not chunks or denied:
        # Misroute guard: structured-graph question sent to document agent →
        # autofix or generic hint. Gated on retrieval having found NOTHING
        # real here (either no chunks at all, or the only "chunk" is the
        # synthetic access_denied marker) -- not on the question's wording
        # alone. Financial-document vocabulary ("sales", "revenue", "profit")
        # is also a Northwind-era structured-data cue, so a bare keyword
        # match used to redirect real 10-K questions away from chunks the
        # document agent had already found (see
        # [[repo_keyword_routing_scaling_risk]]). Only reaching for this
        # guard once retrieval itself came up empty (or denied) keeps it
        # doing its original job -- catching genuine misroutes -- without
        # discarding real content that happens to share vocabulary with the
        # other graph.
        #
        # The `denied` case matters just as much as the empty-chunks case:
        # a user who lacks document/"esg" access but DOES have structured
        # access, asking a structured-shaped question, used to get stuck on
        # the flat "access denied" message below without ever trying the
        # redirect -- because access_denied_response() returns a non-empty
        # chunks list (one marker chunk), so `if not chunks:` alone never
        # fired for this case. Skipped entirely when called as the
        # structured path's own low-confidence fallback
        # (routing.try_document_fallback) -- bouncing back to structured
        # there would just recreate the low-confidence answer we're trying
        # to improve on, an infinite ping-pong for questions phrased like
        # analytics but whose entity only exists in the ingested documents.
        if not skip_structured_guard:
            guard = document_agent_structured_guard(question, user_context)
            if guard is not None:
                return guard

        if denied:
            return {
                "answer": (denied.get("text") or "Access denied for document data.").strip(),
                "low_confidence": False,
            }

        answer = "I could not find relevant information in the ingested documents."
        low_confidence, confidence_note = compute_confidence(
            question, answer, chunks, "", provider=provider, model=CHAT_MODEL
        )
        return {
            "answer": answer,
            "low_confidence": low_confidence,
            "confidence_note": confidence_note,
        }

    mode = (retrieved.get("mode") or "").strip()
    if mode in _STRUCTURAL_FAST_MODES:
        answer = _build_fast_unstructured_answer(chunks)
        if answer:
            return {"answer": answer, "low_confidence": False}

    # Budget the total prompt context in characters, not just per-chunk --
    # a single chunk (e.g. a whole Chapter node's .text) can itself be huge
    # once chapter detection is accurate, so capping only the OUTPUT side
    # (DOCUMENT_SYNTHESIS_MAX_TOKENS) never bounded the INPUT side at all.
    # Truncates/drops the tail of context rather than the LLM call failing
    # outright on a context-window or rate-limit error.
    context_lines: list[str] = []
    used_chars = 0
    for i, c in enumerate(chunks, 1):
        title = c.get("title", "Result")
        text = (c.get("text") or "").strip()
        if not text:
            continue
        remaining = DOCUMENT_SYNTHESIS_CONTEXT_MAX_CHARS - used_chars
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].rstrip() + "\n[... truncated, content continues beyond this excerpt ...]"
        rel = c.get("related") or []
        rel_note = f" (graph: {', '.join(rel)})" if rel else ""
        line = f"[Chunk {i}] {title}{rel_note}\n{text}"
        context_lines.append(line)
        used_chars += len(line)
    context_text = "\n\n".join(context_lines)

    if is_toc_question(question):
        prompt_name = "document_toc"
    elif is_visual_page_question(question):
        prompt_name = "document_visual"
    elif is_page_question(question):
        prompt_name = "document_page"
    elif is_synthesis_question(question):
        prompt_name = "document_synthesis"
    else:
        prompt_name = "document_default"
    system_prompt = load_prompt(
        prompt_name,
        context=context_text,
        question=question,
        structured_entities=structured_entity_summary(),
    )
    response = provider.chat_completion(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(question)},
        ],
        temperature=0.1,
        max_tokens=(
            DOCUMENT_SYNTHESIS_LONG_MAX_TOKENS
            if (
                is_toc_question(question)
                or is_page_question(question)
                or is_visual_page_question(question)
            )
            else DOCUMENT_SYNTHESIS_MAX_TOKENS
        ),
    )
    answer = _fix_misrouted_structured_answer(
        response.choices[0].message.content.strip(), question
    )
    low_confidence, confidence_note = compute_confidence(
        question, answer, chunks, mode, provider=provider, model=CHAT_MODEL
    )
    return {
        "answer": answer,
        "low_confidence": low_confidence,
        "confidence_note": confidence_note,
        "sources": _grounded_sources(answer, chunks),
    }


def should_continue(state: ESGState):
    return "generate"


workflow = StateGraph(ESGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_conditional_edges("retrieve", should_continue, {"generate": "generate"})
workflow.add_edge("generate", END)

esg_agent = workflow.compile()

