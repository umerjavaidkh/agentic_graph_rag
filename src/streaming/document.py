"""Document agent streaming: retrieve, optional early UI, token synthesis."""
from __future__ import annotations

from typing import Any, Iterator, Optional

from ..auth.roles import UserContext
from ..config.prompts import load_prompt
from ..config.settings import (
    CHAT_MODEL,
    DOCUMENT_SYNTHESIS_LONG_MAX_TOKENS,
    DOCUMENT_SYNTHESIS_MAX_TOKENS,
)
from ..model_providers.factory import get_model_provider
from ..presentation import build_presentation
from ..retrieval.unstructured.graph import (
    _STRUCTURAL_FAST_MODES,
    _build_fast_unstructured_answer,
    _fix_misrouted_structured_answer,
    retrieve_node,
)
from ..retrieval.unstructured.retriever import (
    is_page_question,
    is_synthesis_question,
    is_toc_question,
    is_visual_page_question,
)
from ..routing import has_document_cue, is_structured_data_question
from .events import stream_event


def _document_prompt_name(question: str) -> str:
    if is_toc_question(question):
        return "document_toc"
    if is_visual_page_question(question):
        return "document_visual"
    if is_page_question(question):
        return "document_page"
    if is_synthesis_question(question):
        return "document_synthesis"
    return "document_default"


def _document_max_tokens(question: str) -> int:
    if (
        is_toc_question(question)
        or is_page_question(question)
        or is_visual_page_question(question)
    ):
        return DOCUMENT_SYNTHESIS_LONG_MAX_TOKENS
    return DOCUMENT_SYNTHESIS_MAX_TOKENS


def _build_context_text(chunks: list[dict]) -> str:
    context_lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        title = c.get("title", "Result")
        text = (c.get("text") or "").strip()
        if not text:
            continue
        rel = c.get("related") or []
        rel_note = f" (graph: {', '.join(rel)})" if rel else ""
        context_lines.append(f"[Chunk {i}] {title}{rel_note}\n{text}")
    return "\n\n".join(context_lines)


def iter_document_stream(
    question: str,
    *,
    user_context: Optional[UserContext],
    resolved_question: str,
    focus_section_id: Optional[str] = None,
    parent_section_id: Optional[str] = None,
    document_id: Optional[str] = None,
    prior_context: Optional[dict] = None,
) -> Iterator[str]:
    state: dict[str, Any] = {"question": resolved_question}
    if user_context is not None:
        state["user_context"] = user_context
    if focus_section_id:
        state["focus_section_id"] = focus_section_id
        state["parent_section_id"] = parent_section_id
    if document_id:
        state["document_id"] = document_id
    if prior_context:
        state["prior_context"] = prior_context

    yield stream_event(type="status", phase="retrieval", agent="unstructured")
    partial = retrieve_node(state)
    retrieved = partial.get("retrieved_context") or {}
    chunks = retrieved.get("chunks") or []
    query_type = partial.get("query_type") or retrieved.get("strategy") or "graph_rag"

    yield stream_event(
        type="status",
        phase="retrieved",
        agent="unstructured",
        strategy=query_type,
        mode=retrieved.get("mode"),
        chunks=len(chunks),
    )

    if is_structured_data_question(question) and not has_document_cue(question):
        answer = (
            "This question is about the business database (products, orders, customers), "
            "not ingested PDF documents. Re-run with structured access (e.g. regular_001 or "
            "compliance_001) so the system can query product and category data."
        )
        yield stream_event(type="done", agent="unstructured", answer=answer, sources=[], strategy=query_type)
        return

    if not chunks:
        answer = "I could not find relevant information in the ingested documents."
        yield stream_event(type="done", agent="unstructured", answer=answer, sources=[], strategy=query_type)
        return

    denied = next((c for c in chunks if c.get("id") == "access_denied"), None)
    if denied:
        answer = (denied.get("text") or "Access denied for document data.").strip()
        yield stream_event(type="done", agent="unstructured", answer=answer, sources=[], strategy=query_type)
        return

    mode = (retrieved.get("mode") or "").strip()
    if mode in _STRUCTURAL_FAST_MODES:
        answer = _build_fast_unstructured_answer(chunks)
        presentation = build_presentation(
            question=question,
            answer=answer,
            sources=chunks,
            retrieved_context=retrieved,
            query_type=query_type,
        )
        if presentation and presentation.get("blocks"):
            yield stream_event(type="presentation", partial=True, blocks=presentation["blocks"])
        yield stream_event(
            type="done",
            agent="unstructured",
            answer=answer,
            sources=chunks,
            strategy=query_type,
            query_type=query_type,
            presentation=presentation,
        )
        return

    context_text = _build_context_text(chunks)
    prompt_name = _document_prompt_name(question)
    system_prompt = load_prompt(prompt_name, context=context_text, question=question)
    provider = get_model_provider()

    yield stream_event(type="status", phase="synthesis", agent="unstructured")
    parts: list[str] = []
    for delta in provider.chat_completion_stream(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(question)},
        ],
        temperature=0.1,
        max_tokens=_document_max_tokens(question),
    ):
        parts.append(delta)
        yield stream_event(type="token", agent="unstructured", target="markdown", delta=delta)

    answer = _fix_misrouted_structured_answer("".join(parts).strip(), question)
    presentation = build_presentation(
        question=question,
        answer=answer,
        sources=chunks,
        retrieved_context=retrieved,
        query_type=query_type,
    )
    yield stream_event(
        type="done",
        agent="unstructured",
        answer=answer,
        sources=chunks,
        strategy=query_type,
        query_type=query_type,
        presentation=presentation,
    )
