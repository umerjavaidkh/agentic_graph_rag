"""Phased NDJSON query streaming orchestrator."""
from __future__ import annotations

import json
from typing import Iterator, Optional

from ..auth.roles import DEFAULT_PUBLIC_CONTEXT, UserContext
from ..conversation import get_turn, resolve_follow_up, save_turn
from ..presentation import build_presentation
from ..router import _rbac_check
from ..routing import is_structured_data_question, make_structured_access_denied_result, select_mcp_tool
from ..telemetry import clear_telemetry, get_telemetry, pipeline_step, start_telemetry
from .document import (
    _build_context_text,
    _document_max_tokens,
    _document_prompt_name,
    iter_document_stream,
)
from ..config.prompts import load_prompt
from ..config.settings import (
    CHAT_MODEL,
    STRUCTURED_MODEL,
    STRUCTURED_SYNTHESIS_LONG_MAX_TOKENS,
    STRUCTURED_SYNTHESIS_MAX_TOKENS,
)
from ..model_providers.factory import get_model_provider
from ..retrieval.structured.graph import (
    _build_fast_structured_answer,
    _should_fast_structured_answer,
)
from ..retrieval.structured.query_intent import estimate_structured_synthesis_max_tokens
from ..retrieval.unstructured.graph import (
    _STRUCTURAL_FAST_MODES,
    _build_fast_unstructured_answer,
    _fix_misrouted_structured_answer,
)
from ..retrieval.unstructured.graph import retrieve_node as doc_retrieve_node
from ..retrieval.structured.graph import retrieve_node as struct_retrieve_node
from .events import stream_event
from .structured import _viz_blocks_only, iter_structured_stream


def _resolve_tool(question: str, thread_id: str) -> tuple[str, dict]:
    prior = get_turn(thread_id)
    tool_name = None
    resolved = {"question": question, "use_prior": False}
    with pipeline_step("route.select"):
        if prior:
            resolved = resolve_follow_up(question, prior)
            if resolved.get("use_prior"):
                fk = resolved.get("follow_up_kind") or ""
                if fk in ("subsection_detail", "page", "page_visual_focus", "clarification_document"):
                    tool_name = "search_documents"
                elif fk == "structured_clarification":
                    tool_name = "query_data"
        if not tool_name:
            tool_name = select_mcp_tool(question)
    return tool_name, resolved


def _enrich_and_persist(
    *,
    tool_name: str,
    question: str,
    thread_id: str,
    ctx: UserContext,
    resolved: dict,
    final: dict,
    request_id: Optional[str],
) -> dict:
    tel = get_telemetry()
    route_method = "llm_mcp"
    if tel is not None:
        tel.set_route(
            tool_name,
            route_method,
            agent=final.get("agent"),
            strategy=final.get("strategy"),
        )
        telemetry = tel.summary()
    else:
        telemetry = {}

    out = {
        "answer": final.get("answer", ""),
        "sources": final.get("sources", []),
        "keywords": [],
        "agent": final.get("agent", tool_name),
        "strategy": final.get("strategy", ""),
        "query_type": final.get("query_type"),
        "presentation": final.get("presentation"),
        "_route_tool": tool_name,
        "_route_method": route_method,
        "_access_level": ctx.role.value,
        "_follow_up": resolved.get("follow_up_kind") if resolved.get("use_prior") else None,
        "_telemetry": telemetry,
    }
    save_turn(thread_id, question, out)
    clear_telemetry()
    return {
        "type": "done",
        **final,
        "route_tool": tool_name,
        "route_method": final.get("route_method") or route_method,
        "telemetry": telemetry,
        "request_id": request_id,
        "access_level": ctx.role.value,
        "follow_up": out["_follow_up"],
    }


def iter_hybrid_stream(
    question: str,
    *,
    user_context: Optional[UserContext],
) -> Iterator[str]:
    state = {"question": question}
    if user_context is not None:
        state["user_context"] = user_context

    yield stream_event(type="status", phase="retrieval", agent="hybrid")

    doc_partial = doc_retrieve_node(state)
    struct_partial = struct_retrieve_node(state)
    doc_retrieved = doc_partial.get("retrieved_context") or {}
    doc_chunks = doc_retrieved.get("chunks") or []
    struct_retrieved = struct_partial.get("retrieved_context") or {}
    struct_chunks = struct_retrieved.get("chunks") or []
    data_strategy = struct_partial.get("strategy") or struct_retrieved.get("strategy") or "text2cypher"

    viz = _viz_blocks_only(question, struct_chunks)
    if viz:
        yield stream_event(type="presentation", partial=True, agent="hybrid", blocks=viz["blocks"])

    provider = get_model_provider()
    doc_answer = ""
    doc_mode = (doc_retrieved.get("mode") or "").strip()
    if doc_mode in _STRUCTURAL_FAST_MODES:
        doc_answer = _build_fast_unstructured_answer(doc_chunks)
    elif doc_chunks:
        context_text = _build_context_text(doc_chunks)
        prompt_name = _document_prompt_name(question)
        system_prompt = load_prompt(prompt_name, context=context_text, question=question)
        yield stream_event(type="status", phase="synthesis", agent="hybrid", section="documents")
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
            yield stream_event(
                type="token",
                agent="hybrid",
                target="document_markdown",
                delta=delta,
            )
        doc_answer = _fix_misrouted_structured_answer("".join(parts).strip(), question)

    data_answer = ""
    if _should_fast_structured_answer(struct_chunks, data_strategy):
        data_answer = _build_fast_structured_answer(struct_chunks, data_strategy, question)
    elif struct_chunks and not any(c.get("id") == "error" for c in struct_chunks):
        context_lines = []
        for i, c in enumerate(struct_chunks, 1):
            title = c.get("title", "Result")
            text = c.get("text", "")
            context_lines.append(f"#{i}\nTitle: {title}\n{text}")
        system_prompt = load_prompt(
            "structured_synthesis",
            context="\n\n".join(context_lines),
            question=question,
        )
        yield stream_event(type="status", phase="synthesis", agent="hybrid", section="structured")
        parts = []
        for delta in provider.chat_completion_stream(
            model=STRUCTURED_MODEL,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": str(question)},
            ],
            max_tokens=estimate_structured_synthesis_max_tokens(
                question,
                chunk_count=len(struct_chunks),
                default_max=STRUCTURED_SYNTHESIS_MAX_TOKENS,
                long_max=STRUCTURED_SYNTHESIS_LONG_MAX_TOKENS,
            ),
        ):
            parts.append(delta)
            yield stream_event(
                type="token",
                agent="hybrid",
                target="structured_markdown",
                delta=delta,
            )
        data_answer = "".join(parts).strip()

    data_pres = build_presentation(
        question=question,
        answer=data_answer,
        sources=struct_chunks,
        agent="structured",
    )
    if data_pres and data_pres.get("blocks"):
        presentation = {
            "kind": "mixed",
            "blocks": [
                {"type": "markdown", "content": f"### From Documents\n\n{doc_answer}"},
                *data_pres["blocks"],
            ],
        }
    else:
        presentation = build_presentation(
            question=question,
            answer=doc_answer,
            sources=doc_chunks,
            retrieved_context=doc_retrieved,
            query_type=doc_partial.get("query_type"),
        )

    yield stream_event(
        type="done",
        agent="hybrid",
        answer=f"### From Documents:\n{doc_answer}\n\n### From Data:\n{data_answer}",
        sources=doc_chunks,
        document_sources=doc_chunks,
        data_sources=struct_chunks,
        strategy=data_strategy,
        presentation=presentation,
    )


def iter_query_stream(
    question: str,
    *,
    user_context: Optional[UserContext] = None,
    thread_id: str = "default",
    request_id: Optional[str] = None,
) -> Iterator[str]:
    """Yield NDJSON lines: status → presentation (optional) → token* → done."""
    start_telemetry()
    tel = get_telemetry()
    if tel is not None and request_id:
        tel.route["request_id"] = request_id

    yield stream_event(type="status", phase="routing", request_id=request_id)

    try:
        tool_name, resolved = _resolve_tool(question, thread_id)
        ctx = user_context or DEFAULT_PUBLIC_CONTEXT
        yield stream_event(type="status", phase="routed", route_tool=tool_name)

        if (
            tool_name == "query_data"
            and is_structured_data_question(question)
            and not _rbac_check().can_query_knowledge_area(ctx.user_id, "structured")
        ):
            out = make_structured_access_denied_result(question, ctx)
            if tel is not None:
                tel.set_route("query_data", "structured_access_denied")
            final = _enrich_and_persist(
                tool_name="query_data",
                question=question,
                thread_id=thread_id,
                ctx=ctx,
                resolved=resolved,
                final={
                    "type": "done",
                    "agent": "structured",
                    "answer": out.get("answer", ""),
                    "sources": [],
                    "strategy": "access_denied",
                    "route_method": "structured_access_denied",
                },
                request_id=request_id,
            )
            yield stream_event(**final)
            return

        if tool_name == "search_documents":
            stream = iter_document_stream(
                question,
                user_context=user_context,
                resolved_question=resolved.get("question") or question,
                focus_section_id=resolved.get("focus_section_id"),
                parent_section_id=resolved.get("parent_section_id"),
                document_id=resolved.get("document_id"),
                prior_context=get_turn(thread_id) if resolved.get("use_prior") else None,
            )
        elif tool_name == "query_data":
            stream = iter_structured_stream(
                question,
                user_context=user_context,
                resolved_question=resolved.get("question") or question,
            )
        elif tool_name == "query_hybrid":
            stream = iter_hybrid_stream(question, user_context=user_context)
        else:
            stream = iter_document_stream(
                question,
                user_context=user_context,
                resolved_question=question,
            )

        final_payload: Optional[dict] = None
        for line in stream:
            payload = json.loads(line)
            if payload.get("type") == "done":
                final_payload = payload
                continue
            yield line

        if final_payload is None:
            yield stream_event(type="error", message="Stream ended without a done event.")
            return

        enriched = _enrich_and_persist(
            tool_name=tool_name,
            question=question,
            thread_id=thread_id,
            ctx=ctx,
            resolved=resolved,
            final=final_payload,
            request_id=request_id,
        )
        yield stream_event(**enriched)
    except Exception as exc:
        tel = get_telemetry()
        if tel is not None:
            telemetry = tel.summary()
        else:
            telemetry = {}
        clear_telemetry()
        yield stream_event(type="error", message=str(exc), telemetry=telemetry, request_id=request_id)
        raise
