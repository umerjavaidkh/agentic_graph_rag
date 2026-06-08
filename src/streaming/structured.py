"""Structured agent streaming: chart/table first, then narrative tokens."""
from __future__ import annotations

from typing import Any, Iterator, Optional

from ..auth.roles import UserContext
from ..config.prompts import load_prompt
from ..config.settings import (
    STRUCTURED_MODEL,
    STRUCTURED_SYNTHESIS_LONG_MAX_TOKENS,
    STRUCTURED_SYNTHESIS_MAX_TOKENS,
)
from ..model_providers.factory import get_model_provider
from ..presentation import build_presentation
from ..presentation.structured_planner import build_structured_presentation
from ..retrieval.structured.graph import (
    _build_fast_structured_answer,
    _should_fast_structured_answer,
    retrieve_node,
)
from ..retrieval.structured.query_intent import estimate_structured_synthesis_max_tokens
from ..telemetry.pipeline import record_pipeline_step
from .events import stream_event


def _viz_blocks_only(question: str, sources: list[dict]) -> Optional[dict]:
    pres = build_structured_presentation(question, "", sources)
    if not pres:
        return None
    blocks = [b for b in pres.get("blocks") or [] if b.get("type") != "markdown"]
    if not blocks:
        return None
    return {"kind": pres.get("kind") or "mixed", "blocks": blocks}


def iter_structured_stream(
    question: str,
    *,
    user_context: Optional[UserContext],
    resolved_question: str,
) -> Iterator[str]:
    state: dict[str, Any] = {"question": resolved_question}
    if user_context is not None:
        state["user_context"] = user_context

    yield stream_event(type="status", phase="retrieval", agent="structured")
    partial = retrieve_node(state)
    retrieved = partial.get("retrieved_context") or {}
    chunks = retrieved.get("chunks") or []
    strategy = partial.get("strategy") or retrieved.get("strategy") or "text2cypher"

    yield stream_event(
        type="status",
        phase="retrieved",
        agent="structured",
        strategy=strategy,
        chunks=len(chunks),
    )

    if (retrieved.get("mode") or "") == "needs_clarification":
        answer = (
            (chunks[0].get("text") or "").strip()
            if chunks
            else "I need one clarification before I can answer that."
        )
        yield stream_event(type="done", agent="structured", answer=answer, sources=chunks, strategy=strategy)
        return

    if not chunks:
        answer = "No matching records were found in the business database for that query."
        yield stream_event(type="done", agent="structured", answer=answer, sources=[], strategy=strategy)
        return

    denied = next((c for c in chunks if c.get("id") == "access_denied"), None)
    if denied:
        answer = (denied.get("text") or "Access denied for structured data.").strip()
        yield stream_event(type="done", agent="structured", answer=answer, sources=[], strategy=strategy)
        return

    if any(c.get("id") == "error" for c in chunks):
        err_chunk = next((c for c in chunks if c.get("id") == "error"), None)
        err_text = (err_chunk or {}).get("text") or "The database query failed."
        record_pipeline_step("structured.cypher", status="error", error=err_text[:500])
        answer = (
            "I couldn't run that query successfully.\n\n"
            f"{err_text}\n\n"
            "Try rephrasing the question or narrowing the filter."
        )
        yield stream_event(type="done", agent="structured", answer=answer, sources=chunks, strategy=strategy)
        return

    viz = _viz_blocks_only(question, chunks)
    if viz:
        yield stream_event(type="presentation", partial=True, agent="structured", blocks=viz["blocks"])

    if _should_fast_structured_answer(chunks, strategy):
        answer = _build_fast_structured_answer(chunks, strategy, question)
        presentation = build_presentation(
            question=question,
            answer=answer,
            sources=chunks,
            retrieved_context=retrieved,
            agent="structured",
        )
        if presentation and presentation.get("blocks") and not viz:
            yield stream_event(
                type="presentation",
                partial=True,
                agent="structured",
                blocks=presentation["blocks"],
            )
        yield stream_event(
            type="done",
            agent="structured",
            answer=answer,
            sources=chunks,
            strategy=strategy,
            presentation=presentation,
        )
        return

    context_lines = []
    for i, c in enumerate(chunks, 1):
        title = c.get("title", "Result")
        text = c.get("text", "")
        score = c.get("score")
        cypher = c.get("cypher", "")
        meta = f"#{i}"
        if score is not None:
            meta += f" | Score: {score}"
        if cypher:
            meta += f" | Query: {cypher}"
        context_lines.append(f"{meta}\nTitle: {title}\n{text}")
    context_text = "\n\n".join(context_lines)
    system_prompt = load_prompt("structured_synthesis", context=context_text, question=question)
    provider = get_model_provider()

    yield stream_event(type="status", phase="synthesis", agent="structured")
    parts: list[str] = []
    for delta in provider.chat_completion_stream(
        model=STRUCTURED_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(question)},
        ],
        max_tokens=estimate_structured_synthesis_max_tokens(
            question,
            chunk_count=len(chunks),
            default_max=STRUCTURED_SYNTHESIS_MAX_TOKENS,
            long_max=STRUCTURED_SYNTHESIS_LONG_MAX_TOKENS,
        ),
    ):
        parts.append(delta)
        yield stream_event(type="token", agent="structured", target="markdown", delta=delta)

    answer = "".join(parts).strip()
    presentation = build_presentation(
        question=question,
        answer=answer,
        sources=chunks,
        retrieved_context=retrieved,
        agent="structured",
    )
    yield stream_event(
        type="done",
        agent="structured",
        answer=answer,
        sources=chunks,
        strategy=strategy,
        presentation=presentation,
    )
