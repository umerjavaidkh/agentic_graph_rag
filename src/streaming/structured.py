"""Structured agent streaming: chart/table first, then narrative tokens."""
from __future__ import annotations

import json
from typing import Any, Iterator, Optional

from ..auth.roles import UserContext
from ..config.prompts import load_prompt
from ..config.settings import (
    STRUCTURED_MODEL,
    STRUCTURED_SYNTHESIS_LONG_MAX_TOKENS,
    STRUCTURED_SYNTHESIS_MAX_TOKENS,
)
from ..model_providers.factory import get_chat_provider
from ..presentation import build_presentation
from ..presentation.structured_planner import build_structured_presentation
from ..retrieval.structured.graph import (
    _build_fast_structured_answer,
    _should_fast_structured_answer,
    retrieve_node,
)
from ..retrieval.structured.query_intent import estimate_structured_synthesis_max_tokens
from ..retrieval.structured.verification import _COUNT_WORDS, compute_confidence
from ..telemetry.pipeline import record_pipeline_step
from .events import stream_event


def _try_document_fallback_stream(
    question: str, user_context: Optional[UserContext]
) -> Optional[tuple[list[str], Iterator[str]]]:
    """
    Peek at document retrieval before committing to it as a fallback.

    Only consumes iter_document_stream's cheap retrieval-only status event
    first (no LLM synthesis cost yet) — if it found no usable chunks,
    returns None so the caller keeps its original structured answer instead
    of streaming a worse guess. Returns the small buffered event list plus
    the rest of the generator so nothing already consumed is lost.
    """
    from .document import iter_document_stream  # lazy: avoids document.py <-> structured.py cycle

    gen = iter_document_stream(
        question,
        user_context=user_context,
        resolved_question=question,
        skip_structured_guard=True,
    )
    buffered: list[str] = []
    for line in gen:
        buffered.append(line)
        payload = json.loads(line)
        if payload.get("type") == "status" and payload.get("phase") == "retrieved":
            if not payload.get("chunks"):
                return None
            return buffered, gen
        if payload.get("type") == "done":
            # Denied / no-chunks short-circuit before a "retrieved" event.
            if payload.get("agent") == "unstructured" and payload.get("sources"):
                return buffered, gen
            return None
    return None


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
        # Zero rows for a non-aggregate query — try documents before
        # answering flatly (same root cause as the graph.py non-streaming
        # fix: a named entity may only exist in ingested documents).
        if not _COUNT_WORDS.search(question or ""):
            fallback = _try_document_fallback_stream(question, user_context)
            if fallback is not None:
                buffered, rest = fallback
                yield stream_event(type="status", phase="reroute", agent="unstructured", reason="structured_no_results")
                yield from buffered
                yield from rest
                return
        answer = "No matching records were found in the business database for that query."
        yield stream_event(
            type="done",
            agent="structured",
            answer=answer,
            sources=[],
            strategy=strategy,
            low_confidence=not _COUNT_WORDS.search(question or ""),
        )
        return

    denied = next((c for c in chunks if c.get("id") == "access_denied"), None)
    if denied:
        # RBAC denial resolved deep inside the retrieval layer, not the
        # streaming orchestrator's own top-level pre-gate — same fallback:
        # the question may be answerable from documents regardless.
        fallback = _try_document_fallback_stream(question, user_context)
        if fallback is not None:
            buffered, rest = fallback
            yield stream_event(type="status", phase="reroute", agent="unstructured", reason="structured_access_denied")
            yield from buffered
            yield from rest
            return
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
        yield stream_event(
            type="done",
            agent="structured",
            answer=answer,
            sources=chunks,
            strategy=strategy,
            low_confidence=True,
            confidence_note=err_text[:200],
        )
        return

    provider = get_chat_provider()
    low_confidence, confidence_note = compute_confidence(
        question, chunks, provider=provider, model=STRUCTURED_MODEL
    )

    if low_confidence:
        # Known before any structured tokens are streamed — safe to try
        # documents now and switch streams entirely if they're usable,
        # rather than streaming a possibly-wrong structured answer first.
        fallback = _try_document_fallback_stream(question, user_context)
        if fallback is not None:
            buffered, rest = fallback
            yield stream_event(type="status", phase="reroute", agent="unstructured", reason="structured_low_confidence")
            yield from buffered
            yield from rest
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
            low_confidence=low_confidence,
            confidence_note=confidence_note,
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
        low_confidence=low_confidence,
        confidence_note=confidence_note,
    )
