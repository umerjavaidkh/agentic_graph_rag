"""Both axes at once, for questions that genuinely need each.

Owns the thread pool that runs the two sides concurrently. That pool is
created once, here, because it is sized from config."""
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from ...unstructured.retrieval.graph import esg_agent
from ...structured.retrieval.graph import structured_agent
from ...shared.auth.roles import UserContext, DEFAULT_PUBLIC_CONTEXT
from ...shared.audit import AuditEventType, record_audit_event
from ...unstructured.presentation import build_presentation
from ...shared.auth.rbac_setup import GraphRBAC
from ...shared.config.settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from ...shared.feedback import resolve_query_tool
from ...shared.telemetry import clear_telemetry, get_telemetry, start_telemetry
from ...shared.conversation import get_turn, resolve_follow_up, save_turn
from ..routing import (
    TOOL_TO_AGENT,
    enforce_mode,
    is_structured_data_question,
    make_structured_access_denied_result,
    resolve_mode_override,
    run_via_mcp_tool,
    try_document_fallback,
)
from .rbac import _rbac_check
from .documents import search_documents
from .data import query_data
from ...shared.config.settings import DEFAULT_LANGUAGE


_hybrid_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="query_hybrid")


def query_hybrid(
    question: str,
    user_context: Optional[UserContext] = None,
    thread_id: str = "default",
    language: str = DEFAULT_LANGUAGE,
) -> dict:
    start_telemetry()
    state = {"question": question, "language": language}
    if user_context is not None:
        state["user_context"] = user_context

    # esg_agent and structured_agent are independent — data_result never
    # reads doc_result — so run them concurrently instead of back-to-back;
    # each is its own LLM+DB round trip, so this roughly halves this mode's
    # latency. get_telemetry()/pipeline_step() read a contextvars.ContextVar
    # that does NOT propagate into a new thread on its own (submit() alone
    # would silently drop one agent's telemetry, since get_telemetry()
    # would just see the ContextVar's default None there) — copy_context()
    # captures the current value explicitly so each submitted call sees the
    # same Telemetry object start_telemetry() already set above. Each
    # future gets its own copy (not one shared Context) because
    # Context.run() forbids being entered concurrently from two threads at
    # once.
    doc_ctx = contextvars.copy_context()
    data_ctx = contextvars.copy_context()
    doc_future = _hybrid_pool.submit(doc_ctx.run, esg_agent.invoke, state)
    data_future = _hybrid_pool.submit(data_ctx.run, structured_agent.invoke, state)
    doc_result = doc_future.result()
    data_result = data_future.result()
    data_pres = build_presentation(
        question=question,
        answer=data_result.get("answer", ""),
        sources=data_result.get("sources", []),
        agent="structured",
    )
    if data_pres and data_pres.get("blocks"):
        blocks = [
            {
                "type": "markdown",
                "content": f"### From Documents\n\n{doc_result.get('answer', '')}",
            },
            *data_pres["blocks"],
        ]
        presentation = {"kind": "mixed", "blocks": blocks}
    else:
        presentation = build_presentation(
            question=question,
            answer=doc_result.get("answer", ""),
            sources=doc_result.get("sources", []),
            retrieved_context=doc_result.get("retrieved_context", {}),
            query_type=doc_result.get("query_type"),
        )
    out = {
        "answer": (
            f"### From Documents:\n{doc_result.get('answer', '')}\n\n"
            f"### From Data:\n{data_result.get('answer', '')}"
        ),
        "sources": doc_result.get("sources", []),
        "document_sources": doc_result.get("sources", []),
        "data_sources": data_result.get("sources", []),
        "agent": "hybrid",
        "strategy": data_result.get("strategy", ""),
        "presentation": presentation,
        "_access_level": user_context.role.value if user_context else DEFAULT_PUBLIC_CONTEXT.role.value,
    }
    tel = get_telemetry()
    if tel is not None:
        out["_telemetry"] = tel.summary()
    clear_telemetry()
    return out
