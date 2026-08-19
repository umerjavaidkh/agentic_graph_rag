"""The structured half: answering from the business graph.

Text-to-Cypher over whatever schema is loaded. Nothing here reads an
ingested document."""
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
from ..context import _MODE_LOCKED
from .rbac import _rbac_check


def query_data(question: str, user_context: Optional[UserContext] = None, thread_id: str = "default") -> dict:
    start_telemetry()
    prior = get_turn(thread_id)
    resolved = resolve_follow_up(question, prior)

    state = {"question": resolved["question"]}
    if user_context is not None:
        state["user_context"] = user_context

    result = structured_agent.invoke(state)

    fallback_used = False
    # An RBAC denial arrives flagged low_confidence, but it is not a weak
    # answer -- it is "you may not ask this". Falling back to documents turns
    # it into "this document does not cover it", which describes the corpus
    # instead of the missing permission and leaves no way to tell the two
    # apart. Everything else still falls back: a genuinely weak structured
    # answer may well be in the documents.
    # Never reroute away from a mode the caller chose explicitly, and never
    # dress an RBAC denial up as a document answer: both replace what was
    # asked for with "this document does not cover it", which names the wrong
    # reason. A router-picked structured answer that came out weak may still
    # be better served by documents, so that case still falls back.
    denied = result.get("strategy") == "access_denied"
    if result.get("low_confidence") and not denied and not _MODE_LOCKED.get():
        fallback = try_document_fallback(resolved["question"], user_context)
        if fallback is not None:
            fallback_used = True
            result = {
                "answer": fallback.get("answer", ""),
                "sources": fallback.get("sources", []),
                "strategy": fallback.get("query_type", "graph_rag"),
                "low_confidence": False,
                "confidence_note": None,
                "retrieved_context": fallback.get("retrieved_context", {}),
            }

    presentation = build_presentation(
        question=question,
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        retrieved_context=result.get("retrieved_context", {}),
        agent="structured",
    )
    out = {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "strategy": result.get("strategy", ""),
        "agent": "structured",
        "low_confidence": bool(result.get("low_confidence")),
        "confidence_note": result.get("confidence_note"),
        "presentation": presentation,
        "retrieved_context": result.get("retrieved_context", {}),
        "_access_level": user_context.role.value if user_context else DEFAULT_PUBLIC_CONTEXT.role.value,
        "_follow_up": resolved.get("follow_up_kind") if resolved.get("use_prior") else None,
    }
    if fallback_used:
        out["_fallback_agent"] = "unstructured"
    tel = get_telemetry()
    if tel is not None:
        out["_telemetry"] = tel.summary()
    clear_telemetry()
    save_turn(thread_id, question, out)
    return out
