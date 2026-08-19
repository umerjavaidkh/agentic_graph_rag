"""The unstructured half: answering from ingested documents.

Nothing here touches the business graph. The structured autofix it can
trigger lives in routing, not in this module -- the handler's job ends at
producing a document answer."""
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


def search_documents(
    question: str,
    user_context: Optional[UserContext] = None,
    thread_id: str = "default",
) -> dict:
    start_telemetry()
    prior = get_turn(thread_id)
    resolved = resolve_follow_up(question, prior)

    state = {"question": resolved["question"]}
    if user_context is not None:
        state["user_context"] = user_context
    if resolved.get("focus_section_id"):
        state["focus_section_id"] = resolved["focus_section_id"]
        state["parent_section_id"] = resolved.get("parent_section_id")
    if resolved.get("document_id"):
        state["document_id"] = resolved["document_id"]
    if prior:
        state["prior_context"] = prior

    result = esg_agent.invoke(state)
    # A structured-shaped question can reach here (default retrieval mode,
    # or an RBAC denial on document access) and get silently re-answered by
    # the structured agent via document_agent_structured_guard/
    # run_structured_autofix (see retrieval/unstructured/graph.py's
    # _generate_document_answer). Without checking for that marker, this
    # function always reported "agent": "unstructured" / route_tool
    # "search_documents" even when the structured agent produced the real
    # answer -- misleading callers (and eval suites) that key off those
    # fields to know which agent actually served the question.
    autofixed_structured = result.get("_autofix_agent") == "structured"
    presentation = build_presentation(
        question=question,
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        retrieved_context=result.get("retrieved_context", {}),
        query_type=result.get("query_type"),
        agent="structured" if autofixed_structured else None,
    )
    out = {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "keywords": result.get("keywords", []),
        "agent": "structured" if autofixed_structured else "unstructured",
        "strategy": (result.get("strategy") or "structured") if autofixed_structured else result.get("query_type", "semantic"),
        "query_type": result.get("query_type"),
        "low_confidence": bool(result.get("low_confidence")),
        "confidence_note": result.get("confidence_note"),
        "presentation": presentation,
        "retrieved_context": result.get("retrieved_context", {}),
        # Which claim each page supports, so a reader can check one sentence
        # rather than the whole source list. Additive: callers that only read
        # "sources" are unaffected.
        "claims": result.get("claims", []),
        "_access_level": user_context.role.value if user_context else DEFAULT_PUBLIC_CONTEXT.role.value,
        "_follow_up": resolved.get("follow_up_kind") if resolved.get("use_prior") else None,
    }
    if autofixed_structured:
        out["_autofix_agent"] = "structured"
    tel = get_telemetry()
    if tel is not None:
        out["_telemetry"] = tel.summary()
    clear_telemetry()
    save_turn(thread_id, question, out)
    return out
