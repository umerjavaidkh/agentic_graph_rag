"""Dispatching a question to a retrieval axis.

The handlers themselves live in handlers/, one per axis -- documents, data,
hybrid -- because they share nothing but the access check. This module is
only the entry point: it takes a question, lets routing choose a tool, and
runs it.
"""
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from ..unstructured.retrieval.graph import esg_agent
from ..structured.retrieval.graph import structured_agent
from ..shared.auth.roles import UserContext, DEFAULT_PUBLIC_CONTEXT
from ..shared.audit import AuditEventType, record_audit_event
from ..unstructured.presentation import build_presentation
from ..shared.auth.rbac_setup import GraphRBAC
from ..shared.config.settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from ..shared.feedback import resolve_query_tool
from ..shared.telemetry import clear_telemetry, get_telemetry, start_telemetry
from ..shared.conversation import get_turn, resolve_follow_up, save_turn
from .routing import (
    TOOL_TO_AGENT,
    enforce_mode,
    is_structured_data_question,
    make_structured_access_denied_result,
    resolve_mode_override,
    run_via_mcp_tool,
    try_document_fallback,
)
from .context import _MODE_LOCKED
from .handlers import MCP_HANDLERS, MCP_TOOLS, _rbac_check, query_data, query_hybrid, search_documents

"""
router.py — Query router for structured and unstructured retrieval.

Exposes ask() and MCP tool registry via src.bridge.
Routing uses LLM MCP tool selection (no keyword lists).
"""






def ask(
    question: str,
    user_context: Optional[UserContext] = None,
    thread_id: str = "default",
    request_id: Optional[str] = None,
    retrieval_mode: Optional[str] = None,
) -> dict:
    start_telemetry()
    tel = get_telemetry()
    if tel is not None and request_id:
        tel.route["request_id"] = request_id

    try:
        forced_tool = resolve_mode_override(retrieval_mode)
        _MODE_LOCKED.set(forced_tool is not None)
        if tel is not None:
            tel.route["retrieval_mode"] = TOOL_TO_AGENT.get(forced_tool, forced_tool)
        tool_name, _resolved = resolve_query_tool(
            question, thread_id, forced_tool=forced_tool
        )

        ctx = user_context or DEFAULT_PUBLIC_CONTEXT
        if (
            tool_name == "query_data"
            and is_structured_data_question(question)
            and not _rbac_check().can_query_knowledge_area(ctx.user_id, "structured")
        ):
            # Structured access is denied outright — but the question being
            # phrased like analytics doesn't mean the entity actually lives
            # in the structured graph; it may only be in ingested documents,
            # which this user's role can separately have access to. Same
            # remedy as the low-confidence fallback in query_data(): try
            # documents before giving up, never worse than the flat denial.
            record_audit_event(
                event_type=AuditEventType.ACCESS_DENIED,
                user_id=ctx.user_id,
                tenant_id=ctx.tenant_id,
                role=ctx.role.value,
                request_id=request_id,
                resource="structured",
                action=question,
                result="denied",
                reason="rbac_denied",
            )
            # ...but only when the ROUTER chose structured. If the user picked
            # it explicitly, quietly answering from documents misreports what
            # happened: the reply reads "this document does not cover it",
            # which describes the corpus rather than the permission that was
            # actually missing, and gives no hint that access is the problem.
            # Observed with the UI's default user, who has no structured
            # access: every question asked on the Structured tab came back as
            # a document non-answer.
            fallback = None if forced_tool else try_document_fallback(question, ctx)
            if fallback is not None:
                presentation = build_presentation(
                    question=question,
                    answer=fallback.get("answer", ""),
                    sources=fallback.get("sources", []),
                    retrieved_context=fallback.get("retrieved_context", {}),
                )
                out = {
                    "answer": fallback.get("answer", ""),
                    "sources": fallback.get("sources", []),
                    "strategy": fallback.get("query_type", "graph_rag"),
                    "agent": "unstructured",
                    "low_confidence": False,
                    "confidence_note": None,
                    "presentation": presentation,
                    "retrieved_context": fallback.get("retrieved_context", {}),
                    "_access_level": ctx.role.value,
                    "_fallback_agent": "unstructured",
                }
                if tel is not None:
                    tel.set_route("query_data", "structured_denied_document_fallback")
                    out["_telemetry"] = tel.summary()
                clear_telemetry()
                save_turn(thread_id, question, out)
                record_audit_event(
                    event_type=AuditEventType.QUERY,
                    user_id=ctx.user_id,
                    tenant_id=ctx.tenant_id,
                    role=ctx.role.value,
                    request_id=request_id,
                    action=question,
                    result="success",
                    metadata={
                        "route_tool": "query_data",
                        "agent": "unstructured",
                        "strategy": out["strategy"],
                        "fallback_reason": "structured_rbac_denied",
                    },
                )
                return out

            out = make_structured_access_denied_result(question, ctx)
            if tel is not None:
                tel.set_route("query_data", "structured_access_denied")
                out["_telemetry"] = tel.summary()
            clear_telemetry()
            save_turn(thread_id, question, out)
            return out

        result = run_via_mcp_tool(
            question,
            tool_name,
            MCP_HANDLERS,
            user_context=user_context,
            thread_id=thread_id,
        )
        result = enforce_mode(result, forced_tool)
        record_audit_event(
            event_type=AuditEventType.QUERY,
            user_id=ctx.user_id,
            tenant_id=ctx.tenant_id,
            role=ctx.role.value,
            request_id=request_id,
            action=question,
            result="success",
            metadata={
                "route_tool": result.get("_route_tool"),
                "agent": result.get("agent"),
                "strategy": result.get("strategy"),
                "low_confidence": bool(result.get("low_confidence")),
            },
        )
        return result
    except Exception:
        raise
