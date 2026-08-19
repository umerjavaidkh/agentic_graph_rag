"""
router.py — Query router for structured and unstructured retrieval.

Exposes ask() and MCP tool registry via src.bridge.
Routing uses LLM MCP tool selection (no keyword lists).
"""
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from .unstructured.retrieval.graph import esg_agent
from .retrieval.structured.graph import structured_agent
from .shared.auth.roles import UserContext, DEFAULT_PUBLIC_CONTEXT
from .shared.audit import AuditEventType, record_audit_event
from .unstructured.presentation import build_presentation
from .shared.auth.rbac_setup import GraphRBAC
from .shared.config.settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from .shared.feedback import resolve_query_tool
from .shared.telemetry import clear_telemetry, get_telemetry, start_telemetry

_rbac: GraphRBAC | None = None

# Set by ask() when the caller named a retrieval mode instead of letting the
# router pick. query_data() is reached through the MCP handler registry, whose
# signature is fixed, so this travels out-of-band rather than as an argument.
_MODE_LOCKED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "retrieval_mode_locked", default=False
)

# Shared across query_hybrid calls rather than created fresh per request —
# same reasoning as FullHybridStrategy's pool (retrieval/unstructured/
# strategies/full_hybrid.py).
_hybrid_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="query_hybrid")


def _rbac_check() -> GraphRBAC:
    global _rbac
    if _rbac is None:
        _rbac = GraphRBAC(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    return _rbac
from .shared.conversation import get_turn, resolve_follow_up, save_turn
from .routing import (
    TOOL_TO_AGENT,
    enforce_mode,
    is_structured_data_question,
    make_structured_access_denied_result,
    resolve_mode_override,
    run_via_mcp_tool,
    try_document_fallback,
)


# ─────────────────────────────────────────
# MCP TOOLS
# ─────────────────────────────────────────
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


def query_hybrid(question: str, user_context: Optional[UserContext] = None, thread_id: str = "default") -> dict:
    start_telemetry()
    state = {"question": question}
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


MCP_HANDLERS = {
    "search_documents": search_documents,
    "query_data": query_data,
    "query_hybrid": query_hybrid,
}

MCP_TOOLS = [
    {
        "name": "search_documents",
        "description": (
            "Search policy/compliance documents for procedures, protocols, sections, "
            "reporting rules, officer duties, whistleblowing, and organizational policy text."
        ),
        "fn": search_documents,
    },
    {
        "name": "query_data",
        "description": "Query structured Neo4j data (products, orders, customers, analytics, schema).",
        "fn": query_data,
    },
    {
        "name": "query_hybrid",
        "description": "Query both documents and structured data when both are required.",
        "fn": query_hybrid,
    },
]


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
