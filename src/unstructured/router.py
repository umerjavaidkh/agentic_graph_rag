"""
unstructured/router.py — Internal router for unstructured / structured systems.

Exposes ask() and MCP tool registry via src.bridge.
Routing uses LLM MCP tool selection (no keyword lists).
"""
from typing import Optional

from .graph import esg_agent
from ..structured.graph import structured_agent
from ..auth.roles import UserContext, DEFAULT_PUBLIC_CONTEXT
from ..presentation import build_presentation
from ..routing import select_mcp_tool, run_via_mcp_tool
from ..conversation import clear_turn, get_turn, resolve_follow_up, route_tool_for_clarification_reply, save_turn


# ─────────────────────────────────────────
# MCP TOOLS
# ─────────────────────────────────────────
def search_documents(
    question: str,
    user_context: Optional[UserContext] = None,
    thread_id: str = "default",
) -> dict:
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
    presentation = build_presentation(
        question=question,
        answer=result.get("answer", ""),
        sources=result.get("sources", []),
        retrieved_context=result.get("retrieved_context", {}),
        query_type=result.get("query_type"),
    )
    out = {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "keywords": result.get("keywords", []),
        "agent": "unstructured",
        "strategy": result.get("query_type", "semantic"),
        "query_type": result.get("query_type"),
        "presentation": presentation,
        "retrieved_context": result.get("retrieved_context", {}),
        "_access_level": user_context.role.value if user_context else DEFAULT_PUBLIC_CONTEXT.role.value,
        "_follow_up": resolved.get("follow_up_kind") if resolved.get("use_prior") else None,
    }
    if resolved.get("follow_up_kind") in ("clarification_document", "structured_clarification") and prior:
        pending = prior.get("pending_clarification")
        if pending:
            out["_resolved_clarification"] = pending
    save_turn(thread_id, question, out)
    return out


def query_data(question: str, user_context: Optional[UserContext] = None, thread_id: str = "default") -> dict:
    prior = get_turn(thread_id)
    resolved = resolve_follow_up(question, prior)

    state = {"question": resolved["question"]}
    if user_context is not None:
        state["user_context"] = user_context

    result = structured_agent.invoke(state)
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
        "presentation": presentation,
        "retrieved_context": result.get("retrieved_context", {}),
        "_access_level": user_context.role.value if user_context else DEFAULT_PUBLIC_CONTEXT.role.value,
        "_follow_up": resolved.get("follow_up_kind") if resolved.get("use_prior") else None,
    }
    if resolved.get("follow_up_kind") in ("clarification_document", "structured_clarification") and prior:
        pending = prior.get("pending_clarification")
        if pending:
            out["_resolved_clarification"] = pending
    save_turn(thread_id, question, out)
    return out


def query_hybrid(question: str, user_context: Optional[UserContext] = None, thread_id: str = "default") -> dict:
    state = {"question": question}
    if user_context is not None:
        state["user_context"] = user_context

    doc_result = esg_agent.invoke(state)
    data_result = structured_agent.invoke(state)
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
    return {
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


def ask(question: str, user_context: Optional[UserContext] = None, thread_id: str = "default") -> dict:
    prior = get_turn(thread_id)
    tool_name = route_tool_for_clarification_reply(question, prior)
    if not tool_name and prior:
        resolved = resolve_follow_up(question, prior)
        if resolved.get("use_prior"):
            fk = resolved.get("follow_up_kind") or ""
            if fk in (
                "clarification_document",
                "subsection_detail",
                "page",
                "page_visual_focus",
            ):
                tool_name = "search_documents"
            elif fk == "structured_clarification":
                tool_name = "query_data"
    if not tool_name:
        tool_name = select_mcp_tool(question)
    return run_via_mcp_tool(
        question,
        tool_name,
        MCP_HANDLERS,
        user_context=user_context,
        thread_id=thread_id,
    )
