"""
unstructured/router.py — Internal router for unstructured system

This module contains the internal routing logic for unstructured document
and structured graph systems. It exposes `ask()` and the MCP tool registry
but is intended to be used via the top-level `src.bridge` facade.

NOW WITH ROLE-BASED ACCESS CONTROL (RBAC)
"""
import re
from typing import Literal, Optional

# ── System imports ─────────────────────────────────────────────
from .graph import esg_agent        # unstructured
from ..structured.graph import structured_agent  # structured
from ..auth.roles import UserContext, Role, DEFAULT_PUBLIC_CONTEXT


# ─────────────────────────────────────────
# ROUTING SIGNALS
# ─────────────────────────────────────────
STRUCTURED_SIGNALS = {
    "customer", "customers", "product", "products", "order", "orders",
    "supplier", "suppliers", "category", "categories", "employee", "employees",
    "shipper", "shippers", "department", "departments",
    "bought together", "frequently purchased", "most common", "best selling",
    "total sales", "total revenue", "average", "count", "sum", "how many",
    "who ordered", "who bought", "ordered by", "supplied by", "belongs to",
    "relationships between", "connected to", "linked to", "graph", "nodes",
    "what data", "what tables", "show schema", "database", "available data",
}

UNSTRUCTURED_SIGNALS = {
    "compliance", "esg", "stratec", "code of conduct", "antitrust",
    "conflict of interest", "corruption", "money laundering", "insider trading",
    "environmental protection", "equal opportunities", "employment security",
    "gift", "invitation", "reporting obligation", "whistleblower",
    "section", "chapter", "document", "policy", "guideline", "principle",
}


def route_query(question: str) -> str:
    q = question.lower()
    struct_score = sum(1 for s in STRUCTURED_SIGNALS if s in q)
    unstruct_score = sum(1 for s in UNSTRUCTURED_SIGNALS if s in q)

    if struct_score > 0 and struct_score >= unstruct_score:
        return "structured"
    if unstruct_score > 0:
        return "unstructured"

    if any(w in q for w in ["total", "count", "sum", "average", "most", "top", "which"]):
        return "structured"

    return "unstructured"


HYBRID_SIGNALS = {
    "compare", "vs", "versus", "against target",
    "benchmark", "performance vs",
}


# ─────────────────────────────────────────
# MCP TOOLS
# ─────────────────────────────────────────
def search_documents(question: str, user_context: Optional[UserContext] = None) -> dict:
    state = {"question": question}
    if user_context is not None:
        state["user_context"] = user_context

    result = esg_agent.invoke(state)
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "agent": "unstructured",
        "_access_level": user_context.role.value if user_context else DEFAULT_PUBLIC_CONTEXT.role.value,
    }


def query_data(question: str, user_context: Optional[UserContext] = None) -> dict:
    state = {"question": question}
    if user_context is not None:
        state["user_context"] = user_context

    result = structured_agent.invoke(state)
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "strategy": result.get("strategy", ""),
        "agent": "structured",
        "_access_level": user_context.role.value if user_context else DEFAULT_PUBLIC_CONTEXT.role.value,
    }


def query_hybrid(question: str, user_context: Optional[UserContext] = None) -> dict:
    state = {"question": question}
    if user_context is not None:
        state["user_context"] = user_context

    doc_result = esg_agent.invoke(state)
    data_result = structured_agent.invoke(state)
    return {
        "answer": f"### From Documents:\n{doc_result.get('answer', '')}\n\n" \
                  f"### From Data:\n{data_result.get('answer', '')}",
        "document_sources": doc_result.get("sources", []),
        "data_sources": data_result.get("sources", []),
        "agent": "hybrid",
        "_access_level": user_context.role.value if user_context else DEFAULT_PUBLIC_CONTEXT.role.value,
    }


def ask(question: str, user_context: Optional[UserContext] = None) -> dict:
    route = route_query(question)
    if route == "structured":
        return query_data(question, user_context=user_context)
    if route == "hybrid":
        return query_hybrid(question, user_context=user_context)
    return search_documents(question, user_context=user_context)


MCP_TOOLS = [
    {"name": "search_documents", "description": "Search compliance documents and policies.", "fn": search_documents},
    {"name": "query_data", "description": "Query structured data.", "fn": query_data},
    {"name": "query_hybrid", "description": "Query both documents and data.", "fn": query_hybrid},
]
