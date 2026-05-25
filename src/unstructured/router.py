"""
unstructured/router.py — Internal router for unstructured system

This module contains the internal routing logic for unstructured document
and structured graph systems. It exposes `ask()` and the MCP tool registry
but is intended to be used via the top-level `src.bridge` facade.
"""
import re
from typing import Literal

# ── System imports ─────────────────────────────────────────────
from .graph import esg_agent        # unstructured
from ..structured.graph import structured_agent  # structured


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
def search_documents(question: str) -> dict:
    result = esg_agent.invoke({"question": question})
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "agent": "unstructured",
    }


def query_data(question: str) -> dict:
    result = structured_agent.invoke({"question": question})
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", []),
        "strategy": result.get("strategy", ""),
        "agent": "structured",
    }


def query_hybrid(question: str) -> dict:
    doc_result = esg_agent.invoke({"question": question})
    data_result = structured_agent.invoke({"question": question})
    return {
        "answer": f"### From Documents:\n{doc_result.get('answer', '')}\n\n" \
                  f"### From Data:\n{data_result.get('answer', '')}",
        "document_sources": doc_result.get("sources", []),
        "data_sources": data_result.get("sources", []),
        "agent": "hybrid",
    }


def ask(question: str) -> dict:
    route = route_query(question)
    if route == "structured":
        return query_data(question)
    if route == "hybrid":
        return query_hybrid(question)
    return search_documents(question)


MCP_TOOLS = [
    {"name": "search_documents", "description": "Search compliance documents and policies.", "fn": search_documents},
    {"name": "query_data", "description": "Query structured data.", "fn": query_data},
    {"name": "query_hybrid", "description": "Query both documents and data.", "fn": query_hybrid},
]
