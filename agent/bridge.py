"""
agent/bridge.py — Unified MCP Bridge

Single entry point for the entire project.
Routes queries to the correct agent:
  - UnstructuredAgent  → document/policy questions
  - StructuredAgent    → data/analytics questions
  - Hybrid             → questions needing both

This is the layer MCP tools call — one bridge, two agents.
"""
import re
from typing import Literal

# ── Agent imports ─────────────────────────────────────────────
from agent.graph      import esg_agent        # unstructured
from structured.graph import structured_agent  # structured


# ─────────────────────────────────────────
# ROUTING SIGNALS
# ─────────────────────────────────────────
# router.py — top-level query classifier

STRUCTURED_SIGNALS = {
    # Entities
    "customer", "customers", "product", "products", "order", "orders",
    "supplier", "suppliers", "category", "categories", "employee", "employees",
    "shipper", "shippers", "department", "departments",
    # Actions / patterns
    "bought together", "frequently purchased", "most common", "best selling",
    "total sales", "total revenue", "average", "count", "sum", "how many",
    "who ordered", "who bought", "ordered by", "supplied by", "belongs to",
    "relationships between", "connected to", "linked to", "graph", "nodes",
    # Schema / meta
    "what data", "what tables", "show schema", "database", "available data",
    "what can i query", "what labels", "what nodes",
}

UNSTRUCTURED_SIGNALS = {
    "compliance", "esg", "stratec", "code of conduct", "antitrust",
    "conflict of interest", "corruption", "money laundering", "insider trading",
    "environmental protection", "equal opportunities", "employment security",
    "gift", "invitation", "reporting obligation", "whistleblower",
    "section", "chapter", "document", "policy", "guideline", "principle",
}

def route_query(question: str) -> str:
    """
    Returns: 'structured' | 'unstructured'
    """
    q = question.lower()

    struct_score = sum(1 for s in STRUCTURED_SIGNALS if s in q)
    unstruct_score = sum(1 for s in UNSTRUCTURED_SIGNALS if s in q)

    # Tie-breaker: structured wins if both score (graph questions are more specific)
    if struct_score > 0 and struct_score >= unstruct_score:
        return "structured"
    if unstruct_score > 0:
        return "unstructured"
    
    # Default: if no strong signal, check for aggregation/lookup patterns
    if any(w in q for w in ["total", "count", "sum", "average", "most", "top", "which"]):
        return "structured"
    
    return "unstructured"

HYBRID_SIGNALS = {
    "compare", "vs", "versus", "against target",
    "benchmark", "performance vs",
}

# ─────────────────────────────────────────
# MCP TOOLS  (exposed as callable tools)
# ─────────────────────────────────────────
def search_documents(question: str) -> dict:
    """
    MCP Tool — Search unstructured documents (policies, compliance docs).
    Use for: 'what does the policy say about X', 'explain section Y'
    """
    result = esg_agent.invoke({"question": question})
    return {
        "answer":   result.get("answer", ""),
        "sources":  result.get("sources", []),
        "agent":    "unstructured",
    }


def query_data(question: str) -> dict:
    """
    MCP Tool — Query structured graph data (sales, products, orders).
    Use for: 'total sales by category', 'which products are discontinued'
    """
    result = structured_agent.invoke({"question": question})
    return {
        "answer":   result.get("answer", ""),
        "sources":  result.get("sources", []),
        "strategy": result.get("strategy", ""),
        "agent":    "structured",
    }


def query_hybrid(question: str) -> dict:
    """
    MCP Tool — Query both agents and combine answers.
    Use for: 'compare our sales performance against compliance targets'
    """
    doc_result  = esg_agent.invoke({"question": question})
    data_result = structured_agent.invoke({"question": question})

    return {
        "answer":          f"### From Documents:\n{doc_result.get('answer', '')}\n\n"
                           f"### From Data:\n{data_result.get('answer', '')}",
        "document_sources": doc_result.get("sources", []),
        "data_sources":     data_result.get("sources", []),
        "agent":            "hybrid",
    }


# ─────────────────────────────────────────
# UNIFIED ENTRY POINT
# ─────────────────────────────────────────
def ask(question: str) -> dict:
    """
    Single entry point — auto-routes to correct agent.
    Call this from your FastAPI endpoint or MCP server.

    Usage:
        from agent.bridge import ask
        result = ask("total orders by category")
        result = ask("what does the compliance policy say about gifts?")
    """
    route = route_query(question)

    if route == "structured":
        return query_data(question)
    elif route == "hybrid":
        return query_hybrid(question)
    else:
        return search_documents(question)


# ─────────────────────────────────────────
# MCP TOOL REGISTRY  (for LangGraph tool node)
# ─────────────────────────────────────────
MCP_TOOLS = [
    {
        "name":        "search_documents",
        "description": "Search compliance documents and policies. Use for policy questions, regulations, guidelines.",
        "fn":           search_documents,
    },
    {
        "name":        "query_data",
        "description": "Query structured data. Use for sales totals, product lookups, order counts, aggregations.",
        "fn":           query_data,
    },
    {
        "name":        "query_hybrid",
        "description": "Query both documents and data. Use for comparisons between policy targets and actual data.",
        "fn":           query_hybrid,
    },
]