"""
LLM-based query routing (MCP tool selection).

Replaces keyword heuristics: the model picks one of the registered MCP tools.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any, Callable, Optional

from .auth.roles import UserContext
from .config.prompts import load_prompt
from .config.settings import (
    FAST_ROUTE_QUERIES,
    MODEL_PROVIDER,
    OPENAI_API_KEY,
    ROUTING_MODEL,
    estimate_route_max_tokens,
)
from .model_providers.factory import get_model_provider
from .telemetry import get_telemetry, pipeline_step

logger = logging.getLogger(__name__)

# OpenAI function specs for MCP-style tool routing
MCP_ROUTE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search ingested PDF/DOCX documents: policies, annual reports, manuals, "
                "photo credits, appendices, annual reports, manuals, sections, "
                "tables in documents, and any factual answer that comes from document content—not "
                "the Northwind product/order database."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The user's question verbatim"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_data",
            "description": (
                "Query the Northwind-style business graph ONLY: products, orders, customers, "
                "suppliers, sales analytics, Cypher/schema. Never use for PDF text, photo credits, "
                "photographers, or ingested report/PDF content."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The user's question verbatim"},
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_hybrid",
            "description": (
                "Query BOTH documents and structured graph data when the user clearly needs both."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "The user's question verbatim"},
                },
                "required": ["question"],
            },
        },
    },
]

TOOL_TO_AGENT: dict[str, str] = {
    "search_documents": "unstructured",
    "query_data": "structured",
    "query_hybrid": "hybrid",
}

_DATA_ROUTE = re.compile(
    r"\b(?:products?|orders?|customers?|suppliers?|categories?|category|beverages?|sales|"
    r"revenue|profit|sold|northwind|top\s+\d+|best(?:\s+selling)?|most\s+(?:sold|popular)|"
    r"cypher|neo4j|how\s+many\s+(?:orders?|products?|customers?|suppliers?|units?)|"
    r"count\s+of\s+(?:orders?|products?|customers?|suppliers?)|"
    r"belong\s+to\s+(?:the\s+)?\w+\s+categor|aggregate|schema|monthly|timeline|trend|"
    r"volume|chronological)\b",
    re.I,
)


def is_structured_data_question(question: str) -> bool:
    """Northwind-style business graph (products, orders, …) — not PDF documents."""
    return bool(_DATA_ROUTE.search(question or ""))


def has_document_cue(question: str) -> bool:
    """True when the question clearly references documents/PDF/sections (not business analytics)."""
    return bool(_DOC_ROUTE.search(question or ""))
_DOC_ROUTE = re.compile(
    r"\b(?:policy|policies|document|documents|pdf|manual|protocol|section\s+\d|"
    r"whistleblow|compliance\s+officer|procedure|page\s+\d+|figure|table\s+on|"
    r"table\s+of\s+contents?|toc|annex|appendix|acknowledgement|preface|chapter|"
    r"report|reports|annual\s+report|workshop|translated|translation|languages?|"
    r"institution|hosted|"
    r"photo|photograph|credit|photographer|illustration|attribution|caption|"
    r"identify\s+all|list\s+all|enumerate)\b",
    re.I,
)


def _result_has_structured_access_denied(result: dict) -> bool:
    ctx = result.get("retrieved_context") or {}
    for c in ctx.get("chunks") or []:
        if c.get("id") == "access_denied":
            return True
    for s in result.get("sources") or []:
        if s.get("id") == "access_denied":
            return True
    answer = (result.get("answer") or "").lower()
    return "permission to query structured" in answer or "access denied" in answer


def _fast_route_tool(question: str) -> Optional[str]:
    if not FAST_ROUTE_QUERIES:
        return None
    doc = bool(_DOC_ROUTE.search(question))
    data = bool(_DATA_ROUTE.search(question))
    if data and doc:
        return None
    if data:
        return "query_data"
    if doc:
        return "search_documents"
    return None


def select_mcp_tool(
    question: str,
    *,
    provider_name: str = MODEL_PROVIDER,
    api_key: str = OPENAI_API_KEY,
    model: str = ROUTING_MODEL,
) -> str:
    """
    Ask the LLM which MCP tool to invoke. Returns tool name (e.g. search_documents).
    """
    routed = _fast_route_tool(question)
    if routed:
        return routed

    # Business-graph signals without document cues → structured (before LLM mis-routes).
    if is_structured_data_question(question) and not _DOC_ROUTE.search(question):
        return "query_data"

    # Document signals without Northwind analytics → documents (before LLM mis-routes).
    if _DOC_ROUTE.search(question) and not is_structured_data_question(question):
        return "search_documents"

    if not api_key:
        return "search_documents"

    provider = get_model_provider(provider_name, api_key)
    system_prompt = load_prompt("route_query")

    try:
        response = provider.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            tools=MCP_ROUTE_TOOLS,
            tool_choice="required",
            temperature=0,
            max_tokens=estimate_route_max_tokens(question),
        )
        choice = response.choices[0]
        tool_calls = getattr(choice.message, "tool_calls", None) or []
        if tool_calls:
            name = tool_calls[0].function.name
            if name in TOOL_TO_AGENT:
                return name
        content = (choice.message.content or "").strip()
        if content:
            parsed = json.loads(content)
            if isinstance(parsed, dict) and parsed.get("tool") in TOOL_TO_AGENT:
                return parsed["tool"]
    except Exception as exc:
        logger.warning("LLM route selection failed: %s", exc)

    return "search_documents"


def run_via_mcp_tool(
    question: str,
    tool_name: str,
    handlers: dict[str, Callable[..., dict]],
    user_context: Optional[UserContext] = None,
    thread_id: str = "default",
) -> dict:
    """Execute the chosen MCP tool handler."""
    fn = handlers.get(tool_name) or handlers["search_documents"]
    with pipeline_step("agent.invoke", tool=tool_name):
        result = fn(question, user_context=user_context, thread_id=thread_id)
    route_method = "fast" if _fast_route_tool(question) else "llm_mcp"

    if tool_name == "query_data" and _result_has_structured_access_denied(result):
        if is_structured_data_question(question):
            result = make_structured_access_denied_result(
                question, user_context, routed_tool="query_data"
            )
            tool_name = "query_data"
            route_method = "structured_access_denied"
        else:
            logger.info("Routing fallback: query_data access denied → search_documents")
            with pipeline_step("agent.fallback", from_tool="query_data", to_tool="search_documents"):
                doc_fn = handlers.get("search_documents") or fn
                result = doc_fn(question, user_context=user_context, thread_id=thread_id)
            tool_name = "search_documents"
            route_method = "fallback_structured_denied"

    result["_route_tool"] = tool_name
    result["_route_method"] = route_method
    tel = get_telemetry()
    if tel is not None:
        tel.set_route(tool_name, route_method, agent=result.get("agent"), strategy=result.get("strategy"))
    return result


def make_structured_access_denied_result(
    question: str,
    user_context: Optional[UserContext],
    *,
    routed_tool: str = "query_data",
) -> dict:
    """Clear response when the user lacks RBAC for the business database."""
    uid = user_context.user_id if user_context else "unknown"
    answer = (
        "This question requires the business database (products, orders, customers, suppliers). "
        f"Your account ({uid}) does not have permission to query that data. "
        "Use a user with structured access, for example: regular_001, compliance_001, or admin_001."
    )
    return {
        "answer": answer,
        "sources": [],
        "keywords": [],
        "agent": "structured",
        "strategy": "access_denied",
        "query_type": "access_denied",
        "presentation": {"kind": "plain", "blocks": [{"type": "markdown", "content": answer}]},
        "retrieved_context": {"chunks": [], "query": question},
        "_route_tool": routed_tool,
        "_route_method": "structured_access_denied",
        "_access_level": user_context.role.value if user_context else None,
    }
