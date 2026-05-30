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
from .config.settings import CHAT_MODEL, FAST_ROUTE_QUERIES, MODEL_PROVIDER, OPENAI_API_KEY
from .model_providers.factory import get_model_provider

logger = logging.getLogger(__name__)

# OpenAI function specs for MCP-style tool routing
MCP_ROUTE_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "search_documents",
            "description": (
                "Search ingested policy/compliance documents (PDFs, manuals). "
                "Use for procedures, protocols, sections, whistleblowing, officer duties, "
                "reporting obligations, and any answer that should come from document text."
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
                "Query structured Neo4j graph data (products, orders, customers, suppliers). "
                "Use for analytics, counts, aggregations, schema questions—not policy documents."
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
    r"\b(?:products?|orders?|customers?|suppliers?|categories?|sales|revenue|profit|sold|"
    r"northwind|top\s+\d+|best(?:\s+selling)?|most\s+(?:sold|popular)|cypher|neo4j|"
    r"how\s+many|count|aggregate|schema|monthly|timeline|trend|volume|chronological)\b",
    re.I,
)
_DOC_ROUTE = re.compile(
    r"\b(?:policy|policies|document|documents|pdf|manual|protocol|section\s+\d|"
    r"whistleblow|compliance\s+officer|procedure|page\s+\d+|figure|table\s+on)\b",
    re.I,
)


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
    model: str = CHAT_MODEL,
) -> str:
    """
    Ask the LLM which MCP tool to invoke. Returns tool name (e.g. search_documents).
    """
    routed = _fast_route_tool(question)
    if routed:
        return routed

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
    result = fn(question, user_context=user_context, thread_id=thread_id)
    result["_route_tool"] = tool_name
    result["_route_method"] = "llm_mcp"
    return result
