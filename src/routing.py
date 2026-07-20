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
from .audit import AuditEventType, record_audit_event
from .config.prompts import load_prompt
from .config.settings import (
    DEFAULT_TENANT_ID,
    FAST_ROUTE_QUERIES,
    MODEL_PROVIDER,
    OPENAI_API_KEY,
    ROUTING_MODEL,
    estimate_route_max_tokens,
)
from .graph.constants import (
    DOC_REVISION_LABEL,
    DOCUMENT_LOGICAL_LABEL,
    INDEXED_NODE_CYPHER,
)
from .model_providers.factory import get_model_provider
from .telemetry import get_telemetry, pipeline_step

logger = logging.getLogger(__name__)

# Node labels that belong to the ingested-document tree, not the structured
# business graph — excluded when summarizing "what's in the structured graph"
# for router prompts, so that summary doesn't just list Document/Section/Page.
_DOCUMENT_GRAPH_LABELS = set(INDEXED_NODE_CYPHER.split("|")) | {
    DOCUMENT_LOGICAL_LABEL,
    DOC_REVISION_LABEL,
}

_structured_entities_cache: Optional[str] = None


def structured_entity_summary() -> str:
    """
    Human-readable list of the live graph's non-document node labels, used to
    describe the structured-data tool to the router LLM instead of a
    hardcoded domain name — so routing adapts to whatever schema is loaded
    rather than staying tied to this repo's Northwind demo data.

    Cached for process lifetime (schema changes are rare relative to query
    volume, same tradeoff as SchemaProvider's Text-to-Cypher schema cache).
    """
    global _structured_entities_cache
    if _structured_entities_cache is not None:
        return _structured_entities_cache
    try:
        from .graph.driver import get_neo4j_driver

        driver = get_neo4j_driver()
        with driver.session() as session:
            rows = session.run("CALL db.labels() YIELD label RETURN label ORDER BY label")
            labels = [r["label"] for r in rows if r["label"] not in _DOCUMENT_GRAPH_LABELS]
        _structured_entities_cache = ", ".join(labels) if labels else "structured graph data"
    except Exception as exc:
        logger.debug("structured_entity_summary: schema lookup failed: %s", exc)
        _structured_entities_cache = "structured graph data"
    return _structured_entities_cache


def clear_structured_entity_cache() -> None:
    """Reset the cached label summary (tests, or after a schema change)."""
    global _structured_entities_cache
    _structured_entities_cache = None


def _build_mcp_route_tools() -> list[dict[str, Any]]:
    """OpenAI function specs for MCP-style tool routing, built fresh per call
    so the query_data description reflects whatever graph is actually live."""
    entities = structured_entity_summary()
    return [
        {
            "type": "function",
            "function": {
                "name": "search_documents",
                "description": (
                    "Search ingested PDF/DOCX documents: policies, reports, manuals, "
                    "sections, tables, appendices, and any factual answer that comes "
                    "from document content — not the structured graph database."
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
                    f"Query the structured graph database ONLY — entities: {entities}. "
                    "Aggregations, rankings, counts, Cypher/schema questions. Never use "
                    "for narrative text inside ingested PDF/DOCX documents."
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
    r"\b(?:products?|orders?|customers?|suppliers?|categories?|category|sales|"
    r"revenue|profit|sold|top\s+\d+|best(?:\s+selling)?|most\s+(?:sold|popular)|"
    r"cypher|neo4j|how\s+many\s+(?:orders?|products?|customers?|suppliers?|units?)|"
    r"count\s+of\s+(?:orders?|products?|customers?|suppliers?)|"
    r"belong\s+to\s+(?:the\s+)?\w+\s+categor|aggregate|schema|monthly|timeline|trend|"
    r"volume|chronological|"
    r"structured\s+(?:data|graph|query)|graph\s+(?:data|query|analytics)|"
    r"analytics|metrics?|tabular|database\s+query|query\s+(?:the\s+)?(?:graph|database))\b",
    re.I,
)


def is_structured_data_question(question: str) -> bool:
    """Likely a structured graph / analytics query — not ingested PDF documents."""
    return bool(_DATA_ROUTE.search(question or ""))


def is_misrouted_structured_question(question: str) -> bool:
    """Document agent was chosen but the question looks like structured graph analytics."""
    return is_structured_data_question(question) and not has_document_cue(question)


def structured_misroute_message(user_id: str = "unknown") -> str:
    """User-facing hint when document routing cannot serve a structured-graph question."""
    return (
        "This question looks like a structured graph or analytics query, not a document search. "
        f"Your account ({user_id}) does not have permission to query the structured graph. "
        "Use a role with structured access, or rephrase as a document question (PDF, section, policy)."
    )


def user_can_query_structured(user_context: Optional[UserContext]) -> bool:
    if not user_context:
        return False
    from .router import _rbac_check

    return _rbac_check().can_query_knowledge_area(user_context.user_id, "structured")


def run_structured_autofix(question: str, user_context: Optional[UserContext]) -> Optional[dict]:
    """
    When routing sent a structured-graph question to the document agent, re-run on structured.

    Schema-agnostic: works for any Neo4j graph the structured retriever can query.
    Returns structured agent state dict, or None if not applicable / no access.
    """
    if not is_misrouted_structured_question(question):
        return None
    if not user_can_query_structured(user_context):
        return None
    from .retrieval.structured.graph import structured_agent

    state: dict[str, Any] = {"question": question}
    if user_context is not None:
        state["user_context"] = user_context
    return structured_agent.invoke(state)


def document_agent_structured_guard(
    question: str,
    user_context: Optional[UserContext],
) -> Optional[dict]:
    """
    Handle structured-shaped questions that reached the document pipeline.

    Auto-routes to structured when RBAC allows; otherwise returns a generic message dict.
    """
    if not is_misrouted_structured_question(question):
        return None
    fixed = run_structured_autofix(question, user_context)
    if fixed is not None:
        return {
            "answer": (fixed.get("answer") or "").strip(),
            "low_confidence": bool(fixed.get("low_confidence")),
            "confidence_note": fixed.get("confidence_note"),
            "sources": fixed.get("sources") or [],
            "strategy": fixed.get("strategy"),
            "_autofix_agent": "structured",
        }
    uid = user_context.user_id if user_context else "unknown"
    record_audit_event(
        event_type=AuditEventType.ACCESS_DENIED,
        user_id=uid,
        tenant_id=user_context.tenant_id if user_context else DEFAULT_TENANT_ID,
        role=user_context.role.value if user_context else "public",
        resource="structured",
        action=question,
        result="denied",
        reason="misrouted_no_access",
    )
    return {
        "answer": structured_misroute_message(uid),
        "low_confidence": False,
        "confidence_note": None,
    }


def has_document_cue(question: str) -> bool:
    """True when the question clearly references documents/PDF/sections (not business analytics)."""
    return bool(_DOC_ROUTE.search(question or ""))
_DOC_ROUTE = re.compile(
    r"\b(?:policy|policies|document|documents|pdf|manual|protocol|section\s+\d|"
    r"procedure|page\s+\d+|figure|table|workshop|"
    r"toc|annex|appendix|acknowledgement|preface|chapter|"
    r"report|reports|annual\s+report|"
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

    # Document signals without structured-graph analytics → documents (before LLM mis-routes).
    if _DOC_ROUTE.search(question) and not is_structured_data_question(question):
        return "search_documents"

    if not api_key:
        return "search_documents"

    provider = get_model_provider(provider_name, api_key)
    system_prompt = load_prompt("route_query", structured_entities=structured_entity_summary())

    try:
        response = provider.chat_completion(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
            tools=_build_mcp_route_tools(),
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
    """Clear response when the user lacks RBAC for the structured graph."""
    uid = user_context.user_id if user_context else "unknown"
    answer = (
        "This question requires the structured knowledge graph (analytics, entities, relationships). "
        f"Your account ({uid}) does not have permission to query that data. "
        "Use a role with structured graph access."
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
