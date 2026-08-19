"""Concrete routing policies — register new ones here, not in call sites."""
from __future__ import annotations

from typing import Optional

from ..models import AggregateDimension
from .base import PolicyDecision, RoutingPolicy

VALID_ROUTE_TOOLS = frozenset({"search_documents", "query_data", "query_hybrid"})
DOCUMENT_MODES = frozenset({"graph_rag", "graph_rag_hybrid", "graph_rag_lexical"})


class StructuredPathPolicy:
    """Prefer multistep planner vs single-shot Text-to-Cypher."""

    name = "structured_path"
    hint_agent = "structured"

    def decide(self, hint) -> Optional[PolicyDecision]:
        mode = hint.mode
        if mode.startswith("structured_multistep"):
            return PolicyDecision(action="multistep_first", value="multistep_first")
        if mode in ("text2cypher", "cypher"):
            return PolicyDecision(action="text2cypher_first", value="text2cypher_first")
        return None


class DocumentModePolicy:
    """Prefer document hybrid merge mode."""

    name = "document_mode"
    hint_agent = "unstructured"

    def decide(self, hint) -> Optional[PolicyDecision]:
        if hint.mode in DOCUMENT_MODES:
            return PolicyDecision(action=f"prefer_{hint.mode}", value=hint.mode)
        return None


class RouteToolPolicy:
    """Prefer MCP route tool when misroutes are labeled in feedback."""

    name = "route_tool"
    hint_agent = ""
    dimension = AggregateDimension.ROUTE_TOOL

    def decide(self, hint) -> Optional[PolicyDecision]:
        if hint.mode in VALID_ROUTE_TOOLS:
            return PolicyDecision(action="override_route", value=hint.mode)
        return None
