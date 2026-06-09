"""Shared query-tool resolution for sync and streaming paths."""
from __future__ import annotations

from ..conversation import get_turn, resolve_follow_up
from ..routing import select_mcp_tool
from ..telemetry import pipeline_step
from .routing.service import get_feedback_routing

_DOCUMENT_FOLLOW_UPS = frozenset(
    {"subsection_detail", "page", "page_visual_focus", "clarification_document"}
)


def resolve_query_tool(question: str, thread_id: str) -> tuple[str, dict]:
    """
    Resolve MCP tool from follow-up context, baseline routing, and feedback hints.

    Used by both router.ask() and streaming query orchestrator.
    """
    prior = get_turn(thread_id)
    tool_name: str | None = None
    resolved: dict = {"question": question, "use_prior": False}

    with pipeline_step("route.select"):
        if prior:
            resolved = resolve_follow_up(question, prior)
            if resolved.get("use_prior"):
                follow_up_kind = resolved.get("follow_up_kind") or ""
                if follow_up_kind in _DOCUMENT_FOLLOW_UPS:
                    tool_name = "search_documents"
                elif follow_up_kind == "structured_clarification":
                    tool_name = "query_data"

        if not tool_name:
            baseline = select_mcp_tool(question)
            tool_name = get_feedback_routing().route_tool(question, baseline)

    return tool_name, resolved
