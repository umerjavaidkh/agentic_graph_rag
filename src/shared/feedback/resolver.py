"""Shared query-tool resolution for sync and streaming paths."""
from __future__ import annotations

from ..conversation import get_turn, resolve_follow_up
from ...routing import select_mcp_tool
from ..telemetry import pipeline_step
from .routing.service import get_feedback_routing

_DOCUMENT_FOLLOW_UPS = frozenset(
    {"subsection_detail", "page", "page_visual_focus", "clarification_document"}
)


def resolve_query_tool(
    question: str,
    thread_id: str,
    forced_tool: str | None = None,
) -> tuple[str, dict]:
    """
    Resolve MCP tool from follow-up context, baseline routing, and feedback hints.

    Used by both router.ask() and streaming query orchestrator.

    When ``forced_tool`` is given (an explicit retrieval_mode override from the
    UI/API), that tool wins outright: follow-up resolution still runs so the
    turn inherits document continuity (document_id, focus_section_id, rewritten
    question), but neither the follow-up kind nor the baseline LLM router can
    change the agent. This keeps structured and unstructured retrieval fully
    separable on demand and skips the routing LLM call.
    """
    prior = get_turn(thread_id)
    tool_name: str | None = None
    resolved: dict = {"question": question, "use_prior": False}

    with pipeline_step("route.select", forced=bool(forced_tool)):
        if prior:
            resolved = resolve_follow_up(question, prior)
            if resolved.get("use_prior") and not forced_tool:
                follow_up_kind = resolved.get("follow_up_kind") or ""
                if follow_up_kind in _DOCUMENT_FOLLOW_UPS:
                    tool_name = "search_documents"
                elif follow_up_kind == "structured_clarification":
                    tool_name = "query_data"

        if forced_tool:
            return forced_tool, resolved

        if not tool_name:
            baseline = select_mcp_tool(question)
            tool_name = get_feedback_routing().route_tool(question, baseline)

    return tool_name, resolved
