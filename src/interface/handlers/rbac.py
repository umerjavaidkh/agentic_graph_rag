"""The access check both handlers make before answering."""
import contextvars
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from ...unstructured.retrieval.graph import esg_agent
from ...structured.retrieval.graph import structured_agent
from ...shared.auth.roles import UserContext, DEFAULT_PUBLIC_CONTEXT
from ...shared.audit import AuditEventType, record_audit_event
from ...unstructured.presentation import build_presentation
from ...shared.auth.rbac_setup import GraphRBAC
from ...shared.config.settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from ...shared.feedback import resolve_query_tool
from ...shared.telemetry import clear_telemetry, get_telemetry, start_telemetry
from ...shared.conversation import get_turn, resolve_follow_up, save_turn
from ..routing import (
    TOOL_TO_AGENT,
    enforce_mode,
    is_structured_data_question,
    make_structured_access_denied_result,
    resolve_mode_override,
    run_via_mcp_tool,
    try_document_fallback,
)


# Lazily built once per process: `global _rbac` below binds HERE, so the
# declaration has to live in this module rather than the one it was cut from.
_rbac: GraphRBAC | None = None

def _rbac_check() -> GraphRBAC:
    global _rbac
    if _rbac is None:
        _rbac = GraphRBAC(NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
    return _rbac
