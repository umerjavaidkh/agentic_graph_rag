"""Non-blocking audit event recorder."""
from __future__ import annotations

import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

from .config import AuditConfig
from .models import AuditEvent
from .store import get_audit_store

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="audit")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _write_event(event: AuditEvent) -> None:
    try:
        get_audit_store().record(event)
    except Exception:
        logger.debug("audit log write failed", exc_info=True)


def record_audit_event(
    *,
    event_type: str,
    user_id: str,
    tenant_id: str,
    role: str,
    request_id: Optional[str] = None,
    resource: Optional[str] = None,
    action: Optional[str] = None,
    result: str = "success",
    reason: Optional[str] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Fire-and-forget persistence of one audit event. No-op when disabled."""
    config = AuditConfig.load()
    if not config.enabled:
        return

    if action is not None and not config.store_question:
        action = None

    event = AuditEvent(
        event_id=uuid.uuid4().hex,
        ts=_utc_now_iso(),
        event_type=event_type,
        user_id=user_id,
        tenant_id=tenant_id,
        role=role,
        request_id=request_id,
        resource=resource,
        action=action,
        result=result,
        reason=reason,
        metadata=metadata or {},
    )
    _executor.submit(_write_event, event)
