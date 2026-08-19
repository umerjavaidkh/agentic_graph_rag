"""Data models for the audit log."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class AuditEventType(str, Enum):
    """Category of a recorded audit event."""

    QUERY = "query"
    ACCESS_DENIED = "access_denied"
    INGESTION_SUBMITTED = "ingestion_submitted"
    INGESTION_COMPLETED = "ingestion_completed"
    INGESTION_FAILED = "ingestion_failed"


@dataclass
class AuditEvent:
    """Who did what, when, to what data, with what result."""

    event_id: str
    ts: str
    event_type: str
    user_id: str
    tenant_id: str
    role: str
    request_id: Optional[str] = None
    resource: Optional[str] = None
    action: Optional[str] = None
    result: str = "success"
    reason: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
