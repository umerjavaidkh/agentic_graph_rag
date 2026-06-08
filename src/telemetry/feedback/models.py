"""Data models for retrieval feedback (observe-only by default)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class RetrievalFeedbackEvent:
    """Compact feedback row derived from pipeline telemetry (not raw retrieval)."""

    request_id: str
    ts: str
    question_hash: str
    agent: str
    strategy: str
    route_tool: str
    route_method: str
    pattern: str
    pattern_hash: str
    retrieval_mode: str
    retrieval_profile: dict[str, Any] = field(default_factory=dict)
    pipeline: list[dict[str, Any]] = field(default_factory=list)
    outcome: Optional[bool] = None
    case_id: Optional[str] = None
    source: str = "query"
    question_preview: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
