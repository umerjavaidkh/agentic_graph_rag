"""Data models for the retrieval feedback loop."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class AggregateDimension(str, Enum):
    """Which event field to aggregate pass/fail statistics by."""

    RETRIEVAL_MODE = "retrieval_mode"
    ROUTE_TOOL = "route_tool"


@dataclass
class RetrievalFeedbackEvent:
    """Compact feedback row derived from pipeline telemetry."""

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

    def aggregate_key(self, dimension: AggregateDimension) -> str:
        if dimension is AggregateDimension.ROUTE_TOOL:
            return self.route_tool or "unknown"
        return self.retrieval_mode or "unknown"
