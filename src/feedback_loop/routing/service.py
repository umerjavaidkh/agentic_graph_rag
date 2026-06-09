"""Feedback routing service — single entry point for retrieval consumers."""
from __future__ import annotations

import logging
import threading
from typing import Optional

from ..config import FeedbackConfig
from ..hints import ModeHint, hint_for_question, invalidate_hint_cache
from ..models import AggregateDimension
from ..telemetry import record_routing_applied
from .base import PolicyDecision, RoutingPolicy
from .policies import DocumentModePolicy, RouteToolPolicy, StructuredPathPolicy

logger = logging.getLogger(__name__)

_service_singleton: Optional["FeedbackRoutingService"] = None
_service_lock = threading.Lock()


class FeedbackRoutingService:
    """Applies labeled feedback hints to retrieval and routing decisions."""

    def __init__(
        self,
        config: Optional[FeedbackConfig] = None,
        policies: Optional[list[RoutingPolicy]] = None,
    ) -> None:
        self._config = config or FeedbackConfig.load()
        self._policies = policies or [
            StructuredPathPolicy(),
            DocumentModePolicy(),
            RouteToolPolicy(),
        ]
        self._policy_by_name = {p.name: p for p in self._policies}

    @property
    def enabled(self) -> bool:
        return self._config.routing_enabled

    def _lookup_hint_for_question(
        self,
        question: str,
        policy: RoutingPolicy,
    ) -> Optional[ModeHint]:
        if not self.enabled:
            return None
        dimension = getattr(policy, "dimension", AggregateDimension.RETRIEVAL_MODE)
        return hint_for_question(
            question,
            agent=policy.hint_agent,
            dimension=dimension,
            config=self._config,
        )

    def _apply_policy(
        self,
        question: str,
        policy_name: str,
    ) -> Optional[PolicyDecision]:
        policy = self._policy_by_name.get(policy_name)
        if policy is None:
            return None
        hint = self._lookup_hint_for_question(question, policy)
        if hint is None:
            return None
        decision = policy.decide(hint)
        if decision is None:
            return None
        record_routing_applied(
            agent=policy.hint_agent or "route",
            hint=hint,
            action=decision.action,
        )
        return decision

    def structured_path(self, question: str) -> Optional[str]:
        decision = self._apply_policy(question, StructuredPathPolicy.name)
        return decision.value if decision else None

    def document_mode(self, question: str) -> Optional[str]:
        decision = self._apply_policy(question, DocumentModePolicy.name)
        return decision.value if decision else None

    def route_tool(self, question: str, baseline: str) -> str:
        if not self.enabled or baseline not in {"search_documents", "query_data", "query_hybrid"}:
            return baseline
        decision = self._apply_policy(question, RouteToolPolicy.name)
        if decision is None or decision.value == baseline:
            return baseline
        logger.info(
            "feedback route override baseline=%s chosen=%s",
            baseline,
            decision.value,
        )
        return decision.value


def get_feedback_routing() -> FeedbackRoutingService:
    global _service_singleton
    if _service_singleton is not None:
        return _service_singleton
    with _service_lock:
        if _service_singleton is None:
            _service_singleton = FeedbackRoutingService()
        return _service_singleton


def reset_feedback_routing(service: Optional[FeedbackRoutingService] = None) -> None:
    """Test hook: replace the process singleton."""
    global _service_singleton
    with _service_lock:
        _service_singleton = service
    if service is None:
        invalidate_hint_cache()


def structured_path_hint(question: str) -> Optional[str]:
    return get_feedback_routing().structured_path(question)


def document_mode_hint(question: str) -> Optional[str]:
    return get_feedback_routing().document_mode(question)


def retrieval_hint(question: str, *, agent: str) -> Optional[ModeHint]:
    if not get_feedback_routing().enabled:
        return None
    from ..models import AggregateDimension

    return hint_for_question(question, agent=agent, dimension=AggregateDimension.RETRIEVAL_MODE)


def record_hint_applied(*, agent: str, hint: ModeHint, action: str) -> None:
    record_routing_applied(agent=agent, hint=hint, action=action)
