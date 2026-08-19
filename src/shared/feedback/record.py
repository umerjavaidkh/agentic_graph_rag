"""Non-blocking feedback recorder."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Optional

from .config import FeedbackConfig
from .profile import build_feedback_event
from .store import get_feedback_store

logger = logging.getLogger(__name__)

_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="feedback")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _write_event(event_kwargs: dict[str, Any]) -> None:
    try:
        event = build_feedback_event(**event_kwargs)
        get_feedback_store().record(event)
    except Exception:
        logger.debug("feedback loop write failed", exc_info=True)


def maybe_record_retrieval_feedback(
    *,
    request_id: str,
    question: str,
    result: dict[str, Any],
    outcome: Optional[bool] = None,
    case_id: Optional[str] = None,
    source: str = "query",
) -> None:
    """Fire-and-forget persistence of pipeline telemetry. No-op when disabled."""
    config = FeedbackConfig.load()
    if not config.enabled:
        return

    telemetry = result.get("_telemetry") or {}
    event_kwargs = {
        "request_id": request_id,
        "ts": _utc_now_iso(),
        "question": question,
        "telemetry": telemetry,
        "agent": str(result.get("agent") or ""),
        "strategy": str(result.get("strategy") or ""),
        "route_tool": str(result.get("_route_tool") or ""),
        "route_method": str(result.get("_route_method") or ""),
        "outcome": outcome,
        "case_id": case_id,
        "source": source,
        "store_question_preview": config.store_question,
    }
    _executor.submit(_write_event, event_kwargs)


def maybe_attach_feedback_outcome(
    request_id: str,
    *,
    passed: bool,
    case_id: Optional[str] = None,
) -> None:
    """Label a prior request (eval runner, user thumbs). Non-blocking."""
    if not FeedbackConfig.load().enabled:
        return

    def _attach() -> None:
        try:
            ok = get_feedback_store().attach_outcome(
                request_id,
                passed=passed,
                case_id=case_id,
            )
            if not ok:
                logger.debug("feedback outcome attach miss request_id=%s", request_id)
        except Exception:
            logger.debug("feedback outcome attach failed", exc_info=True)

    _executor.submit(_attach)
