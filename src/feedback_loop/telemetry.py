"""Telemetry helpers for the feedback loop."""
from __future__ import annotations

import logging

from .hints import ModeHint
from ..telemetry import pipeline_step

logger = logging.getLogger(__name__)


def record_routing_applied(*, agent: str, hint: ModeHint, action: str) -> None:
    with pipeline_step(
        "feedback.routing",
        agent=agent,
        mode=hint.mode,
        pass_rate=round(hint.pass_rate, 3),
        samples=hint.samples,
        action=action,
    ):
        logger.info(
            "feedback routing agent=%s mode=%s action=%s samples=%s pass_rate=%.2f",
            agent,
            hint.mode,
            action,
            hint.samples,
            hint.pass_rate,
        )
