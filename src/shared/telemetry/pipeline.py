"""Pipeline step tracing for RAG requests (logs + telemetry payload)."""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any, Iterator

from .context import get_telemetry

logger = logging.getLogger(__name__)


@contextmanager
def pipeline_step(name: str, **meta: Any) -> Iterator[None]:
    """
    Record a named pipeline step with duration and optional metadata.

    On success: status=ok. On exception: status=error, logs stack trace, re-raises.
    """
    tel = get_telemetry()
    t0 = time.perf_counter()
    try:
        yield
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if tel is not None:
            tel.record_step(name, status="ok", duration_ms=duration_ms, meta=meta)
        if meta:
            logger.info("pipeline ok: %s (%dms) %s", name, duration_ms, meta)
        else:
            logger.info("pipeline ok: %s (%dms)", name, duration_ms)
    except Exception as exc:
        duration_ms = int((time.perf_counter() - t0) * 1000)
        if tel is not None:
            tel.record_step(
                name,
                status="error",
                duration_ms=duration_ms,
                meta=meta,
                error=str(exc),
            )
            tel.record_error(name, exc)
        logger.warning(
            "pipeline failed: %s (%dms) — %s",
            name,
            duration_ms,
            exc,
            exc_info=True,
        )
        raise


def record_pipeline_step(
    name: str,
    *,
    status: str = "ok",
    duration_ms: int = 0,
    meta: dict[str, Any] | None = None,
    error: str | None = None,
) -> None:
    """Record a step without a context manager (e.g. after routing completes)."""
    tel = get_telemetry()
    if tel is not None:
        tel.record_step(
            name,
            status=status,
            duration_ms=duration_ms,
            meta=meta or {},
            error=error,
        )
    if status == "error":
        logger.warning("pipeline step error: %s — %s", name, error or status)
    elif meta:
        logger.info("pipeline step: %s %s", name, meta)
