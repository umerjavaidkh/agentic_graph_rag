"""
queue.py — RQ queue wiring for the ingestion pipeline.

Provides:
  get_ingest_queue()   — returns the RQ Queue (or None when Redis is not configured).
  enqueue_ingest()     — enqueue a job_id for processing by the `run_ingest_job` task,
                         with auto-retry and TTL.
  list_failed_jobs()   — introspect the FailedJobRegistry (dead-letter visibility).

When REDIS_URL is not set, all functions degrade gracefully and return None/[],
so callers can fall back to the in-process BackgroundTasks path.
"""
from __future__ import annotations

import logging
from typing import Any, List, Optional

log = logging.getLogger(__name__)

# ── Lazy queue singleton ─────────────────────────────────────────────────────

_queue: Optional[Any] = None  # rq.Queue | None


def get_ingest_queue() -> Optional[Any]:
    """
    Return the RQ Queue singleton for the ingest queue.

    Returns None if REDIS_URL is not configured or rq/redis is not installed.
    Thread-safe: multiple callers get the same instance.
    """
    global _queue
    if _queue is not None:
        return _queue

    from ..config.settings import INGEST_QUEUE_NAME, REDIS_URL

    if not REDIS_URL:
        return None

    try:
        import redis as _redis
        import rq as _rq

        conn = _redis.from_url(REDIS_URL, decode_responses=False)
        conn.ping()
        _queue = _rq.Queue(
            INGEST_QUEUE_NAME,
            connection=conn,
            # Keep finished job results for 1 h; failed jobs go to FailedJobRegistry.
            result_ttl=3600,
            failure_ttl=7 * 24 * 3600,
        )
        log.info("RQ queue '%s' connected to %s", INGEST_QUEUE_NAME, REDIS_URL)
        return _queue
    except Exception as exc:
        log.warning("Cannot connect RQ queue: %s — falling back to BackgroundTasks", exc)
        return None


# ── Enqueue ──────────────────────────────────────────────────────────────────

def enqueue_ingest(job_id: str) -> Optional[Any]:
    """
    Push job_id onto the ingest queue for `run_ingest_job` to pick up.

    Retries:  up to 2 retries with 30 s / 2 min back-off.
    Job TTL:  enqueued jobs expire after 24 h if never dequeued.

    Returns the rq.job.Job object on success, None when running in-process mode.
    """
    queue = get_ingest_queue()
    if queue is None:
        return None

    try:
        from rq import Retry
        from .tasks import run_ingest_job

        rq_job = queue.enqueue(
            run_ingest_job,
            job_id,
            job_id=f"ingest:{job_id}",
            job_timeout="30m",
            ttl=24 * 3600,
            retry=Retry(max=2, interval=[30, 120]),
        )
        log.info("Enqueued job %s → RQ job %s", job_id, rq_job.id)
        return rq_job
    except Exception as exc:
        log.error("Failed to enqueue job %s: %s", job_id, exc)
        return None


# ── Dead-letter / failed job introspection ───────────────────────────────────

def list_failed_jobs(limit: int = 50) -> List[dict]:
    """
    Return a summary of jobs in the FailedJobRegistry (dead-letter queue).

    Each entry has: id, description, enqueued_at, ended_at, exc_info (truncated).
    Returns [] when Redis is not configured or no failed jobs exist.
    """
    queue = get_ingest_queue()
    if queue is None:
        return []

    try:
        from rq.job import Job
        from rq.registry import FailedJobRegistry

        registry = FailedJobRegistry(queue=queue)
        job_ids = registry.get_job_ids(0, limit - 1)
        results = []
        for jid in job_ids:
            try:
                job = Job.fetch(jid, connection=queue.connection)
                exc = (job.exc_info or "")[-500:]  # cap at 500 chars
                results.append({
                    "rq_job_id": jid,
                    "description": job.description,
                    "enqueued_at": job.enqueued_at.isoformat() if job.enqueued_at else None,
                    "ended_at": job.ended_at.isoformat() if job.ended_at else None,
                    "exc_info": exc,
                })
            except Exception:
                results.append({"rq_job_id": jid})
        return results
    except Exception as exc:
        log.warning("list_failed_jobs: %s", exc)
        return []


def queue_depth() -> Optional[int]:
    """Return the number of jobs waiting in the ingest queue, or None."""
    queue = get_ingest_queue()
    if queue is None:
        return None
    try:
        return len(queue)
    except Exception:
        return None
