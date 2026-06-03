"""
tasks.py — RQ-callable task for ingestion workers.

The only public symbol is ``run_ingest_job``, which is the function RQ
workers import and execute.  It:
  1. Loads the IngestionJob from the durable JobStore.
  2. Delegates to IngestionManager.run_job() (all parsing / LLM / Neo4j logic
     lives there, unchanged).
  3. The per-logical-doc Redis lock is acquired inside IngestionManager around
     the Neo4j revision-install step so different documents run in parallel but
     the same logical document serialises its writes.

This module must be importable by RQ workers without pulling in FastAPI or any
request-scoped state.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)


def run_ingest_job(job_id: str) -> None:
    """
    Top-level RQ task callable.

    RQ workers call this function with the job_id.  The function is
    intentionally thin so the real processing logic stays in IngestionManager
    and can be unit-tested without RQ.
    """
    from .job_store import get_job_store
    from .service import IngestionManager

    store = get_job_store()
    job = store.get(job_id)
    if job is None:
        raise ValueError(f"run_ingest_job: job '{job_id}' not found in store")

    log.info("Worker picked up job %s (type=%s name=%s)", job_id, job.type, job.name)

    manager = IngestionManager(store=store)
    manager.run_job(job_id)

    log.info("Worker finished job %s (status=%s)", job_id, manager.get_job(job_id).status)
