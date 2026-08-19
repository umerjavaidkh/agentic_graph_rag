"""Process-wide singletons the routers and handlers share.

The two thread pools especially MUST be created once. Each is sized from
config, so a per-router copy would silently multiply the thread count by the
number of routers -- a behaviour change no test would notice.

Imports are kept to what this module actually uses. Carrying api.py's whole
import block, as an earlier version did, pulled in `from .bridge import ask`,
which imports the router, which imports this -- a cycle that surfaced as
"cannot import name 'ask' from partially initialized module".
"""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from fastapi import BackgroundTasks

from ..shared.config.settings import API_INGEST_EXECUTOR_WORKERS, API_QUERY_EXECUTOR_WORKERS
from ..pipeline.ingestion.queue import enqueue_ingest
from ..unstructured.ingestion.service import IngestionManager

# Shared ingestion manager (store-backed — works in both in-process and worker modes).
ingestion_manager = IngestionManager()

# Fallback executor: used only when REDIS_URL is not set (dev / single-process mode).
_ingest_executor = ThreadPoolExecutor(max_workers=API_INGEST_EXECUTOR_WORKERS, thread_name_prefix="ingest")
# Run sync RAG pipeline (LLM + Neo4j) off the asyncio event loop.
_query_executor = ThreadPoolExecutor(max_workers=API_QUERY_EXECUTOR_WORKERS, thread_name_prefix="query")



async def _run_ingest_job_local(job_id: str) -> None:
    """In-process fallback: run the job in a thread when Redis is not configured."""
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_ingest_executor, ingestion_manager.run_job, job_id)


def _dispatch_ingest_job(
    job_id: str, background_tasks: BackgroundTasks, *, job_timeout: str = "30m"
) -> str:
    """
    Dispatch a job to RQ workers when Redis is configured, or run it
    locally via BackgroundTasks when it is not.  Returns the dispatch mode.

    job_timeout only affects the RQ path — the BackgroundTasks/ThreadPoolExecutor
    fallback has no timeout concept.
    """
    rq_job = enqueue_ingest(job_id, job_timeout=job_timeout)  # None when REDIS_URL not set
    if rq_job is not None:
        return "worker"
    background_tasks.add_task(_run_ingest_job_local, job_id)
    return "background_task"
