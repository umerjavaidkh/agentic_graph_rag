"""
job_store.py — Durable job state for the ingestion pipeline.

Two backends:
  RedisJobStore   — production; survives restarts, shareable across workers.
  InMemoryJobStore — dev fallback; identical interface, no dependencies.

The store serialises the full IngestionJob (status, logs, versioning metadata)
so any worker process can read or update it without sharing in-process state.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import IngestionStatus

# Import IngestionJob lazily to avoid circular imports.
def _job_class():
    from .service import IngestionJob
    return IngestionJob


# ── Serialisation helpers ────────────────────────────────────────────────────

_DATETIME_FMT = "%Y-%m-%dT%H:%M:%S.%f"


def _ser_dt(dt: Optional[datetime]) -> Optional[str]:
    return dt.strftime(_DATETIME_FMT) if dt else None


def _de_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        return datetime.strptime(s, _DATETIME_FMT)
    except ValueError:
        return None


def job_to_dict(job: Any) -> Dict[str, Any]:
    """Serialise an IngestionJob to a plain dict (JSON-safe)."""
    return {
        "id": job.id,
        "type": job.type,
        "name": job.name,
        "doc_key": job.doc_key,
        "status": job.status.value if hasattr(job.status, "value") else str(job.status),
        "created_at": _ser_dt(job.created_at),
        "started_at": _ser_dt(job.started_at),
        "finished_at": _ser_dt(job.finished_at),
        "input_path": str(job.input_path) if job.input_path else None,
        "output_dir": str(job.output_dir) if job.output_dir else None,
        "cypher_params": job.cypher_params,
        "neo4j_load_status": job.neo4j_load_status,
        "neo4j_load_message": job.neo4j_load_message,
        "error": job.error,
        "logical_doc_id": job.logical_doc_id,
        "revision_id": job.revision_id,
        "content_hash": job.content_hash,
        "version_number": job.version_number,
        "skipped_duplicate": job.skipped_duplicate,
    }


def job_from_dict(data: Dict[str, Any]) -> Any:
    """Deserialise a plain dict back to an IngestionJob."""
    IngestionJob = _job_class()
    job = IngestionJob(
        id=data["id"],
        type=data["type"],
        name=data.get("name"),
        doc_key=data.get("doc_key"),
        status=IngestionStatus(data.get("status", "queued")),
        created_at=_de_dt(data.get("created_at")) or datetime.utcnow(),
        started_at=_de_dt(data.get("started_at")),
        finished_at=_de_dt(data.get("finished_at")),
        input_path=Path(data["input_path"]) if data.get("input_path") else None,
        output_dir=Path(data["output_dir"]) if data.get("output_dir") else None,
        cypher_params=data.get("cypher_params"),
        neo4j_load_status=data.get("neo4j_load_status"),
        neo4j_load_message=data.get("neo4j_load_message"),
        error=data.get("error"),
        logical_doc_id=data.get("logical_doc_id"),
        revision_id=data.get("revision_id"),
        content_hash=data.get("content_hash"),
        version_number=data.get("version_number"),
        skipped_duplicate=bool(data.get("skipped_duplicate", False)),
    )
    return job


# ── Interface ────────────────────────────────────────────────────────────────

class JobStore(ABC):
    @abstractmethod
    def save(self, job: Any) -> None: ...

    @abstractmethod
    def get(self, job_id: str) -> Optional[Any]: ...

    @abstractmethod
    def append_log(self, job_id: str, line: str) -> None: ...

    @abstractmethod
    def get_logs(self, job_id: str) -> List[str]: ...

    @abstractmethod
    def list_ids(self, limit: int = 100) -> List[str]: ...

    @abstractmethod
    def delete(self, job_id: str) -> None: ...


# ── In-memory fallback (dev, no Redis) ───────────────────────────────────────

class InMemoryJobStore(JobStore):
    """Thread-safe for single-process use (API + BackgroundTasks path)."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._logs: Dict[str, List[str]] = {}
        self._order: List[str] = []

    def save(self, job: Any) -> None:
        data = job_to_dict(job)
        self._jobs[job.id] = data
        if job.id not in self._logs:
            self._logs[job.id] = []
        if job.id not in self._order:
            self._order.append(job.id)

    def get(self, job_id: str) -> Optional[Any]:
        data = self._jobs.get(job_id)
        if data is None:
            return None
        job = job_from_dict(data)
        job.logs = list(self._logs.get(job_id, []))
        return job

    def append_log(self, job_id: str, line: str) -> None:
        if job_id not in self._logs:
            self._logs[job_id] = []
        self._logs[job_id].append(line)

    def get_logs(self, job_id: str) -> List[str]:
        return list(self._logs.get(job_id, []))

    def list_ids(self, limit: int = 100) -> List[str]:
        return self._order[-limit:]

    def delete(self, job_id: str) -> None:
        self._jobs.pop(job_id, None)
        self._logs.pop(job_id, None)
        try:
            self._order.remove(job_id)
        except ValueError:
            pass


# ── Redis-backed store (production) ──────────────────────────────────────────

class RedisJobStore(JobStore):
    """
    Stores each job as a Redis hash  key=``job:{id}``
    and logs as a Redis list         key=``job:{id}:logs``.

    TTL (default 7 days) prevents unbounded growth.
    """

    _JOB_PREFIX = "job:"
    _LOG_SUFFIX = ":logs"
    _INDEX_KEY  = "jobs:index"
    _DEFAULT_TTL = 7 * 24 * 3600  # 7 days

    def __init__(self, redis_client: Any, ttl: int = _DEFAULT_TTL) -> None:
        self._r = redis_client
        self._ttl = ttl

    # -- JobStore interface ---------------------------------------------------

    def save(self, job: Any) -> None:
        key = self._key(job.id)
        data = json.dumps(job_to_dict(job))
        pipe = self._r.pipeline()
        pipe.set(key, data, ex=self._ttl)
        pipe.expire(self._log_key(job.id), self._ttl)
        pipe.lpush(self._INDEX_KEY, job.id)
        pipe.ltrim(self._INDEX_KEY, 0, 9999)
        pipe.expire(self._INDEX_KEY, self._ttl)
        pipe.execute()

    def get(self, job_id: str) -> Optional[Any]:
        raw = self._r.get(self._key(job_id))
        if raw is None:
            return None
        data = json.loads(raw)
        job = job_from_dict(data)
        job.logs = self.get_logs(job_id)
        return job

    def append_log(self, job_id: str, line: str) -> None:
        log_key = self._log_key(job_id)
        pipe = self._r.pipeline()
        pipe.rpush(log_key, line)
        pipe.expire(log_key, self._ttl)
        pipe.execute()

    def get_logs(self, job_id: str) -> List[str]:
        items = self._r.lrange(self._log_key(job_id), 0, -1)
        return [item.decode() if isinstance(item, bytes) else item for item in items]

    def list_ids(self, limit: int = 100) -> List[str]:
        raw = self._r.lrange(self._INDEX_KEY, 0, limit - 1)
        seen: set = set()
        result: List[str] = []
        for item in raw:
            jid = item.decode() if isinstance(item, bytes) else item
            if jid not in seen:
                seen.add(jid)
                result.append(jid)
        return result

    def delete(self, job_id: str) -> None:
        pipe = self._r.pipeline()
        pipe.delete(self._key(job_id))
        pipe.delete(self._log_key(job_id))
        pipe.execute()

    # -- Helpers -------------------------------------------------------------

    def _key(self, job_id: str) -> str:
        return f"{self._JOB_PREFIX}{job_id}"

    def _log_key(self, job_id: str) -> str:
        return f"{self._JOB_PREFIX}{job_id}{self._LOG_SUFFIX}"


# ── Factory ──────────────────────────────────────────────────────────────────

_store_singleton: Optional[JobStore] = None


def get_job_store() -> JobStore:
    """
    Return the process-level JobStore singleton.

    Uses RedisJobStore when REDIS_URL is set; otherwise InMemoryJobStore.
    Thread-safe for typical FastAPI/gunicorn usage (called at import time or on
    first request — whichever comes first).
    """
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton

    from ..config.settings import REDIS_URL
    if REDIS_URL:
        try:
            import redis as _redis
            client = _redis.from_url(REDIS_URL, decode_responses=False)
            client.ping()
            _store_singleton = RedisJobStore(client)
            return _store_singleton
        except Exception as exc:
            import warnings
            warnings.warn(
                f"RedisJobStore: could not connect to {REDIS_URL!r} ({exc}). "
                "Falling back to InMemoryJobStore.",
                RuntimeWarning,
                stacklevel=2,
            )

    _store_singleton = InMemoryJobStore()
    return _store_singleton
