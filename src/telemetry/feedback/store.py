"""Durable feedback storage: Redis (production) or daily JSONL (dev)."""
from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .models import RetrievalFeedbackEvent

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


class FeedbackStore(ABC):
    @abstractmethod
    def record(self, event: RetrievalFeedbackEvent) -> None:
        """Persist one feedback event (append-only)."""

    @abstractmethod
    def attach_outcome(
        self,
        request_id: str,
        *,
        passed: bool,
        case_id: Optional[str] = None,
    ) -> bool:
        """Label a prior event by request_id; return False if not found."""

    @abstractmethod
    def aggregate_stats(self, pattern_hash: str) -> dict[str, dict[str, int]]:
        """Return {mode: {pass, fail, queries}} for a pattern hash."""


class JsonlFeedbackStore(FeedbackStore):
    """
    Dev/single-node backend: one JSONL file per UTC day.

    Writes are serialized with a process lock; reads for aggregates scan recent files.
    """

    def __init__(self, directory: Path, *, retain_days: int = 30) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._retain_days = max(1, retain_days)
        self._req_index: dict[str, RetrievalFeedbackEvent] = {}

    def _path_for_ts(self, ts: str) -> Path:
        day = (ts or _utc_now_iso())[:10]
        return self._dir / f"{day}.jsonl"

    def record(self, event: RetrievalFeedbackEvent) -> None:
        line = json.dumps(event.to_dict(), separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            path = self._path_for_ts(event.ts)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._req_index[event.request_id] = event

    def attach_outcome(
        self,
        request_id: str,
        *,
        passed: bool,
        case_id: Optional[str] = None,
    ) -> bool:
        with self._lock:
            base = self._req_index.get(request_id)
            if base is None:
                base = self._load_request_from_disk(request_id)
            if base is None:
                return False
            updated = RetrievalFeedbackEvent(
                **{
                    **base.to_dict(),
                    "outcome": passed,
                    "case_id": case_id or base.case_id,
                    "source": base.source if base.source != "query" else "outcome_attach",
                }
            )
            line = json.dumps(updated.to_dict(), separators=(",", ":"), ensure_ascii=False)
            path = self._path_for_ts(updated.ts)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
            self._req_index[request_id] = updated
            return True

    def _load_request_from_disk(self, request_id: str) -> Optional[RetrievalFeedbackEvent]:
        files = sorted(self._dir.glob("*.jsonl"), reverse=True)
        for path in files[: self._retain_days]:
            try:
                for line in reversed(path.read_text(encoding="utf-8").splitlines()):
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("request_id") == request_id:
                        return RetrievalFeedbackEvent(**data)
            except (OSError, json.JSONDecodeError):
                continue
        return None

    def aggregate_stats(self, pattern_hash: str) -> dict[str, dict[str, int]]:
        stats: dict[str, dict[str, int]] = {}
        files = sorted(self._dir.glob("*.jsonl"), reverse=True)
        for path in files[: self._retain_days]:
            try:
                for line in path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    if data.get("pattern_hash") != pattern_hash:
                        continue
                    mode = data.get("retrieval_mode") or "unknown"
                    bucket = stats.setdefault(mode, {"pass": 0, "fail": 0, "queries": 0})
                    bucket["queries"] += 1
                    outcome = data.get("outcome")
                    if outcome is True:
                        bucket["pass"] += 1
                    elif outcome is False:
                        bucket["fail"] += 1
            except (OSError, json.JSONDecodeError):
                continue
        return stats


class RedisFeedbackStore(FeedbackStore):
    """
    Production backend:
      - Redis Stream for raw events (XADD, bounded MAXLEN)
      - Hash per pattern for O(1) aggregate reads
      - Request index key for late outcome attachment (TTL)
    """

    _REQ_PREFIX = "rag:feedback:req:"
    _AGG_PREFIX = "rag:feedback:agg:"

    def __init__(
        self,
        redis_client: Any,
        *,
        stream_key: str,
        stream_maxlen: int,
        req_ttl_sec: int,
        agg_ttl_sec: int,
    ) -> None:
        self._r = redis_client
        self._stream = stream_key
        self._stream_maxlen = max(1000, stream_maxlen)
        self._req_ttl = max(3600, req_ttl_sec)
        self._agg_ttl = max(86400, agg_ttl_sec)

    def record(self, event: RetrievalFeedbackEvent) -> None:
        payload = event.to_dict()
        compact_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        mode = event.retrieval_mode or "unknown"
        agg_key = f"{self._AGG_PREFIX}{event.pattern_hash}"

        pipe = self._r.pipeline()
        pipe.xadd(
            self._stream,
            {"json": compact_json},
            maxlen=self._stream_maxlen,
            approximate=True,
        )
        pipe.set(
            f"{self._REQ_PREFIX}{event.request_id}",
            compact_json,
            ex=self._req_ttl,
        )
        pipe.hincrby(agg_key, f"{mode}:queries", 1)
        if event.outcome is True:
            pipe.hincrby(agg_key, f"{mode}:pass", 1)
        elif event.outcome is False:
            pipe.hincrby(agg_key, f"{mode}:fail", 1)
        pipe.expire(agg_key, self._agg_ttl)
        pipe.execute()

    def attach_outcome(
        self,
        request_id: str,
        *,
        passed: bool,
        case_id: Optional[str] = None,
    ) -> bool:
        req_key = f"{self._REQ_PREFIX}{request_id}"
        raw = self._r.get(req_key)
        if raw is None:
            return False
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        data = json.loads(raw)
        mode = data.get("retrieval_mode") or "unknown"
        pattern_hash = data.get("pattern_hash") or ""
        data["outcome"] = passed
        if case_id:
            data["case_id"] = case_id
        updated = json.dumps(data, separators=(",", ":"), ensure_ascii=False)

        agg_key = f"{self._AGG_PREFIX}{pattern_hash}"
        field = f"{mode}:{'pass' if passed else 'fail'}"
        pipe = self._r.pipeline()
        pipe.set(req_key, updated, ex=self._req_ttl)
        pipe.hincrby(agg_key, field, 1)
        pipe.xadd(
            self._stream,
            {"json": updated, "kind": "outcome_attach"},
            maxlen=self._stream_maxlen,
            approximate=True,
        )
        pipe.expire(agg_key, self._agg_ttl)
        pipe.execute()
        return True

    def aggregate_stats(self, pattern_hash: str) -> dict[str, dict[str, int]]:
        agg_key = f"{self._AGG_PREFIX}{pattern_hash}"
        raw = self._r.hgetall(agg_key)
        if not raw:
            return {}

        flat: dict[str, str] = {}
        for k, v in raw.items():
            key = k.decode() if isinstance(k, bytes) else str(k)
            val = v.decode() if isinstance(v, bytes) else str(v)
            flat[key] = val

        stats: dict[str, dict[str, int]] = {}
        for compound, count_str in flat.items():
            if ":" not in compound:
                continue
            mode, metric = compound.rsplit(":", 1)
            bucket = stats.setdefault(mode, {"pass": 0, "fail": 0, "queries": 0})
            try:
                bucket[metric] = int(count_str)
            except ValueError:
                continue
        return stats


_store_singleton: Optional[FeedbackStore] = None
_store_lock = threading.Lock()


def get_feedback_store() -> FeedbackStore:
    """Process singleton: Redis when configured, else daily JSONL under data/feedback."""
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton

    with _store_lock:
        if _store_singleton is not None:
            return _store_singleton

        from ...config.settings import (
            REDIS_URL,
            RETRIEVAL_FEEDBACK_AGG_TTL_DAYS,
            RETRIEVAL_FEEDBACK_DIR,
            RETRIEVAL_FEEDBACK_JSONL_RETAIN_DAYS,
            RETRIEVAL_FEEDBACK_REDIS_STREAM,
            RETRIEVAL_FEEDBACK_REQ_TTL_SEC,
            RETRIEVAL_FEEDBACK_STREAM_MAXLEN,
        )

        if REDIS_URL:
            try:
                import redis as _redis

                client = _redis.from_url(REDIS_URL, decode_responses=False)
                client.ping()
                _store_singleton = RedisFeedbackStore(
                    client,
                    stream_key=RETRIEVAL_FEEDBACK_REDIS_STREAM,
                    stream_maxlen=RETRIEVAL_FEEDBACK_STREAM_MAXLEN,
                    req_ttl_sec=RETRIEVAL_FEEDBACK_REQ_TTL_SEC,
                    agg_ttl_sec=RETRIEVAL_FEEDBACK_AGG_TTL_DAYS * 86400,
                )
                logger.info("Retrieval feedback store: Redis stream %s", RETRIEVAL_FEEDBACK_REDIS_STREAM)
                return _store_singleton
            except Exception as exc:
                logger.warning(
                    "Retrieval feedback: Redis unavailable (%s); using JSONL at %s",
                    exc,
                    RETRIEVAL_FEEDBACK_DIR,
                )

        _store_singleton = JsonlFeedbackStore(
            Path(RETRIEVAL_FEEDBACK_DIR),
            retain_days=RETRIEVAL_FEEDBACK_JSONL_RETAIN_DAYS,
        )
        logger.info("Retrieval feedback store: JSONL %s", RETRIEVAL_FEEDBACK_DIR)
        return _store_singleton


def reset_feedback_store(store: Optional[FeedbackStore] = None) -> None:
    """Test hook: replace the process singleton."""
    global _store_singleton
    with _store_lock:
        _store_singleton = store
