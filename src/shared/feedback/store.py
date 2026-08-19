"""Durable feedback storage: Redis (production) or daily JSONL (dev)."""
from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import FeedbackConfig
from .models import AggregateDimension, RetrievalFeedbackEvent

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
    def aggregate_stats(
        self,
        pattern_hash: str,
        *,
        dimension: AggregateDimension = AggregateDimension.RETRIEVAL_MODE,
    ) -> dict[str, dict[str, int]]:
        """Return {bucket: {pass, fail, queries}} for a pattern hash."""

    @abstractmethod
    def recent_events(self, *, limit: int = 50) -> list[RetrievalFeedbackEvent]:
        """Most recent unique events (latest row per request_id)."""


def _merge_outcome_bucket(
    stats: dict[str, dict[str, int]],
    bucket: str,
    outcome: Optional[bool],
) -> None:
    row = stats.setdefault(bucket, {"pass": 0, "fail": 0, "queries": 0})
    row["queries"] += 1
    if outcome is True:
        row["pass"] += 1
    elif outcome is False:
        row["fail"] += 1


class JsonlFeedbackStore(FeedbackStore):
    """Dev/single-node backend: one JSONL file per UTC day."""

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

    def aggregate_stats(
        self,
        pattern_hash: str,
        *,
        dimension: AggregateDimension = AggregateDimension.RETRIEVAL_MODE,
    ) -> dict[str, dict[str, int]]:
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
                    event = RetrievalFeedbackEvent(**data)
                    bucket = event.aggregate_key(dimension)
                    _merge_outcome_bucket(stats, bucket, event.outcome)
            except (OSError, json.JSONDecodeError):
                continue
        return stats

    def recent_events(self, *, limit: int = 50) -> list[RetrievalFeedbackEvent]:
        by_id: dict[str, RetrievalFeedbackEvent] = {}
        files = sorted(self._dir.glob("*.jsonl"), reverse=True)
        for path in files[: self._retain_days]:
            try:
                for line in reversed(path.read_text(encoding="utf-8").splitlines()):
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    event = RetrievalFeedbackEvent(**data)
                    by_id[event.request_id] = event
                    if len(by_id) >= limit * 3:
                        break
            except (OSError, json.JSONDecodeError):
                continue
            if len(by_id) >= limit * 3:
                break
        ranked = sorted(by_id.values(), key=lambda e: e.ts, reverse=True)
        return ranked[:limit]


class RedisFeedbackStore(FeedbackStore):
    """Production backend: Redis Stream + per-dimension aggregate hashes."""

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

    def _agg_key(self, pattern_hash: str, dimension: AggregateDimension) -> str:
        if dimension is AggregateDimension.ROUTE_TOOL:
            return f"{self._AGG_PREFIX}route:{pattern_hash}"
        return f"{self._AGG_PREFIX}{pattern_hash}"

    def _incr_bucket(
        self,
        pipe: Any,
        agg_key: str,
        bucket: str,
        outcome: Optional[bool],
    ) -> None:
        pipe.hincrby(agg_key, f"{bucket}:queries", 1)
        if outcome is True:
            pipe.hincrby(agg_key, f"{bucket}:pass", 1)
        elif outcome is False:
            pipe.hincrby(agg_key, f"{bucket}:fail", 1)

    def record(self, event: RetrievalFeedbackEvent) -> None:
        payload = event.to_dict()
        compact_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

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
        for dimension in AggregateDimension:
            agg_key = self._agg_key(event.pattern_hash, dimension)
            self._incr_bucket(pipe, agg_key, event.aggregate_key(dimension), event.outcome)
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
        event = RetrievalFeedbackEvent(**data)
        data["outcome"] = passed
        if case_id:
            data["case_id"] = case_id
        updated = json.dumps(data, separators=(",", ":"), ensure_ascii=False)
        field = "pass" if passed else "fail"

        pipe = self._r.pipeline()
        pipe.set(req_key, updated, ex=self._req_ttl)
        for dimension in AggregateDimension:
            agg_key = self._agg_key(event.pattern_hash, dimension)
            bucket = event.aggregate_key(dimension)
            pipe.hincrby(agg_key, f"{bucket}:{field}", 1)
            pipe.expire(agg_key, self._agg_ttl)
        pipe.xadd(
            self._stream,
            {"json": updated, "kind": "outcome_attach"},
            maxlen=self._stream_maxlen,
            approximate=True,
        )
        pipe.execute()
        return True

    def aggregate_stats(
        self,
        pattern_hash: str,
        *,
        dimension: AggregateDimension = AggregateDimension.RETRIEVAL_MODE,
    ) -> dict[str, dict[str, int]]:
        agg_key = self._agg_key(pattern_hash, dimension)
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
            bucket, metric = compound.rsplit(":", 1)
            row = stats.setdefault(bucket, {"pass": 0, "fail": 0, "queries": 0})
            try:
                row[metric] = int(count_str)
            except ValueError:
                continue
        return stats

    def recent_events(self, *, limit: int = 50) -> list[RetrievalFeedbackEvent]:
        raw_entries = self._r.xrevrange(self._stream, count=max(limit * 4, 100))
        by_id: dict[str, RetrievalFeedbackEvent] = {}
        for _entry_id, fields in raw_entries:
            payload = fields.get(b"json") or fields.get("json")
            if payload is None:
                continue
            if isinstance(payload, bytes):
                payload = payload.decode("utf-8")
            try:
                data = json.loads(payload)
                event = RetrievalFeedbackEvent(**data)
            except (json.JSONDecodeError, TypeError):
                continue
            by_id[event.request_id] = event
        ranked = sorted(by_id.values(), key=lambda e: e.ts, reverse=True)
        return ranked[:limit]


_store_singleton: Optional[FeedbackStore] = None
_store_lock = threading.Lock()


def get_feedback_store() -> FeedbackStore:
    """Process singleton: Redis when configured, else daily JSONL."""
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton

    with _store_lock:
        if _store_singleton is not None:
            return _store_singleton

        config = FeedbackConfig.load()
        from ..config.settings import REDIS_URL

        if REDIS_URL:
            try:
                import redis as _redis

                client = _redis.from_url(REDIS_URL, decode_responses=False)
                client.ping()
                _store_singleton = RedisFeedbackStore(
                    client,
                    stream_key=config.redis_stream,
                    stream_maxlen=config.stream_maxlen,
                    req_ttl_sec=config.req_ttl_sec,
                    agg_ttl_sec=config.agg_ttl_days * 86400,
                )
                logger.info("Feedback loop store: Redis stream %s", config.redis_stream)
                return _store_singleton
            except Exception as exc:
                logger.warning(
                    "Feedback loop: Redis unavailable (%s); using JSONL at %s",
                    exc,
                    config.data_dir,
                )

        _store_singleton = JsonlFeedbackStore(
            Path(config.data_dir),
            retain_days=config.jsonl_retain_days,
        )
        logger.info("Feedback loop store: JSONL %s", config.data_dir)
        return _store_singleton


def reset_feedback_store(store: Optional[FeedbackStore] = None) -> None:
    """Test hook: replace the process singleton."""
    global _store_singleton
    with _store_lock:
        _store_singleton = store
