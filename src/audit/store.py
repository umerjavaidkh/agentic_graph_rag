"""Durable audit storage: Redis (production) or daily JSONL (dev)."""
from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from .config import AuditConfig
from .models import AuditEvent

logger = logging.getLogger(__name__)

_TS_FMT = "%Y-%m-%dT%H:%M:%S.%fZ"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime(_TS_FMT)


def _ts_to_epoch_ms(ts: str) -> int:
    dt = datetime.strptime(ts, _TS_FMT).replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _matches(event: AuditEvent, *, user_id, tenant_id, event_type, since, until) -> bool:
    if user_id is not None and event.user_id != user_id:
        return False
    if tenant_id is not None and event.tenant_id != tenant_id:
        return False
    if event_type is not None and event.event_type != event_type:
        return False
    if since is not None and event.ts < since:
        return False
    if until is not None and event.ts > until:
        return False
    return True


class AuditStore(ABC):
    @abstractmethod
    def record(self, event: AuditEvent) -> None:
        """Persist one audit event (append-only)."""

    @abstractmethod
    def query(
        self,
        *,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        """Most recent matching events, newest first."""


class JsonlAuditStore(AuditStore):
    """Dev/single-node backend: one JSONL file per UTC day."""

    def __init__(self, directory: Path, *, retain_days: int = 90) -> None:
        self._dir = directory
        self._dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._retain_days = max(1, retain_days)

    def _path_for_ts(self, ts: str) -> Path:
        day = (ts or _utc_now_iso())[:10]
        return self._dir / f"{day}.jsonl"

    def record(self, event: AuditEvent) -> None:
        line = json.dumps(event.to_dict(), separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            path = self._path_for_ts(event.ts)
            with path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def query(
        self,
        *,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        matched: list[AuditEvent] = []
        files = sorted(self._dir.glob("*.jsonl"), reverse=True)
        for path in files[: self._retain_days]:
            # Cheap prune: filenames are "YYYY-MM-DD.jsonl" — skip files
            # entirely outside the requested date range.
            day = path.stem
            if since is not None and day < since[:10]:
                continue
            if until is not None and day > until[:10]:
                continue
            try:
                for line in reversed(path.read_text(encoding="utf-8").splitlines()):
                    if not line.strip():
                        continue
                    data = json.loads(line)
                    event = AuditEvent(**data)
                    if _matches(
                        event,
                        user_id=user_id,
                        tenant_id=tenant_id,
                        event_type=event_type,
                        since=since,
                        until=until,
                    ):
                        matched.append(event)
                        if len(matched) >= limit:
                            return matched
            except (OSError, json.JSONDecodeError, TypeError):
                continue
        return matched


class RedisAuditStore(AuditStore):
    """Production backend: Redis Stream + per-user/per-tenant sorted-set indexes."""

    _EVENT_PREFIX = "rag:audit:event:"
    _USER_IDX_PREFIX = "rag:audit:idx:user:"
    _TENANT_IDX_PREFIX = "rag:audit:idx:tenant:"

    def __init__(
        self,
        redis_client: Any,
        *,
        stream_key: str,
        stream_maxlen: int,
        req_ttl_sec: int,
    ) -> None:
        self._r = redis_client
        self._stream = stream_key
        self._stream_maxlen = max(1000, stream_maxlen)
        self._req_ttl = max(3600, req_ttl_sec)

    def record(self, event: AuditEvent) -> None:
        payload = event.to_dict()
        compact_json = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        score = _ts_to_epoch_ms(event.ts)

        pipe = self._r.pipeline()
        pipe.xadd(
            self._stream,
            {"json": compact_json},
            maxlen=self._stream_maxlen,
            approximate=True,
        )
        pipe.set(f"{self._EVENT_PREFIX}{event.event_id}", compact_json, ex=self._req_ttl)
        pipe.zadd(f"{self._USER_IDX_PREFIX}{event.user_id}", {event.event_id: score})
        pipe.expire(f"{self._USER_IDX_PREFIX}{event.user_id}", self._req_ttl)
        pipe.zadd(f"{self._TENANT_IDX_PREFIX}{event.tenant_id}", {event.event_id: score})
        pipe.expire(f"{self._TENANT_IDX_PREFIX}{event.tenant_id}", self._req_ttl)
        pipe.execute()

    def _fetch_events(self, event_ids: list[Any]) -> list[AuditEvent]:
        if not event_ids:
            return []
        keys = []
        for eid in event_ids:
            eid = eid.decode() if isinstance(eid, bytes) else str(eid)
            keys.append(f"{self._EVENT_PREFIX}{eid}")
        raw_values = self._r.mget(keys)
        events: list[AuditEvent] = []
        for raw in raw_values:
            if raw is None:
                continue
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                events.append(AuditEvent(**json.loads(raw)))
            except (json.JSONDecodeError, TypeError):
                continue
        return events

    def query(
        self,
        *,
        user_id: Optional[str] = None,
        tenant_id: Optional[str] = None,
        event_type: Optional[str] = None,
        since: Optional[str] = None,
        until: Optional[str] = None,
        limit: int = 100,
    ) -> list[AuditEvent]:
        min_score = _ts_to_epoch_ms(since) if since else "-inf"
        max_score = _ts_to_epoch_ms(until) if until else "+inf"

        if user_id is not None:
            idx_key = f"{self._USER_IDX_PREFIX}{user_id}"
            ids = self._r.zrevrangebyscore(idx_key, max_score, min_score, start=0, num=limit * 3)
            events = self._fetch_events(ids)
        elif tenant_id is not None:
            idx_key = f"{self._TENANT_IDX_PREFIX}{tenant_id}"
            ids = self._r.zrevrangebyscore(idx_key, max_score, min_score, start=0, num=limit * 3)
            events = self._fetch_events(ids)
        else:
            raw_entries = self._r.xrevrange(self._stream, count=max(limit * 3, 100))
            events = []
            for _entry_id, fields in raw_entries:
                payload = fields.get(b"json") or fields.get("json")
                if payload is None:
                    continue
                if isinstance(payload, bytes):
                    payload = payload.decode("utf-8")
                try:
                    events.append(AuditEvent(**json.loads(payload)))
                except (json.JSONDecodeError, TypeError):
                    continue

        matched = [
            e
            for e in events
            if _matches(
                e,
                user_id=user_id,
                tenant_id=tenant_id,
                event_type=event_type,
                since=since,
                until=until,
            )
        ]
        matched.sort(key=lambda e: e.ts, reverse=True)
        return matched[:limit]


_store_singleton: Optional[AuditStore] = None
_store_lock = threading.Lock()


def get_audit_store() -> AuditStore:
    """Process singleton: Redis when configured, else daily JSONL."""
    global _store_singleton
    if _store_singleton is not None:
        return _store_singleton

    with _store_lock:
        if _store_singleton is not None:
            return _store_singleton

        config = AuditConfig.load()
        from ..config.settings import REDIS_URL

        if REDIS_URL:
            try:
                import redis as _redis

                client = _redis.from_url(REDIS_URL, decode_responses=False)
                client.ping()
                _store_singleton = RedisAuditStore(
                    client,
                    stream_key=config.redis_stream,
                    stream_maxlen=config.stream_maxlen,
                    req_ttl_sec=config.req_ttl_sec,
                )
                logger.info("Audit log store: Redis stream %s", config.redis_stream)
                return _store_singleton
            except Exception as exc:
                logger.warning(
                    "Audit log: Redis unavailable (%s); using JSONL at %s",
                    exc,
                    config.data_dir,
                )

        _store_singleton = JsonlAuditStore(
            Path(config.data_dir),
            retain_days=config.jsonl_retain_days,
        )
        logger.info("Audit log store: JSONL %s", config.data_dir)
        return _store_singleton


def reset_audit_store(store: Optional[AuditStore] = None) -> None:
    """Test hook: replace the process singleton."""
    global _store_singleton
    with _store_lock:
        _store_singleton = store
