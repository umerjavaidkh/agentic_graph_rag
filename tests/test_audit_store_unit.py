"""
tests/test_audit_store_unit.py — AuditStore (Jsonl + Redis) and record_audit_event.

Run with:
    python -m pytest tests/test_audit_store_unit.py -v
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest


from src.audit.models import AuditEvent
from src.audit.store import JsonlAuditStore, RedisAuditStore, _utc_now_iso


def _event(**overrides) -> AuditEvent:
    base = dict(
        event_id="evt1",
        ts=_utc_now_iso(),
        event_type="query",
        user_id="user_a",
        tenant_id="tenant_a",
        role="regular_office",
    )
    base.update(overrides)
    return AuditEvent(**base)


# ── JsonlAuditStore ───────────────────────────────────────────────────────────


def test_jsonl_record_and_query_round_trip(tmp_path):
    store = JsonlAuditStore(tmp_path)
    store.record(_event(event_id="e1", user_id="alice", tenant_id="tenant_a"))
    store.record(_event(event_id="e2", user_id="bob", tenant_id="tenant_b"))

    all_events = store.query(limit=10)
    assert {e.event_id for e in all_events} == {"e1", "e2"}


def test_jsonl_query_filters_by_user_id(tmp_path):
    store = JsonlAuditStore(tmp_path)
    store.record(_event(event_id="e1", user_id="alice"))
    store.record(_event(event_id="e2", user_id="bob"))

    result = store.query(user_id="alice", limit=10)
    assert [e.event_id for e in result] == ["e1"]


def test_jsonl_query_filters_by_tenant_id(tmp_path):
    store = JsonlAuditStore(tmp_path)
    store.record(_event(event_id="e1", tenant_id="tenant_a"))
    store.record(_event(event_id="e2", tenant_id="tenant_b"))

    result = store.query(tenant_id="tenant_b", limit=10)
    assert [e.event_id for e in result] == ["e2"]


def test_jsonl_query_filters_by_event_type(tmp_path):
    store = JsonlAuditStore(tmp_path)
    store.record(_event(event_id="e1", event_type="query"))
    store.record(_event(event_id="e2", event_type="access_denied"))

    result = store.query(event_type="access_denied", limit=10)
    assert [e.event_id for e in result] == ["e2"]


def test_jsonl_query_respects_limit(tmp_path):
    store = JsonlAuditStore(tmp_path)
    for i in range(5):
        store.record(_event(event_id=f"e{i}"))

    result = store.query(limit=2)
    assert len(result) == 2


def test_jsonl_query_time_range(tmp_path):
    store = JsonlAuditStore(tmp_path)
    store.record(_event(event_id="old", ts="2020-01-01T00:00:00.000000Z"))
    store.record(_event(event_id="new", ts="2030-01-01T00:00:00.000000Z"))

    result = store.query(since="2025-01-01T00:00:00.000000Z", limit=10)
    assert [e.event_id for e in result] == ["new"]

    result2 = store.query(until="2025-01-01T00:00:00.000000Z", limit=10)
    assert [e.event_id for e in result2] == ["old"]


def test_jsonl_empty_store_returns_empty(tmp_path):
    store = JsonlAuditStore(tmp_path)
    assert store.query(limit=10) == []


# ── RedisAuditStore (fake client) ─────────────────────────────────────────────


class _FakePipeline:
    def __init__(self, client: "_FakeRedis"):
        self._client = client
        self._ops: list[tuple[str, tuple, dict]] = []

    def xadd(self, *a, **kw):
        self._ops.append(("xadd", a, kw))
        return self

    def set(self, *a, **kw):
        self._ops.append(("set", a, kw))
        return self

    def zadd(self, *a, **kw):
        self._ops.append(("zadd", a, kw))
        return self

    def expire(self, *a, **kw):
        self._ops.append(("expire", a, kw))
        return self

    def execute(self):
        for name, a, kw in self._ops:
            getattr(self._client, name)(*a, **kw)
        self._ops = []


class _FakeRedis:
    """Minimal in-memory stand-in for the subset of redis-py used by RedisAuditStore."""

    def __init__(self):
        self._kv: dict[str, str] = {}
        self._zsets: dict[str, dict[str, float]] = {}
        self._stream: list[tuple[str, dict]] = []
        self._next_id = 0

    def pipeline(self):
        return _FakePipeline(self)

    def xadd(self, key, fields, maxlen=None, approximate=None):
        self._next_id += 1
        entry_id = f"{self._next_id}-0"
        self._stream.append((entry_id, dict(fields)))
        return entry_id

    def set(self, key, value, ex=None):
        self._kv[key] = value

    def zadd(self, key, mapping):
        self._zsets.setdefault(key, {}).update(mapping)

    def expire(self, key, seconds):
        pass

    def zrevrangebyscore(self, key, max_score, min_score, start=0, num=None):
        zset = self._zsets.get(key, {})

        def _score_ok(score):
            lo = -float("inf") if min_score == "-inf" else float(min_score)
            hi = float("inf") if max_score == "+inf" else float(max_score)
            return lo <= score <= hi

        items = sorted(
            ((member, score) for member, score in zset.items() if _score_ok(score)),
            key=lambda kv: kv[1],
            reverse=True,
        )
        members = [m for m, _ in items]
        if num is not None:
            members = members[start : start + num]
        return members

    def mget(self, keys):
        return [self._kv.get(k) for k in keys]

    def xrevrange(self, key, count=None):
        entries = list(reversed(self._stream))
        if count is not None:
            entries = entries[:count]
        return entries

    def ping(self):
        return True


def test_redis_record_and_query_by_user_id():
    client = _FakeRedis()
    store = RedisAuditStore(client, stream_key="test:stream", stream_maxlen=1000, req_ttl_sec=3600)

    store.record(_event(event_id="e1", user_id="alice", ts="2024-01-01T00:00:00.000000Z"))
    store.record(_event(event_id="e2", user_id="bob", ts="2024-01-02T00:00:00.000000Z"))

    result = store.query(user_id="alice", limit=10)
    assert [e.event_id for e in result] == ["e1"]


def test_redis_record_and_query_by_tenant_id():
    client = _FakeRedis()
    store = RedisAuditStore(client, stream_key="test:stream", stream_maxlen=1000, req_ttl_sec=3600)

    store.record(_event(event_id="e1", tenant_id="tenant_a", ts="2024-01-01T00:00:00.000000Z"))
    store.record(_event(event_id="e2", tenant_id="tenant_b", ts="2024-01-02T00:00:00.000000Z"))

    result = store.query(tenant_id="tenant_b", limit=10)
    assert [e.event_id for e in result] == ["e2"]


def test_redis_query_with_no_filter_falls_back_to_stream():
    client = _FakeRedis()
    store = RedisAuditStore(client, stream_key="test:stream", stream_maxlen=1000, req_ttl_sec=3600)

    store.record(_event(event_id="e1", ts="2024-01-01T00:00:00.000000Z"))
    store.record(_event(event_id="e2", ts="2024-01-02T00:00:00.000000Z"))

    result = store.query(limit=10)
    assert {e.event_id for e in result} == {"e1", "e2"}


def test_redis_query_time_range_by_user():
    client = _FakeRedis()
    store = RedisAuditStore(client, stream_key="test:stream", stream_maxlen=1000, req_ttl_sec=3600)

    store.record(_event(event_id="old", user_id="alice", ts="2020-01-01T00:00:00.000000Z"))
    store.record(_event(event_id="new", user_id="alice", ts="2030-01-01T00:00:00.000000Z"))

    result = store.query(user_id="alice", since="2025-01-01T00:00:00.000000Z", limit=10)
    assert [e.event_id for e in result] == ["new"]


# ── record_audit_event: gating + fail-open ───────────────────────────────────


def test_record_audit_event_noop_when_disabled(monkeypatch, tmp_path):
    import src.audit.record as record_mod

    class _Boom:
        def record(self, event):
            raise AssertionError("store.record should not be called when disabled")

    monkeypatch.setattr(record_mod, "get_audit_store", lambda: _Boom())
    monkeypatch.setattr(record_mod.AuditConfig, "load", classmethod(lambda cls: _cfg(enabled=False)))

    record_mod.record_audit_event(
        event_type="query", user_id="u", tenant_id="t", role="regular_office"
    )
    record_mod._executor.shutdown(wait=True)
    record_mod._executor = record_mod.ThreadPoolExecutor(max_workers=1, thread_name_prefix="audit")


def test_record_audit_event_fails_open_on_store_exception(monkeypatch):
    import src.audit.record as record_mod

    class _RaisingStore:
        def record(self, event):
            raise RuntimeError("disk full")

    monkeypatch.setattr(record_mod, "get_audit_store", lambda: _RaisingStore())
    monkeypatch.setattr(record_mod.AuditConfig, "load", classmethod(lambda cls: _cfg(enabled=True)))

    # Must not raise, even though the underlying store always raises.
    record_mod.record_audit_event(
        event_type="query", user_id="u", tenant_id="t", role="regular_office"
    )
    time.sleep(0.2)  # let the background worker actually run _write_event


def test_record_audit_event_respects_store_question_toggle(monkeypatch, tmp_path):
    import src.audit.record as record_mod

    captured = {}

    class _CapturingStore:
        def record(self, event):
            captured["event"] = event

    monkeypatch.setattr(record_mod, "get_audit_store", lambda: _CapturingStore())
    monkeypatch.setattr(
        record_mod.AuditConfig, "load", classmethod(lambda cls: _cfg(enabled=True, store_question=False))
    )

    record_mod.record_audit_event(
        event_type="query", user_id="u", tenant_id="t", role="regular_office", action="what is the budget?"
    )
    time.sleep(0.2)
    assert captured["event"].action is None


def _cfg(*, enabled: bool, store_question: bool = True):
    from src.audit.config import AuditConfig

    return AuditConfig(
        enabled=enabled,
        store_question=store_question,
        data_dir="/tmp/unused",
        jsonl_retain_days=90,
        redis_stream="test:stream",
        stream_maxlen=1000,
        req_ttl_sec=3600,
    )
