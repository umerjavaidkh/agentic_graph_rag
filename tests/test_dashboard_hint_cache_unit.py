"""tests/test_dashboard_hint_cache_unit.py — dashboard hint lookups use a
short cache TTL instead of cache_sec=0.

Covers the fix for build_dashboard_overview calling best_mode_for_question
twice per pattern (retrieval + route hint) with caching fully disabled —
up to 2 * pattern_limit real feedback-store scans on every dashboard load,
none of them ever reused even for back-to-back requests a second apart.

Run with:
    python -m pytest tests/test_dashboard_hint_cache_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.feedback_loop import dashboard as dashboard_mod
from src.feedback_loop.config import FeedbackConfig
from src.feedback_loop.dashboard import build_dashboard_overview
from src.feedback_loop.models import RetrievalFeedbackEvent


class _FakeStore:
    def __init__(self, events):
        self._events = events

    def recent_events(self, limit: int = 50):
        return self._events[:limit]


def _event(pattern: str = "what is $X") -> RetrievalFeedbackEvent:
    return RetrievalFeedbackEvent(
        request_id="r1",
        ts="2026-01-01T00:00:00Z",
        question_hash="h1",
        agent="unstructured",
        strategy="graph_rag",
        route_tool="search_documents",
        route_method="llm_mcp",
        pattern=pattern,
        pattern_hash="ph1",
        retrieval_mode="graph_rag_hybrid",
        outcome=True,
    )


def _config() -> FeedbackConfig:
    return FeedbackConfig(
        enabled=True,
        routing_enabled=False,
        store_question=False,
        data_dir="/tmp/unused",
        jsonl_retain_days=30,
        redis_stream="unused",
        stream_maxlen=1000,
        req_ttl_sec=3600,
        agg_ttl_days=30,
        min_samples=1,
        min_margin=0.0,
        hint_cache_sec=60,
    )


def test_dashboard_passes_nonzero_cache_sec(monkeypatch):
    seen_cache_secs: list[int] = []

    def _fake_best_mode_for_question(*args, **kwargs):
        seen_cache_secs.append(kwargs["cache_sec"])
        return None

    monkeypatch.setattr(dashboard_mod, "best_mode_for_question", _fake_best_mode_for_question)

    build_dashboard_overview(store=_FakeStore([_event()]), config=_config())

    assert seen_cache_secs, "best_mode_for_question was never called"
    assert all(cs > 0 for cs in seen_cache_secs)
    assert all(cs == dashboard_mod._DASHBOARD_HINT_CACHE_SEC for cs in seen_cache_secs)
