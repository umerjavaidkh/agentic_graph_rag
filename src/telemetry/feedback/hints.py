"""Read-side hints from aggregated feedback (opt-in routing only)."""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from .extract import pattern_hash, retrieval_pattern
from .store import get_feedback_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModeHint:
    mode: str
    pass_rate: float
    samples: int
    confidence: float


_cache_lock = threading.Lock()
_hint_cache: dict[str, tuple[float, Optional[ModeHint]]] = {}


def _score_mode(bucket: dict[str, int]) -> tuple[float, int]:
    passed = int(bucket.get("pass") or 0)
    failed = int(bucket.get("fail") or 0)
    labeled = passed + failed
    if labeled <= 0:
        return 0.0, 0
    return passed / labeled, labeled


def best_mode_for_question(
    question: str,
    *,
    agent: str = "",
    min_samples: int,
    min_margin: float,
    cache_sec: int,
) -> Optional[ModeHint]:
    """
    Return the historically best retrieval mode for this question pattern.

    Read-only; returns None when data is insufficient. Never affects retrieval
    unless the caller explicitly acts on the hint.
    """
    pattern = retrieval_pattern(question, agent=agent)
    p_hash = pattern_hash(pattern)
    now = time.monotonic()

    with _cache_lock:
        cached = _hint_cache.get(p_hash)
        if cached and (now - cached[0]) < cache_sec:
            return cached[1]

    stats = get_feedback_store().aggregate_stats(p_hash)
    if not stats:
        hint = None
    else:
        ranked: list[tuple[str, float, int]] = []
        for mode, bucket in stats.items():
            rate, labeled = _score_mode(bucket)
            if labeled > 0:
                ranked.append((mode, rate, labeled))
        ranked.sort(key=lambda x: (-x[1], -x[2], x[0]))

        hint = None
        if ranked:
            best_mode, best_rate, best_n = ranked[0]
            if best_n >= min_samples:
                second_rate = ranked[1][1] if len(ranked) > 1 else 0.0
                margin = best_rate - second_rate
                if margin >= min_margin:
                    hint = ModeHint(
                        mode=best_mode,
                        pass_rate=best_rate,
                        samples=best_n,
                        confidence=min(1.0, margin + best_rate),
                    )

    with _cache_lock:
        _hint_cache[p_hash] = (now, hint)
    return hint


def invalidate_hint_cache() -> None:
    with _cache_lock:
        _hint_cache.clear()
