"""Read-side hints from aggregated feedback."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional

from .config import FeedbackConfig
from .models import AggregateDimension
from .pattern import pattern_hash, retrieval_pattern
from .store import get_feedback_store


@dataclass(frozen=True)
class ModeHint:
    mode: str
    pass_rate: float
    samples: int
    confidence: float


_cache_lock = threading.Lock()
_hint_cache: dict[tuple[str, str, str], tuple[float, Optional[ModeHint]]] = {}


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
    dimension: AggregateDimension = AggregateDimension.RETRIEVAL_MODE,
    min_samples: int,
    min_margin: float,
    cache_sec: int,
) -> Optional[ModeHint]:
    """
    Return the historically best bucket for this question pattern.

    Read-only; returns None when data is insufficient.
    """
    pattern = retrieval_pattern(question, agent=agent)
    p_hash = pattern_hash(pattern)
    cache_key = (p_hash, agent, dimension.value)
    now = time.monotonic()

    with _cache_lock:
        cached = _hint_cache.get(cache_key)
        if cached and (now - cached[0]) < cache_sec:
            return cached[1]

    stats = get_feedback_store().aggregate_stats(p_hash, dimension=dimension)
    hint = _rank_stats(stats, min_samples=min_samples, min_margin=min_margin)

    with _cache_lock:
        _hint_cache[cache_key] = (now, hint)
    return hint


def _rank_stats(
    stats: dict[str, dict[str, int]],
    *,
    min_samples: int,
    min_margin: float,
) -> Optional[ModeHint]:
    ranked: list[tuple[str, float, int]] = []
    for mode, bucket in stats.items():
        rate, labeled = _score_mode(bucket)
        if labeled > 0:
            ranked.append((mode, rate, labeled))
    ranked.sort(key=lambda x: (-x[1], -x[2], x[0]))

    if not ranked:
        return None

    best_mode, best_rate, best_n = ranked[0]
    if best_n < min_samples:
        return None
    second_rate = ranked[1][1] if len(ranked) > 1 else 0.0
    margin = best_rate - second_rate
    if margin < min_margin:
        return None
    return ModeHint(
        mode=best_mode,
        pass_rate=best_rate,
        samples=best_n,
        confidence=min(1.0, margin + best_rate),
    )


def hint_for_question(
    question: str,
    *,
    agent: str = "",
    dimension: AggregateDimension = AggregateDimension.RETRIEVAL_MODE,
    config: Optional[FeedbackConfig] = None,
) -> Optional[ModeHint]:
    """Convenience wrapper using loaded FeedbackConfig thresholds."""
    cfg = config or FeedbackConfig.load()
    return best_mode_for_question(
        question,
        agent=agent,
        dimension=dimension,
        min_samples=cfg.min_samples,
        min_margin=cfg.min_margin,
        cache_sec=cfg.hint_cache_sec,
    )


def invalidate_hint_cache() -> None:
    with _cache_lock:
        _hint_cache.clear()
