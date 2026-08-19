"""Feedback loop configuration — single source loaded from environment."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeedbackConfig:
    enabled: bool
    routing_enabled: bool
    store_question: bool
    data_dir: str
    jsonl_retain_days: int
    redis_stream: str
    stream_maxlen: int
    req_ttl_sec: int
    agg_ttl_days: int
    min_samples: int
    min_margin: float
    hint_cache_sec: int

    @classmethod
    def load(cls) -> FeedbackConfig:
        from ..config.settings import (
            RETRIEVAL_FEEDBACK_AGG_TTL_DAYS,
            RETRIEVAL_FEEDBACK_DIR,
            RETRIEVAL_FEEDBACK_ENABLED,
            RETRIEVAL_FEEDBACK_HINT_CACHE_SEC,
            RETRIEVAL_FEEDBACK_JSONL_RETAIN_DAYS,
            RETRIEVAL_FEEDBACK_MIN_MARGIN,
            RETRIEVAL_FEEDBACK_MIN_SAMPLES,
            RETRIEVAL_FEEDBACK_REDIS_STREAM,
            RETRIEVAL_FEEDBACK_REQ_TTL_SEC,
            RETRIEVAL_FEEDBACK_ROUTING,
            RETRIEVAL_FEEDBACK_STORE_QUESTION,
            RETRIEVAL_FEEDBACK_STREAM_MAXLEN,
        )

        return cls(
            enabled=RETRIEVAL_FEEDBACK_ENABLED,
            routing_enabled=RETRIEVAL_FEEDBACK_ROUTING,
            store_question=RETRIEVAL_FEEDBACK_STORE_QUESTION,
            data_dir=RETRIEVAL_FEEDBACK_DIR,
            jsonl_retain_days=RETRIEVAL_FEEDBACK_JSONL_RETAIN_DAYS,
            redis_stream=RETRIEVAL_FEEDBACK_REDIS_STREAM,
            stream_maxlen=RETRIEVAL_FEEDBACK_STREAM_MAXLEN,
            req_ttl_sec=RETRIEVAL_FEEDBACK_REQ_TTL_SEC,
            agg_ttl_days=RETRIEVAL_FEEDBACK_AGG_TTL_DAYS,
            min_samples=RETRIEVAL_FEEDBACK_MIN_SAMPLES,
            min_margin=RETRIEVAL_FEEDBACK_MIN_MARGIN,
            hint_cache_sec=RETRIEVAL_FEEDBACK_HINT_CACHE_SEC,
        )
