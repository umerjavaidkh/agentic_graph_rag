"""Audit logging configuration — single source loaded from environment."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AuditConfig:
    enabled: bool
    store_question: bool
    data_dir: str
    jsonl_retain_days: int
    redis_stream: str
    stream_maxlen: int
    req_ttl_sec: int

    @classmethod
    def load(cls) -> AuditConfig:
        from ..config.settings import (
            AUDIT_LOG_DIR,
            AUDIT_LOG_ENABLED,
            AUDIT_LOG_JSONL_RETAIN_DAYS,
            AUDIT_LOG_REDIS_STREAM,
            AUDIT_LOG_REQ_TTL_SEC,
            AUDIT_LOG_STORE_QUESTION,
            AUDIT_LOG_STREAM_MAXLEN,
        )

        return cls(
            enabled=AUDIT_LOG_ENABLED,
            store_question=AUDIT_LOG_STORE_QUESTION,
            data_dir=AUDIT_LOG_DIR,
            jsonl_retain_days=AUDIT_LOG_JSONL_RETAIN_DAYS,
            redis_stream=AUDIT_LOG_REDIS_STREAM,
            stream_maxlen=AUDIT_LOG_STREAM_MAXLEN,
            req_ttl_sec=AUDIT_LOG_REQ_TTL_SEC,
        )
