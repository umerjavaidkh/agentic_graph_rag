"""Retrieval feedback — observe pipeline steps, learn offline, route only when opted in."""
from .extract import (
    compact_pipeline,
    extract_retrieval_profile,
    pattern_hash,
    question_hash,
    retrieval_pattern,
)
from .hints import ModeHint, best_mode_for_question, invalidate_hint_cache
from .sink import maybe_attach_feedback_outcome, maybe_record_retrieval_feedback

__all__ = [
    "ModeHint",
    "best_mode_for_question",
    "compact_pipeline",
    "extract_retrieval_profile",
    "invalidate_hint_cache",
    "maybe_attach_feedback_outcome",
    "maybe_record_retrieval_feedback",
    "pattern_hash",
    "question_hash",
    "retrieval_pattern",
]
