"""Deprecated — use src.feedback_loop instead."""
from __future__ import annotations

from ...feedback_loop import (
    ModeHint,
    RetrievalFeedbackEvent,
    best_mode_for_question,
    build_feedback_event,
    compact_pipeline,
    extract_retrieval_profile,
    invalidate_hint_cache,
    maybe_attach_feedback_outcome,
    maybe_record_retrieval_feedback,
    pattern_hash,
    question_hash,
    retrieval_pattern,
)
from ...feedback_loop.routing.service import (
    FeedbackRoutingService,
    document_mode_hint,
    get_feedback_routing,
    record_hint_applied,
    retrieval_hint,
    structured_path_hint,
)
from ...feedback_loop.store import JsonlFeedbackStore, get_feedback_store, reset_feedback_store

__all__ = [
    "FeedbackRoutingService",
    "JsonlFeedbackStore",
    "ModeHint",
    "RetrievalFeedbackEvent",
    "best_mode_for_question",
    "build_feedback_event",
    "compact_pipeline",
    "document_mode_hint",
    "extract_retrieval_profile",
    "get_feedback_routing",
    "get_feedback_store",
    "invalidate_hint_cache",
    "maybe_attach_feedback_outcome",
    "maybe_record_retrieval_feedback",
    "pattern_hash",
    "question_hash",
    "record_hint_applied",
    "retrieval_hint",
    "retrieval_pattern",
    "reset_feedback_store",
    "structured_path_hint",
]
