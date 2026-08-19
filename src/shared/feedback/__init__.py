"""
Retrieval feedback loop — observe, label, learn, optionally apply.

Public API for the rest of the application. Import from here, not from
telemetry.feedback (deprecated shim).
"""
from .config import FeedbackConfig
from .hints import ModeHint, best_mode_for_question, hint_for_question, invalidate_hint_cache
from .models import AggregateDimension, RetrievalFeedbackEvent
from .pattern import pattern_hash, question_hash, retrieval_pattern
from .profile import build_feedback_event, compact_pipeline, extract_retrieval_profile
from .record import maybe_attach_feedback_outcome, maybe_record_retrieval_feedback
from .dashboard import build_dashboard_overview
from .resolver import resolve_query_tool
from .routing import FeedbackRoutingService, get_feedback_routing
from .store import FeedbackStore, JsonlFeedbackStore, get_feedback_store, reset_feedback_store
from .routing.service import reset_feedback_routing

__all__ = [
    "AggregateDimension",
    "FeedbackConfig",
    "FeedbackRoutingService",
    "FeedbackStore",
    "JsonlFeedbackStore",
    "ModeHint",
    "RetrievalFeedbackEvent",
    "best_mode_for_question",
    "build_dashboard_overview",
    "build_feedback_event",
    "compact_pipeline",
    "extract_retrieval_profile",
    "get_feedback_routing",
    "get_feedback_store",
    "hint_for_question",
    "invalidate_hint_cache",
    "maybe_attach_feedback_outcome",
    "maybe_record_retrieval_feedback",
    "pattern_hash",
    "question_hash",
    "resolve_query_tool",
    "reset_feedback_routing",
    "reset_feedback_store",
    "retrieval_pattern",
]
