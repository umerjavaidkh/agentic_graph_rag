"""Unit tests for retrieval feedback (extract + JSONL store + hints)."""
from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

from src.telemetry.feedback.extract import (
    build_feedback_event,
    compact_pipeline,
    extract_retrieval_profile,
    pattern_hash,
    retrieval_pattern,
)
from src.telemetry.feedback.hints import best_mode_for_question, invalidate_hint_cache
from src.telemetry.feedback.models import RetrievalFeedbackEvent
from src.telemetry.feedback.sink import maybe_record_retrieval_feedback
from src.telemetry.feedback.store import JsonlFeedbackStore, reset_feedback_store


class TestFeedbackExtract(unittest.TestCase):
    def test_compact_pipeline_keeps_retrieval_steps_only(self):
        pipeline = [
            {"step": "route.select", "status": "ok", "duration_ms": 1, "meta": {}},
            {"step": "chat.synthesis", "status": "ok", "duration_ms": 50, "meta": {"tokens": 999}},
            {
                "step": "document.hybrid.merge",
                "status": "ok",
                "duration_ms": 12,
                "meta": {"mode": "graph_rag_hybrid", "vector_seeds": 3},
            },
        ]
        compact = compact_pipeline(pipeline)
        self.assertEqual(len(compact), 2)
        self.assertEqual(compact[-1]["meta"]["mode"], "graph_rag_hybrid")

    def test_extract_document_hybrid_mode(self):
        telemetry = {
            "pipeline": [
                {
                    "step": "document.hybrid.merge",
                    "status": "ok",
                    "meta": {
                        "mode": "graph_rag_lexical",
                        "vector_seeds": 0,
                        "fulltext_hits": 2,
                        "returned": 5,
                    },
                }
            ]
        }
        mode, profile = extract_retrieval_profile(telemetry, agent="unstructured")
        self.assertEqual(mode, "graph_rag_lexical")
        self.assertEqual(profile["path"], "document")

    def test_pattern_from_intent_flags(self):
        q = "Compare whistleblowing policy across sections and summarize differences"
        pattern = retrieval_pattern(q, agent="unstructured")
        self.assertIn("synthesis", pattern)
        self.assertIn("agent:unstructured", pattern)


class TestJsonlFeedbackStore(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = JsonlFeedbackStore(Path(self._tmpdir.name), retain_days=7)
        reset_feedback_store(self.store)
        invalidate_hint_cache()

    def tearDown(self):
        reset_feedback_store(None)
        invalidate_hint_cache()
        self._tmpdir.cleanup()

    def _event(
        self,
        request_id: str,
        *,
        mode: str = "graph_rag_hybrid",
        pattern: str = "agent:unstructured|synthesis",
        outcome: bool | None = None,
    ) -> RetrievalFeedbackEvent:
        return RetrievalFeedbackEvent(
            request_id=request_id,
            ts="2026-06-04T12:00:00.000000Z",
            question_hash="abc123",
            agent="unstructured",
            strategy="graph_rag",
            route_tool="search_documents",
            route_method="llm_mcp",
            pattern=pattern,
            pattern_hash="p" * 16,
            retrieval_mode=mode,
            outcome=outcome,
        )

    def test_record_and_aggregate(self):
        self.store.record(self._event("r1", mode="graph_rag_lexical", outcome=True))
        self.store.record(self._event("r2", mode="graph_rag_lexical", outcome=True))
        self.store.record(self._event("r3", mode="graph_rag_hybrid", outcome=False))

        stats = self.store.aggregate_stats("p" * 16)
        self.assertEqual(stats["graph_rag_lexical"]["pass"], 2)
        self.assertEqual(stats["graph_rag_hybrid"]["fail"], 1)

    def test_attach_outcome_updates_aggregate(self):
        self.store.record(self._event("r9", mode="graph_rag_lexical", outcome=None))
        ok = self.store.attach_outcome("r9", passed=True, case_id="doc_01")
        self.assertTrue(ok)
        stats = self.store.aggregate_stats("p" * 16)
        self.assertEqual(stats["graph_rag_lexical"]["pass"], 1)

    def test_hint_requires_min_samples(self):
        question = "summarize the policy"
        pattern = retrieval_pattern(question, agent="unstructured")
        p_hash = pattern_hash(pattern)
        for i in range(5):
            ev = self._event(f"r{i}", mode="graph_rag_lexical", outcome=True)
            ev.pattern = pattern
            ev.pattern_hash = p_hash
            self.store.record(ev)
        hint = best_mode_for_question(
            question,
            agent="unstructured",
            min_samples=30,
            min_margin=0.15,
            cache_sec=0,
        )
        self.assertIsNone(hint)


class TestFeedbackSinkNoOp(unittest.TestCase):
    def test_disabled_by_default(self):
        # Should not raise when feedback env is false (default in tests).
        maybe_record_retrieval_feedback(
            request_id="x",
            question="test",
            result={"agent": "unstructured", "_telemetry": {}},
        )


if __name__ == "__main__":
    unittest.main()
