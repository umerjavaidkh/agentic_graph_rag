"""Unit tests for the feedback_loop package."""
from __future__ import annotations

import tempfile
import unittest
import unittest.mock
from pathlib import Path

from src.feedback_loop import (
    AggregateDimension,
    JsonlFeedbackStore,
    RetrievalFeedbackEvent,
    best_mode_for_question,
    build_feedback_event,
    compact_pipeline,
    extract_retrieval_profile,
    get_feedback_routing,
    invalidate_hint_cache,
    maybe_record_retrieval_feedback,
    pattern_hash,
    reset_feedback_routing,
    reset_feedback_store,
    retrieval_pattern,
)
from src.feedback_loop.routing.service import FeedbackRoutingService


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
        reset_feedback_routing(None)
        invalidate_hint_cache()
        self._tmpdir.cleanup()

    def _event(
        self,
        request_id: str,
        *,
        mode: str = "graph_rag_hybrid",
        route_tool: str = "search_documents",
        pattern: str = "agent:unstructured|synthesis",
        outcome: bool | None = None,
    ) -> RetrievalFeedbackEvent:
        return RetrievalFeedbackEvent(
            request_id=request_id,
            ts="2026-06-04T12:00:00.000000Z",
            question_hash="abc123",
            agent="unstructured",
            strategy="graph_rag",
            route_tool=route_tool,
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

    def test_route_tool_aggregate_dimension(self):
        pattern = "synthesis|general"
        p_hash = pattern_hash(pattern)
        for i in range(35):
            ev = self._event(
                f"r{i}",
                route_tool="query_data",
                pattern=pattern,
                outcome=True,
            )
            ev.pattern_hash = p_hash
            self.store.record(ev)
        for i in range(5):
            ev = self._event(
                f"bad{i}",
                route_tool="search_documents",
                pattern=pattern,
                outcome=False,
            )
            ev.pattern_hash = p_hash
            self.store.record(ev)

        stats = self.store.aggregate_stats(p_hash, dimension=AggregateDimension.ROUTE_TOOL)
        self.assertEqual(stats["query_data"]["pass"], 35)
        self.assertEqual(stats["search_documents"]["fail"], 5)

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


class TestFeedbackRoutingService(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.store = JsonlFeedbackStore(Path(self._tmpdir.name), retain_days=7)
        reset_feedback_store(self.store)
        invalidate_hint_cache()
        config = unittest.mock.Mock(
            routing_enabled=True,
            min_samples=30,
            min_margin=0.15,
            hint_cache_sec=0,
        )
        reset_feedback_routing(FeedbackRoutingService(config=config))

    def tearDown(self):
        reset_feedback_store(None)
        reset_feedback_routing(None)
        invalidate_hint_cache()
        self._tmpdir.cleanup()

    def _seed(
        self,
        question: str,
        *,
        agent: str,
        bucket: str,
        dimension: AggregateDimension,
        passed: bool,
        n: int = 35,
    ) -> None:
        pattern = retrieval_pattern(question, agent=agent)
        p_hash = pattern_hash(pattern)
        for i in range(n):
            ev = RetrievalFeedbackEvent(
                request_id=f"{bucket}-{i}",
                ts="2026-06-04T12:00:00.000000Z",
                question_hash="abc",
                agent=agent,
                strategy="",
                route_tool=bucket if dimension is AggregateDimension.ROUTE_TOOL else "search_documents",
                route_method="llm_mcp",
                pattern=pattern,
                pattern_hash=p_hash,
                retrieval_mode=bucket,
                outcome=passed,
            )
            self.store.record(ev)
        for i in range(5):
            ev = RetrievalFeedbackEvent(
                request_id=f"alt-{bucket}-{i}",
                ts="2026-06-04T12:00:00.000000Z",
                question_hash="abc",
                agent=agent,
                strategy="",
                route_tool="search_documents",
                route_method="llm_mcp",
                pattern=pattern,
                pattern_hash=p_hash,
                retrieval_mode="other_mode",
                outcome=False,
            )
            self.store.record(ev)

    def test_structured_path_policy(self):
        q = "top 3 products per category by revenue"
        self._seed(q, agent="structured", bucket="structured_multistep:gate", dimension=AggregateDimension.RETRIEVAL_MODE, passed=True)
        self.assertEqual(get_feedback_routing().structured_path(q), "multistep_first")

    def test_document_mode_policy(self):
        q = "summarize whistleblowing policy differences"
        self._seed(q, agent="unstructured", bucket="graph_rag_lexical", dimension=AggregateDimension.RETRIEVAL_MODE, passed=True)
        self.assertEqual(get_feedback_routing().document_mode(q), "graph_rag_lexical")

    def test_route_tool_override(self):
        q = "Compare revenue trends and policy sections"
        self._seed(
            q,
            agent="",
            bucket="query_data",
            dimension=AggregateDimension.ROUTE_TOOL,
            passed=True,
        )
        chosen = get_feedback_routing().route_tool(q, "search_documents")
        self.assertEqual(chosen, "query_data")


class TestFeedbackSinkNoOp(unittest.TestCase):
    def test_disabled_by_default(self):
        maybe_record_retrieval_feedback(
            request_id="x",
            question="test",
            result={"agent": "unstructured", "_telemetry": {}},
        )


if __name__ == "__main__":
    unittest.main()
