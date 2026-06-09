"""
retrieval/structured/retriever.py — Structured Neo4j retriever facade.

Orchestrates schema, Text-to-Cypher, multistep planning, RBAC, and formatting.
Implementation details live in subpackages (cypher/, multistep/, schema/, etc.).
"""

from __future__ import annotations

from typing import Optional

from ...auth.rbac_setup import GraphRBAC
from ...auth.roles import DEFAULT_PUBLIC_CONTEXT, UserContext
from ...config.settings import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    STRUCTURED_ALWAYS_MULTISTEP_PLAN,
    STRUCTURED_EMPTY_MULTISTEP_FALLBACK,
)
from ...graph.driver import get_neo4j_driver
from ...telemetry import pipeline_step
from .cypher.generator import OpenAICypherGenerator
from .cypher.pipeline import Text2CypherPipeline
from .formatting.chunks import format_response
from .multistep.executor import MultiStepExecutor
from .multistep.planner import MultiStepPlanner
from .policies.clarification import needs_clarification
from .policies.rbac import StructuredRbac
from ...feedback_loop import get_feedback_routing
from .query_intent import likely_needs_multistep_plan
from .schema.provider import SchemaProvider


class StructuredRetriever:
    """Thin facade over structured retrieval subsystems."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD,
        user_context: Optional[UserContext] = None,
    ):
        self.driver = get_neo4j_driver(uri, user, password)
        self.user_context = user_context or DEFAULT_PUBLIC_CONTEXT
        self.rbac = GraphRBAC(uri, user, password, driver=self.driver)
        self._rbac = StructuredRbac(self.rbac)
        self._schema = SchemaProvider(self.driver)
        self._cypher = OpenAICypherGenerator()
        self._planner = MultiStepPlanner()
        self._text2cypher_pipeline = Text2CypherPipeline(
            self.driver,
            self._schema,
            self._cypher,
            can_query=self._rbac.can_query,
        )
        self._multistep = MultiStepExecutor(
            self.driver,
            self._schema,
            self._cypher,
            can_query=self._rbac.can_query,
        )

    def close(self) -> None:
        """No-op: driver is process-wide; use close_neo4j_driver() on shutdown."""

    def _run_multistep(
        self,
        query: str,
        schema: str,
        user_context: UserContext,
        *,
        reason: str = "gate",
    ) -> list | None:
        """Plan and execute multistep Cypher; return chunks or None if not applicable."""
        with pipeline_step("structured.multistep.plan", reason=reason):
            plan = self._planner.plan(query, schema)
        if not plan or not plan.needs_multistep or not plan.steps:
            return None
        with pipeline_step(
            "structured.multistep.execute",
            steps=len(plan.steps),
            reason=reason,
        ):
            return self._multistep.execute(plan, user_context=user_context, query=query)

    def retrieve(
        self,
        query: str,
        limit: int = 5,
        user_context: Optional[UserContext] = None,
    ) -> dict:
        ctx = user_context or self.user_context
        with pipeline_step("structured.retrieve", limit=limit):
            clarification = needs_clarification(query)
            if clarification:
                return clarification

            schema = self._schema.fetch()
            routing = get_feedback_routing()
            use_multistep = STRUCTURED_ALWAYS_MULTISTEP_PLAN or likely_needs_multistep_plan(query)
            path_hint = routing.structured_path(query)
            if path_hint == "text2cypher_first" and not STRUCTURED_ALWAYS_MULTISTEP_PLAN:
                use_multistep = False
            elif path_hint == "multistep_first":
                use_multistep = True

            if use_multistep:
                chunks = self._run_multistep(query, schema, ctx, reason="gate")
                if chunks is not None:
                    return format_response(query, chunks, strategy="multistep")

            with pipeline_step("structured.text2cypher"):
                chunks = self._text2cypher_pipeline.run(query, limit, user_context=ctx)

            if (
                STRUCTURED_EMPTY_MULTISTEP_FALLBACK
                and not chunks
                and not any(c.get("id") == "error" for c in chunks)
            ):
                fallback_chunks = self._run_multistep(
                    query, schema, ctx, reason="empty_text2cypher"
                )
                if fallback_chunks:
                    return format_response(query, fallback_chunks, strategy="multistep")

            return format_response(query, chunks, strategy="text2cypher")

    def get_schema(self) -> dict:
        schema = self._schema.fetch()
        return {
            "query": "schema",
            "chunks": [{"id": "schema", "title": "Graph Schema", "text": schema, "related": []}],
            "total_available": 1,
        }

    # Backward-compatible hooks for tests and internal callers.
    def _fetch_schema(self) -> str:
        return self._schema.fetch()

    def _can_query_structured(self, user_id: str) -> bool:
        return self._rbac.can_query(user_id)

    def _generate_cypher(self, *args, **kwargs):
        return self._cypher.generate(*args, **kwargs)

    def _plan_multistep(self, query: str, schema: str):
        return self._planner.plan(query, schema)

    def _execute_multistep(self, plan, user_context: UserContext, query: str = ""):
        return self._multistep.execute(plan, user_context, query)

    def _text2cypher(self, query: str, limit: int, user_context: Optional[UserContext] = None):
        ctx = user_context or self.user_context
        return self._text2cypher_pipeline.run(query, limit, ctx)

    def _format_response(self, query: str, items: list, strategy: str) -> dict:
        return format_response(query, items, strategy)
