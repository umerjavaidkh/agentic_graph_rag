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
from ..strategy_registry import get_structured, register_structured
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
from .strategies.multistep import MultiStepStrategy
from .strategies.text2cypher import Text2CypherStrategy


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

        # Register this instance's strategies under the same flat registry
        # the unstructured side uses (strategy_registry). StructuredRetriever
        # is a module-level singleton in graph.py in production, so this is
        # a one-time registration; re-instantiating (e.g. in tests) simply
        # re-registers with the new instance's collaborators, which is safe
        # since they all wrap the same process-wide Neo4j driver.
        text2cypher_strategy = Text2CypherStrategy(self._text2cypher_pipeline)
        multistep_strategy = MultiStepStrategy(self._planner, self._multistep)
        register_structured("text2cypher", lambda *a, **kw: text2cypher_strategy)
        register_structured("multistep", lambda *a, **kw: multistep_strategy)

    def close(self) -> None:
        """No-op: driver is process-wide; use close_neo4j_driver() on shutdown."""

    def _run_multistep(
        self,
        query: str,
        schema: str,
        user_context: UserContext,
        *,
        limit: int = 5,
        reason: str = "gate",
    ):
        """Plan and execute multistep Cypher via the registered strategy; return
        a formatted response, or None if the planner decided multistep isn't
        applicable (no plan / no steps) — an empty-but-valid plan result is
        still a real answer, not "not applicable"."""
        return get_structured("multistep").retrieve(
            query, schema=schema, ctx=user_context, limit=limit, reason=reason
        )

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
                response = self._run_multistep(query, schema, ctx, limit=limit, reason="gate")
                if response is not None:
                    return response

            text2cypher_response = get_structured("text2cypher").retrieve(
                query, schema=schema, ctx=ctx, limit=limit
            )
            chunks = text2cypher_response.get("chunks", [])

            if (
                STRUCTURED_EMPTY_MULTISTEP_FALLBACK
                and not chunks
                and not any(c.get("id") == "error" for c in chunks)
            ):
                fallback_response = self._run_multistep(
                    query, schema, ctx, limit=limit, reason="empty_text2cypher"
                )
                # Original behavior: only take the fallback if it actually
                # produced non-empty chunks — an empty-but-valid fallback
                # plan still falls through to the (also empty) text2cypher
                # response, unlike the primary path above which accepts any
                # valid-plan result including an empty one.
                if fallback_response is not None and fallback_response.get("chunks"):
                    return fallback_response

            return text2cypher_response

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
