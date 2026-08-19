"""multistep.py — multistep Cypher planning strategy adapter.

Thin wrapper around the existing MultiStepPlanner + MultiStepExecutor
collaborators (already standalone, constructor-injected classes). Preserves
the original _run_multistep behavior exactly: returns None when the
planner decides multistep isn't needed (no plan / no steps) — even an
EMPTY execution result (successfully planned, zero rows) is NOT "not
applicable," it's a real answer, so it's returned formatted rather than
treated as a signal to fall back to text2cypher. That fallback decision is
the orchestrator's (see StructuredRetriever.retrieve()), not this
strategy's.

`reason` is an extra optional kwarg beyond the base StructuredStrategy
Protocol — purely a telemetry label distinguishing "gate" (the primary
multistep attempt) from "empty_text2cypher" (the fallback attempt after
text2cypher returned nothing), matching the two call sites in the original
_run_multistep usage.
"""
from __future__ import annotations

from typing import Any, Optional

from ....shared.auth.roles import UserContext
from ....shared.telemetry import pipeline_step
from ..formatting.chunks import format_response
from ..multistep.executor import MultiStepExecutor
from ..multistep.planner import MultiStepPlanner


class MultiStepStrategy:
    name = "multistep"

    def __init__(self, planner: MultiStepPlanner, executor: MultiStepExecutor):
        self._planner = planner
        self._executor = executor

    def retrieve(
        self,
        query: str,
        *,
        schema: str,
        ctx: UserContext,
        limit: int,
        reason: str = "gate",
    ) -> Optional[dict[str, Any]]:
        with pipeline_step("structured.multistep.plan", reason=reason):
            plan = self._planner.plan(query, schema)
        if not plan or not plan.needs_multistep or not plan.steps:
            return None
        with pipeline_step("structured.multistep.execute", steps=len(plan.steps), reason=reason):
            chunks = self._executor.execute(plan, user_context=ctx, query=query)
        return format_response(query, chunks, strategy="multistep")
