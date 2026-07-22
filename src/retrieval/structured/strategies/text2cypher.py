"""text2cypher.py — single-shot Text-to-Cypher strategy adapter.

Thin wrapper around the existing Text2CypherPipeline collaborator (already
a standalone, constructor-injected class — see cypher/pipeline.py) so it
can be looked up via strategy_registry alongside MultiStepStrategy, rather
than StructuredRetriever calling self._text2cypher_pipeline.run(...)
directly. No retrieval logic moves here — this only adds the registry
dispatch indirection.

Unlike MultiStepStrategy, this one always returns a formatted response
(never None) — text2cypher has no "not applicable" concept, it always runs
and returns something (possibly empty). Whether to try multistep as a
fallback when the result is empty is an orchestrator-level decision (see
StructuredRetriever.retrieve()), not something this strategy owns.
"""
from __future__ import annotations

from typing import Any, Optional

from ....auth.roles import UserContext
from ....telemetry import pipeline_step
from ..cypher.pipeline import Text2CypherPipeline
from ..formatting.chunks import format_response


class Text2CypherStrategy:
    name = "text2cypher"

    def __init__(self, pipeline: Text2CypherPipeline):
        self._pipeline = pipeline

    def retrieve(
        self,
        query: str,
        *,
        schema: str,
        ctx: UserContext,
        limit: int,
    ) -> Optional[dict[str, Any]]:
        with pipeline_step("structured.text2cypher"):
            chunks = self._pipeline.run(query, limit, user_context=ctx)
        return format_response(query, chunks, strategy="text2cypher")
