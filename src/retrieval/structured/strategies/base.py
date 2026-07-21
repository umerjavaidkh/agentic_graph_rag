"""base.py — structural interface all structured retrieval strategies conform to.

Mirrors src/retrieval/unstructured/strategies/base.py's shape and contract:
`retrieve()` returns a complete, already-formatted response dict (via the
existing `formatting.chunks.format_response`) or None to mean "not
applicable," not a raw chunk list — each strategy owns building its own
terminal response, same as the unstructured side.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol

from ....auth.roles import UserContext


class StructuredStrategy(Protocol):
    """Any strategy registered in strategy_registry's structured half must implement this shape."""

    name: str  # registry key AND response["strategy"] value, e.g. "text2cypher", "multistep"

    def retrieve(
        self,
        query: str,
        *,
        schema: str,
        ctx: UserContext,
        limit: int,
    ) -> Optional[dict[str, Any]]:
        """
        Attempt to answer `query` against the structured graph. Returns a
        terminal response dict (via format_response) or None to mean "not
        applicable / no rows," signaling the caller to try a fallback.
        """
        ...
