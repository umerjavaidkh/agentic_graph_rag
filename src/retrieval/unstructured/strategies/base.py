"""base.py — structural interface all unstructured retrieval strategies conform to.

Mirrors src/document/parser_base.py's shape (a structural Protocol, no
explicit inheritance required — any class with a matching `retrieve` method
satisfies it).
"""
from __future__ import annotations

from typing import Any, Optional, Protocol

from ....auth.roles import UserContext


class UnstructuredStrategy(Protocol):
    """Any strategy registered in strategy_registry's unstructured half must implement this shape.

    `name` doubles as the registry key and the value written to
    `response["mode"]`/`response["strategy"]` — existing consumers (the
    feedback loop's retrieval_mode tracking, the eval suite's route_tool/
    agent checks) key off that string today, so keeping the registry key
    and the mode label identical means nothing downstream needs to change
    as strategies are migrated behind this interface.
    """

    name: str

    def retrieve(
        self,
        session: Any,
        query: str,
        *,
        tenant_id: str,
        limit: int,
        ctx: UserContext,
    ) -> Optional[dict[str, Any]]:
        """
        Attempt to answer `query`. Returns a terminal response dict (same
        shape produced by the formatter service — `query`, `chunks`,
        `total_available`, `mode`, `strategy`, ...) or None to mean "not
        applicable / no hits," signaling the caller to try the next
        strategy in whatever selection order it uses.
        """
        ...
