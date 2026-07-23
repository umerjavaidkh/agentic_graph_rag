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
        document_id_hint: str = "",
    ) -> Optional[dict[str, Any]]:
        """
        Attempt to answer `query`. Returns a terminal response dict (same
        shape produced by the formatter service — `query`, `chunks`,
        `total_available`, `mode`, `strategy`, ...) or None to mean "not
        applicable / no hits," signaling the caller to try the next
        strategy in whatever selection order it uses.

        `document_id_hint`: the document the current conversation thread
        was already discussing (see conversation/thread_memory.py) — passed
        through to DocumentResolver.resolve_document_for_query, used only
        when the query has no stronger signal of its own (an explicit
        document name always wins). A response that resolves a document
        should echo it back as `document_id`/`document_title` so the next
        turn can carry it forward in turn.
        """
        ...
