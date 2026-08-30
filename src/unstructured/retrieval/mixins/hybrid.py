"""Document RAG retriever — hybrid."""
from __future__ import annotations

from typing import Any, Optional

from ....shared.auth.rbac_setup import GraphRBAC
from ....shared.auth.roles import DEFAULT_PUBLIC_CONTEXT, UserContext
from ....shared.config.settings import (
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    RETRIEVAL_FINAL_LIMIT,
)
from ....shared.neo4j.driver import get_neo4j_driver
from ....shared.config.settings import DEFAULT_LANGUAGE, HYBRID_STRATEGY
from ....shared.registries.strategy_registry import get_unstructured, list_unstructured
from ..services.formatter import access_denied_response
from ..strategies import registration as _strategy_registration  # noqa: F401  (side-effect: registers strategies)


class HybridRetrieveMixin:
    def semantic_retrieve(
        self,
        query: str,
        limit: int = RETRIEVAL_FINAL_LIMIT,
        user_context: Optional[UserContext] = None,
        document_id_hint: str = "",
        language: str = DEFAULT_LANGUAGE,
    ) -> dict[str, Any]:
        return self.hybrid_retrieve(
            query, limit=limit, user_context=user_context,
            document_id_hint=document_id_hint, language=language,
        )

    def document_candidates(self, query: str, user_context=None, limit: int = 10) -> list:
        """Plausible documents for a query that named none clearly.

        Empty when one document wins outright, and empty when nothing
        matches -- a list of documents that match nothing is worse than
        saying so. Capped at `limit`, because a picker long enough to
        scroll is a worse answer than the guess it replaces.
        """
        # Imported here, not at module scope: registration.py builds the
        # strategies that import this mixin, so a top-level import is a cycle.
        from ..strategies.registration import document_resolver

        tenant_id = getattr(user_context, "tenant_id", "") or ""
        with self.driver.session() as session:
            return document_resolver.candidates_for_query(
                session, query, tenant_id, limit=limit
            )

    def hybrid_retrieve(
        self,
        query: str,
        limit: int = RETRIEVAL_FINAL_LIMIT,
        user_context: Optional[UserContext] = None,
        document_id_hint: str = "",
        language: str = DEFAULT_LANGUAGE,
    ) -> dict[str, Any]:
        """
        Neo4j Graph RAG (all run together for normal queries):
        1. Semantic — embed query, vector search on Section embeddings
        2. Full-text — Lucene on node_text_index
        3. Graph — 1–2 hop expand from vector seeds via structural/semantic edges
        4. Lexical — phrase CONTAINS + keyword overlap (merged in ranker, not a bypass)

        Early exit (no semantic): TOC, PDF page, page visual lookups only.

        `document_id_hint`: the document the current conversation thread was
        already discussing (see conversation/thread_memory.py) — passed
        through to every strategy, used only as a fallback when the query
        has no stronger document signal of its own.
        """
        ctx = user_context or self.user_context
        tenant_id = ctx.tenant_id
        # `language` is a parameter, not a field on UserContext. tenant_id
        # lives there because it is a security boundary -- a missing one is
        # a bug and nothing may widen it. Language scopes which corpus is
        # searched, never what the user is allowed to see, and putting it
        # beside tenant_id would blur a distinction that matters more than
        # the convenience of having one object to pass.
        denied = access_denied_response(self.rbac, query, ctx)
        if denied:
            return denied

        # One universal unstructured strategy: every document question takes
        # the same path, so document scoping is decided in one place.
        #
        # Five structural fast-paths used to sit here -- TOC, page, box,
        # subsection, filing date -- each resolving its own document before
        # dispatching to its own strategy. That is why a TOC question
        # bypassed the scoping fix entirely and still answered from the wrong
        # document: the fix lived in one path and there were six. They were
        # superseded by the universal strategy, kept behind a flag during the
        # migration, and the flag has been on ever since; what remained was a
        # second retrieval stack that nothing reached and every reader of
        # this file still had to account for.
        #
        # Which hybrid runs is a setting, not a constant, so the vector-
        # scoped alternative can be exercised end to end without a code
        # change. An unknown name falls back rather than failing the query --
        # a typo in an env var should not take retrieval down.
        key = HYBRID_STRATEGY or "graph_rag_hybrid"
        if key not in list_unstructured():
            key = "graph_rag_hybrid"
        return get_unstructured(key).retrieve(
            None, query, tenant_id=tenant_id, language=language, limit=limit, ctx=ctx,
            document_id_hint=document_id_hint,
        )

    def close(self) -> None:
        """No-op: driver is process-wide; use close_neo4j_driver() on shutdown."""

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

