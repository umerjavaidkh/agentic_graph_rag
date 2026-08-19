"""HybridRetrieveMixin — the sole remaining retrieval entry point for DocumentRAGRetriever.

The other nine mixins that used to live here (box_strategy, document_resolver,
graph_seeds, lexical, page_strategy, policies, ranking, subsection,
toc_strategy) have been fully extracted into standalone, registry-based
strategies and services (see ../strategies/ and ../services/) as part of the
loosely-coupled retrieval refactor. HybridRetrieveMixin's `hybrid_retrieve()`
is now pure dispatch: it resolves the applicable strategy from
strategy_registry and calls it — no retrieval logic of its own.
"""
from .hybrid import HybridRetrieveMixin

__all__ = ["HybridRetrieveMixin"]
