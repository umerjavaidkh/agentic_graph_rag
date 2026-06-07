"""
retrieval/unstructured/retriever.py — Neo4j Graph RAG for documents.

Flow (document-agnostic — works for any ingested PDF):
1. Vector seed — embed query, find entry Section nodes
2. Full-text seed — keyword lookup on node_text_index
3. Graph expand — 1–2 hop traversal via structural + semantic edges
4. Merge, rank, return top-k chunks for LLM synthesis
"""

from __future__ import annotations

from .mixins import (
    BoxStrategyMixin,
    DocumentResolverMixin,
    GraphSeedsMixin,
    HybridRetrieveMixin,
    LexicalRetrievalMixin,
    PageStrategyMixin,
    PoliciesMixin,
    RankingMixin,
    SubsectionMixin,
    TocStrategyMixin,
)
from .query_intent import (
    is_enumeration_question,
    is_fact_lookup_question,
    is_page_question,
    is_synthesis_question,
    is_toc_question,
    is_visual_page_question,
)

__all__ = [
    "DocumentRAGRetriever",
    "ESGComplianceRetriever",
    "RAGDataRetriever",
    "is_enumeration_question",
    "is_fact_lookup_question",
    "is_page_question",
    "is_synthesis_question",
    "is_toc_question",
    "is_visual_page_question",
]


class DocumentRAGRetriever(
    HybridRetrieveMixin,
    GraphSeedsMixin,
    RankingMixin,
    LexicalRetrievalMixin,
    TocStrategyMixin,
    PageStrategyMixin,
    DocumentResolverMixin,
    BoxStrategyMixin,
    SubsectionMixin,
    PoliciesMixin,
):
    """Neo4j graph RAG over ingested document content."""


ESGComplianceRetriever = DocumentRAGRetriever
RAGDataRetriever = DocumentRAGRetriever
