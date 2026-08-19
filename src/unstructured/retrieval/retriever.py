"""
retrieval/unstructured/retriever.py — Neo4j Graph RAG for documents.

Flow (document-agnostic — works for any ingested PDF):
1. Vector seed — embed query, find entry Section nodes
2. Full-text seed — keyword lookup on node_text_index
3. Graph expand — 1–2 hop traversal via structural + semantic edges
4. Merge, rank, return top-k chunks for LLM synthesis
"""

from __future__ import annotations

from .mixins import HybridRetrieveMixin
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


class DocumentRAGRetriever(HybridRetrieveMixin):
    """
    Neo4j graph RAG over ingested document content.

    A thin facade: HybridRetrieveMixin's __init__ builds the driver/RBAC/
    executor state, and hybrid_retrieve() dispatches to registered
    strategies (see strategies/registration.py) rather than implementing
    retrieval logic itself.
    """


ESGComplianceRetriever = DocumentRAGRetriever
RAGDataRetriever = DocumentRAGRetriever
