"""Shared limits and graph traversal constants for document RAG."""
from __future__ import annotations

from ...config.settings import RETRIEVAL_CANDIDATE_POOL

GRAPH_REL_TYPES = (
    "SEMANTICALLY_SIMILAR",
    "SHARES_ENTITY",
    "REFERENCES",
    "ELABORATES",
    "SAME_CATEGORY",
    "FOLLOWS",
    "PRECEDES",
    "CONTAINS",
    "PART_OF",
)

TEXT_NODE_LABELS = ("Section", "Page", "Chapter", "Region")

VECTOR_SEED_LIMIT = min(RETRIEVAL_CANDIDATE_POOL, 12)
FULLTEXT_LIMIT = 8
GRAPH_1HOP_LIMIT = 24
GRAPH_2HOP_LIMIT = 16

# Backward-compat aliases (used by mixins split from monolithic retriever)
_GRAPH_REL_TYPES = GRAPH_REL_TYPES
_TEXT_NODE_LABELS = TEXT_NODE_LABELS
_VECTOR_SEED_LIMIT = VECTOR_SEED_LIMIT
_FULLTEXT_LIMIT = FULLTEXT_LIMIT
_GRAPH_1HOP_LIMIT = GRAPH_1HOP_LIMIT
_GRAPH_2HOP_LIMIT = GRAPH_2HOP_LIMIT
