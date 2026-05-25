"""
structured/state.py — State for structured graph queries.
Compatible with ESGState shape so both can flow through same graph.
"""
from typing import TypedDict, Optional


class StructuredState(TypedDict):
    question:          str
    retrieved_context: dict
    answer:            str
    sources:           list
    keywords:          list
    query_type:        str        # text2cypher | vector
    strategy:          str        # which retrieval strategy was used
    low_confidence:    bool
    cypher_generated:  Optional[str]  # for debugging/logging