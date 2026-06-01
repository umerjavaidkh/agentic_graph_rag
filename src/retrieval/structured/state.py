"""
retrieval/structured/state.py — State for structured graph queries.
"""
from typing import TypedDict, Optional
from ...auth.roles import UserContext


class StructuredState(TypedDict, total=False):
    question:          str
    retrieved_context: dict
    answer:            str
    sources:           list
    keywords:          list
    query_type:        str        # text2cypher | vector
    strategy:          str        # which retrieval strategy was used
    low_confidence:    bool
    cypher_generated:  Optional[str]  # for debugging/logging
    user_context:      Optional[UserContext]