"""
models.py — Internal node/edge dataclasses.
These are the in-memory representations before export to Neo4j.
"""
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ─────────────────────────────────────────
# NODE TYPES
# ─────────────────────────────────────────
class NodeType(str, Enum):
    BOOK    = "Book"
    CHAPTER = "Chapter"
    SECTION = "Section"
    PAGE    = "Page"
    CONCEPT = "Concept"


# ─────────────────────────────────────────
# RELATIONSHIP TYPES  (two axes)
# ─────────────────────────────────────────
class RelType(str, Enum):
    # ── Axis 1: Structural (Vertical) ──────
    CONTAINS          = "CONTAINS"
    PART_OF           = "PART_OF"
    PRECEDES          = "PRECEDES"
    FOLLOWS           = "FOLLOWS"

    # ── Axis 2: Semantic (Horizontal) ──────
    SEMANTICALLY_SIMILAR = "SEMANTICALLY_SIMILAR"
    REFERENCES           = "REFERENCES"
    SHARES_ENTITY        = "SHARES_ENTITY"
    CONTRADICTS          = "CONTRADICTS"
    ELABORATES           = "ELABORATES"
    PREREQUISITE_OF      = "PREREQUISITE_OF"
    SAME_CATEGORY        = "SAME_CATEGORY"

    # ── Concept bridge ──────────────────────
    MENTIONS = "MENTIONS"


# ─────────────────────────────────────────
# NODE
# ─────────────────────────────────────────
@dataclass
class DKGNode:
    id:         str                        # unique: "chapter_1", "page_12", etc.
    type:       NodeType
    title:      str                        # heading or first sentence
    text:       str                        # full text content
    order:      int                        # sequential position at this level
    page_start: int  = 0
    page_end:   int  = 0
    depth:      int  = 0                   # 0=Book, 1=Chapter, 2=Section, 3=Page
    embedding:  Optional[list] = field(default=None, repr=False)
    entities:   list = field(default_factory=list)   # NER results
    cluster_id: Optional[int] = None                 # for SAME_CATEGORY


# ─────────────────────────────────────────
# RELATIONSHIP
# ─────────────────────────────────────────
@dataclass
class DKGEdge:
    source_id:  str
    target_id:  str
    rel_type:   RelType
    weight:     float = 1.0          # similarity score where relevant
    axis:       int   = 1            # 1 = structural, 2 = semantic
    properties: dict  = field(default_factory=dict)
