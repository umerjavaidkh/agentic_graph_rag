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
    DOCUMENT = "Document"
    DOCUMENT_LOGICAL = "DocumentLogical"
    DOC_REVISION = "DocRevision"
    CHAPTER = "Chapter"
    SECTION = "Section"
    PAGE    = "Page"
    REGION  = "Region"
    CONCEPT = "Concept"

    # Deprecated alias — use DOCUMENT
    BOOK = "Document"


# ─────────────────────────────────────────
# RELATIONSHIP TYPES  (two axes)
# ─────────────────────────────────────────
class RelType(str, Enum):
    # ── Axis 1: Structural (Vertical) ──────
    CONTAINS          = "CONTAINS"
    PART_OF           = "PART_OF"
    PRECEDES          = "PRECEDES"
    FOLLOWS           = "FOLLOWS"
    HAS_REVISION      = "HAS_REVISION"
    ACTIVE_REVISION   = "ACTIVE_REVISION"
    ROOT              = "ROOT"

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
    type:       str | NodeType
    title:      str                        # heading or first sentence
    text:       str                        # full text content
    order:      int                        # sequential position at this level
    page_start: int  = 0
    page_end:   int  = 0
    depth:      int  = 0                   # 0=Document root, 1=Chapter, 2=Section, 3=Page
    embedding:  Optional[list] = field(default=None, repr=False)
    entities:   list = field(default_factory=list)   # NER results
    cluster_id: Optional[int] = None                 # for SAME_CATEGORY
    visual_content: Optional[str] = None  # vision LLM: tables, charts, diagrams, shapes (Page)
    pdf_page: Optional[int] = None       # 1-based index in uploaded PDF file
    document_page: Optional[str] = None  # label printed on page: "43", "iii", "A"
    page_tags: list = field(default_factory=list)  # searchable: pdf:51, doc:43, …
    region_kind: Optional[str] = None  # table | figure
    region_tags: list = field(default_factory=list)  # table:a6, figure:3, pdf:12, …
    bbox: Optional[list] = None  # [l, t, r, b] top-left origin in parser page units
    bbox_page_size: Optional[list] = None  # [width, height] of parser page
    # Document lineage (revision snapshot ingest)
    logical_doc_id: Optional[str] = None
    revision_id: Optional[str] = None
    lifecycle_status: Optional[str] = None  # ACTIVE | EXPIRED
    content_hash: Optional[str] = None
    version_number: Optional[int] = None
    ingested_at: Optional[str] = None
    source_filename: Optional[str] = None


# ─────────────────────────────────────────
# RELATIONSHIP
# ─────────────────────────────────────────
@dataclass
class DKGEdge:
    source_id:  str
    target_id:  str
    rel_type:   str | RelType
    weight:     float = 1.0          # similarity score where relevant
    axis:       int   = 1            # 1 = structural, 2 = semantic
    properties: dict  = field(default_factory=dict)
