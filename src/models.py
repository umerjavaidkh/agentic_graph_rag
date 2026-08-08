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
    # Optional: entity text (lowercased) -> NER type (ORG/PERSON/...).
    # Additive alongside `entities`, not a replacement -- every existing
    # reader of `entities` as a plain string list (retrieval ranking,
    # page/box strategies, ontology validation) keeps working unchanged.
    # Populated only by NER paths that extract typed entities; empty dict
    # means "no type info", not "no entities".
    entity_types: dict = field(default_factory=dict)
    cluster_id: Optional[int] = None                 # for SAME_CATEGORY
    summary:    Optional[str] = None                 # Chapter-level rollup (chapter_summary enrichment)
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
    # Blob-store keys (set by the exporter at write time, not by the parser)
    # when text/visual_content are dual-written to a BlobStore.
    blob_key_text: Optional[str] = None
    blob_key_visual: Optional[str] = None
    # Lean-storage plumbing (docs/DESIGN_unstructured_graph_v2.md phase 3):
    # search_text is a chunk-bounded property Neo4j keeps for Lucene/IDF
    # lexical matching once `text` itself is no longer written there --
    # set by the parser/graph-construction layer, same as `text`.
    search_text: Optional[str] = None
    # vector_id is set by the exporter at write time (same timing as
    # blob_key_text above) -- a persisted copy of the deterministic Qdrant
    # point id, so callers don't have to re-derive it from node.id.
    vector_id: Optional[str] = None
    # Multi-tenancy: stamped by apply_revision_to_graph, not the parser itself.
    tenant_id: Optional[str] = None


# ─────────────────────────────────────────
# EDGE CONFIDENCE / PROVENANCE
# ─────────────────────────────────────────
class EdgeConfidenceTier(str, Enum):
    """How this edge was derived, so a query can distinguish directly-observed
    facts from derived/uncertain ones (borrowed from Graphify's model)."""
    EXTRACTED = "EXTRACTED"  # deterministic: directly found in the document/graph structure
    INFERRED  = "INFERRED"   # derived from a real numeric signal (similarity, LLM judgment)
    AMBIGUOUS = "AMBIGUOUS"  # weak signal, no strong per-pair confidence available


# ─────────────────────────────────────────
# RELATIONSHIP
# ─────────────────────────────────────────
@dataclass
class DKGEdge:
    source_id:        str
    target_id:        str
    rel_type:         str | RelType
    weight:           float = 1.0          # similarity score where relevant
    axis:             int   = 1            # 1 = structural, 2 = semantic
    properties:       dict  = field(default_factory=dict)
    confidence:       float = 1.0
    confidence_tier:  str | EdgeConfidenceTier = EdgeConfidenceTier.EXTRACTED
    # Multi-tenancy: stamped by apply_revision_to_graph, not the parser itself.
    tenant_id:        Optional[str] = None
