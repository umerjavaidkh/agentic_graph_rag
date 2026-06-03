"""Neo4j graph schema constants for ingested documents."""

# Root node for an uploaded PDF/DOCX (one document = one root node).
DOCUMENT_ROOT_LABEL = "Document"
LEGACY_DOCUMENT_ROOT_LABEL = "Book"  # pre-rename data in existing Neo4j DBs

# Use in Cypher: MATCH (d:Document|Book)
DOCUMENT_ROOT_CYPHER = f"{DOCUMENT_ROOT_LABEL}|{LEGACY_DOCUMENT_ROOT_LABEL}"

# Node types that participate in full-text / vector indexes with the document tree.
INDEXED_NODE_CYPHER = f"{DOCUMENT_ROOT_CYPHER}|Chapter|Section|Page|Region|Concept"

DOCUMENT_LOGICAL_LABEL = "DocumentLogical"
DOC_REVISION_LABEL = "DocRevision"
