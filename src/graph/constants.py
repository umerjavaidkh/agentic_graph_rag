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


# Labels that are never business entities: the document tree, plus RBAC and
# plumbing nodes. Excluded wherever the question is "what does the STRUCTURED
# graph hold" -- router summaries, and the metric candidates offered when a
# question is ambiguous. Without this, a graph holding both documents and
# business data offers `Chapter.order` as a candidate meaning of "order".
NON_BUSINESS_LABELS = frozenset(
    set(INDEXED_NODE_CYPHER.split("|"))
    | {DOCUMENT_LOGICAL_LABEL, DOC_REVISION_LABEL}
    | {"User", "Role", "Tenant", "Chunk"}
)


# Field documentation, stored in the graph rather than in code so it travels
# with whatever dataset is loaded. A loader (or a human) writes one node per
# label/property/relationship it wants to explain; SchemaProvider folds them
# into the schema the Cypher generator sees. Absent nodes change nothing.
SCHEMA_DOC_LABEL = "SchemaDoc"
