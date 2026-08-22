"""Cypher helpers for ACTIVE revision-scoped retrieval."""

from ...unstructured.graph.constants import DOCUMENT_ROOT_CYPHER

LIFECYCLE_ACTIVE = "ACTIVE"


def lifecycle_active(alias: str = "n") -> str:
    """Active-node predicate, written so an index can serve it.

    This was `coalesce(lifecycle_status, 'ACTIVE') = 'ACTIVE'`, which is
    correct and unusable: a function call around a property is not
    seekable, so the composite indexes on (logical_doc_id,
    lifecycle_status) could never be used. Every document-scoped read fell
    back to a scan of the whole database -- which also holds the structured
    business graph -- and measured 1,110,089 db hits against 53 for the
    same query with a plain equality. Some questions took three minutes.

    The coalesce existed for legacy nodes written before lifecycle_status,
    and `OR ... IS NULL` is no better: it defeats the index just as
    thoroughly. Those nodes are backfilled at schema setup instead (see
    Neo4jExporter._ensure_indexes), so the property is always present and a
    direct equality is both correct and fast.
    """
    return f"{alias}.lifecycle_status = '{LIFECYCLE_ACTIVE}'"


def logical_doc_filter(alias: str = "n", param: str = "$logical_doc_id") -> str:
    return (
        f"({alias}.logical_doc_id = {param} "
        f"OR ({param} IS NOT NULL AND {alias}.id STARTS WITH {param} + ':'))"
    )
