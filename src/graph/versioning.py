"""Cypher helpers for ACTIVE revision-scoped retrieval."""

from .constants import DOCUMENT_ROOT_CYPHER

LIFECYCLE_ACTIVE = "ACTIVE"


def lifecycle_active(alias: str = "n") -> str:
    """Legacy nodes without lifecycle_status remain visible."""
    return f"coalesce({alias}.lifecycle_status, '{LIFECYCLE_ACTIVE}') = '{LIFECYCLE_ACTIVE}'"


def logical_doc_filter(alias: str = "n", param: str = "$logical_doc_id") -> str:
    return (
        f"({alias}.logical_doc_id = {param} "
        f"OR ({param} IS NOT NULL AND {alias}.id STARTS WITH {param} + ':'))"
    )
