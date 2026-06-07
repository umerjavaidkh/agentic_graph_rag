"""Cypher scope fragments for document-scoped Neo4j queries."""
from __future__ import annotations

import re
from typing import Optional

_JOB_PREFIX_RE = re.compile(r"^[0-9a-f]{32}_", re.I)


def doc_scope_cypher(alias: str = "d") -> str:
    """Match logical document id or revision-prefixed content root id."""
    return (
        f"($doc_id IS NULL OR {alias}.logical_doc_id = $doc_id "
        f"OR {alias}.id = $doc_id "
        f"OR ($doc_id IS NOT NULL AND {alias}.id STARTS WITH $doc_id + ':'))"
    )


def node_scope_cypher(alias: str = "n") -> str:
    """
    Scope any content node (Page/Section/...) to a document without relying on
    variable-length CONTAINS paths.
    """
    return (
        f"($doc_id IS NULL "
        f"OR {alias}.logical_doc_id = $doc_id "
        f"OR {alias}.id STARTS WITH $doc_id + ':' "
        f"OR ({alias}.logical_doc_id IS NULL "
        f"AND {alias}.id STARTS WITH $doc_id + '_'))"
    )


# Backward-compat aliases
_doc_scope_cypher = doc_scope_cypher
_node_scope_cypher = node_scope_cypher


def clean_doc_title(title: Optional[str]) -> str:
    """Strip a leading 32-hex job-id prefix left on older ingests' titles."""
    t = (title or "").strip()
    return _JOB_PREFIX_RE.sub("", t) or t


_clean_doc_title = clean_doc_title
