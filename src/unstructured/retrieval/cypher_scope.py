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


def doc_scope_cypher_multi(alias: str = "d", param: str = "$doc_ids") -> str:
    """Scope a document root to any of several documents.

    The single-id form has one failure mode that costs more than being
    wrong: when nothing resolves confidently it degrades to unscoped, and
    an unscoped lexical pass runs a 6-hop membership check against every
    content node in the corpus -- measured at 20.13s over 4,517 candidates
    against 0.42s over 72 when scoped to one document.

    "I am not sure which of these three" is far better information than
    "no idea", and it is what the resolver already computes before
    discarding it. Scoping to a handful of candidates keeps the query fast
    and keeps the right document in range.

    A list is index-friendly: `IN` becomes a seek per value against the
    same composite index a single equality would use.
    """
    return (
        f"({param} IS NULL OR size({param}) = 0 "
        f"OR {alias}.logical_doc_id IN {param} "
        f"OR {alias}.id IN {param} "
        f"OR any(x IN {param} WHERE {alias}.id STARTS WITH x + ':'))"
    )


def node_scope_cypher_multi(alias: str = "n", param: str = "$doc_ids") -> str:
    """Scope any content node to any of several documents.

    Same shape as node_scope_cypher, and for the same reason it exists:
    membership is decided by an indexed property rather than by walking
    CONTAINS relationships per node.
    """
    return (
        f"({param} IS NULL OR size({param}) = 0 "
        f"OR {alias}.logical_doc_id IN {param} "
        f"OR any(x IN {param} WHERE {alias}.id STARTS WITH x + ':') "
        f"OR ({alias}.logical_doc_id IS NULL "
        f"AND any(x IN {param} WHERE {alias}.id STARTS WITH x + '_')))"
    )


def as_doc_id_list(document_id) -> list[str] | None:
    """Normalise a caller's scope to a list, or None for unscoped.

    Accepts what call sites actually hold: a single id, a list of them, or
    the empty string the resolver returns when it found nothing.
    """
    if document_id is None or document_id == "":
        return None
    if isinstance(document_id, str):
        return [document_id]
    ids = [d for d in document_id if d]
    return ids or None


# Backward-compat aliases
_doc_scope_cypher = doc_scope_cypher
_node_scope_cypher = node_scope_cypher


def clean_doc_title(title: Optional[str]) -> str:
    """Strip a leading 32-hex job-id prefix left on older ingests' titles."""
    t = (title or "").strip()
    return _JOB_PREFIX_RE.sub("", t) or t


_clean_doc_title = clean_doc_title
