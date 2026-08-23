"""Cypher scope fragments for document-scoped Neo4j queries."""
from __future__ import annotations

import re
from typing import Optional

from ...shared.neo4j.versioning import LIFECYCLE_ACTIVE

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


def content_match_cypher(alias: str = "n", labels: "tuple[str, ...]" = ()) -> str:
    """`MATCH` pattern naming every text-bearing label explicitly.

    Neo4j can only use an index when the query names the label. The shape
    this replaces --

        MATCH (n) WHERE any(l IN labels(n) WHERE l IN $labels)

    -- is an AllNodesScan followed by a per-node label test, and no index
    can serve it, however many exist. That scanned all 611,814 nodes in a
    corpus of 1,001 documents, most of them the structured business graph
    that shares this database and can never match.
    """
    from ..retrieval.constants import TEXT_NODE_LABELS

    names = labels or TEXT_NODE_LABELS
    return f"({alias}:" + "|".join(names) + ")"


def content_scope_where(alias: str = "n", param: str = "$doc_id") -> str:
    """Indexed document scope for a text-bearing content node.

    Replaces `EXISTS { MATCH (d)-[:CONTAINS*0..6]->(n) }`, which re-walks a
    six-hop path *per candidate node*. Every content node already carries
    `logical_doc_id`, so membership is a property equality the composite
    (logical_doc_id, lifecycle_status) index answers directly.

    Verified equivalent, not assumed: across 40 documents both forms
    selected the same 4,001 nodes -- 0 lost, 0 gained, 0 documents
    differing -- while database accesses for one scoped keyword lookup fell
    from 2,533,234 to 286.

    `$doc_id` NULL still means unscoped, matching the predicate this
    replaces.
    """
    return (
        f"({param} IS NULL OR {alias}.logical_doc_id = {param}) "
        f"AND {alias}.lifecycle_status = '{LIFECYCLE_ACTIVE}'"
    )


def content_scope_where_multi(
    alias: str = "n", param: str = "$doc_ids", *, scoped: bool = False
) -> str:
    """Indexed document scope for a content node, across several documents.

    `scoped` must say whether the caller actually holds document ids, because
    the convenient null-guard form is what stops the index being used:

        ($doc_ids IS NULL OR size($doc_ids) = 0 OR n.logical_doc_id IN $doc_ids)

    Neo4j cannot seek through that disjunction -- the predicate might be
    satisfied by every node, so the planner falls back to AllNodesScan and
    then filters. Measured on the phrase query: 611,815 rows scanned with the
    guard, against a seek per document id without it. The guard looked free
    and cost more than the traversal it replaced.

    So the shape is chosen up front: a seekable equality when there are ids,
    and an honest "no filter" when there are none.
    """
    lifecycle = f"{alias}.lifecycle_status = '{LIFECYCLE_ACTIVE}'"
    if scoped:
        return f"{alias}.logical_doc_id IN {param} AND {lifecycle}"
    return (
        f"({param} IS NULL OR size({param}) = 0 "
        f"OR {alias}.logical_doc_id IN {param}) "
        f"AND {lifecycle}"
    )
