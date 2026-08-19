"""Deleting a revision everywhere it exists, not only in Neo4j.

A revision's data is spread over three stores: structure in Neo4j, text and
visual content and reports in the blob store, embeddings in the vector store.
Supersede used to DETACH DELETE the Neo4j half and stop there. `delete_prefix`
(BlobStore) and `delete_by_filter` (VectorStore) were both written for exactly
this purge and never called from anywhere, so every re-ingest left its whole
previous text and embedding set behind.

What that cost, measured on the dev instance: Neo4j held 2 revisions while the
blob store held 51,406 objects across 23 of them and Qdrant 6,509 points
across 18. Retrieval stayed correct -- vector hits are resolved against Neo4j
and a hit with no node is dropped -- so nothing failed loudly; the stores just
grew without bound and every scan paid for data no query could ever return.

This module is the single place that knows the full set, so a store added
later has one obvious place to be registered.
"""
from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


def revision_blob_prefixes(
    *, tenant_id: str, logical_id: str, revision_id: str, keep_source: bool = False
) -> list[str]:
    """Every blob key prefix owned by one revision.

    Four separate roots, each defined by its own key builder:
    `Neo4jExporter._dual_write_chunk` (node text and visual content),
    `versioning.source_file_blob_key` (the uploaded file),
    `graph_snapshot.snapshot_key` and `page_report.report_key`.

    Trailing slashes matter: without one, revision `x:r1` is a prefix of
    `x:r10` and purging the first would take the second with it.

    `keep_source` spares `documents/`, which holds the original uploaded
    file. Everything else here is derived and can be rebuilt by re-parsing;
    that one cannot be rebuilt from anything, and deleting it destroys the
    only copy of what the user handed us. Supersede leaves it False -- the
    replacing revision writes its own source file over the same logical
    document -- while a reclaim pass over revisions Neo4j has forgotten
    should keep them, since there is no replacement coming.
    """
    prefixes = [
        f"{tenant_id}/{logical_id}/{revision_id}/",
        f"graph_snapshots/{tenant_id}/{logical_id}/{revision_id}/",
    ]
    if not keep_source:
        prefixes.append(f"documents/{tenant_id}/{logical_id}/{revision_id}/")
    return prefixes


def revision_blob_keys(
    *, tenant_id: str, logical_id: str, revision_id: str
) -> list[str]:
    """Individual blob keys owned by one revision (not prefixes).

    The page report is a single object named for the revision rather than a
    directory under it, so it needs an exact delete -- a prefix delete on
    `page_reports/t/l/x:r1` would also match `x:r10.json`.
    """
    return [f"page_reports/{tenant_id}/{logical_id}/{revision_id}.json"]


def purge_revision(
    *,
    tenant_id: str,
    logical_id: str,
    revision_id: str,
    blob_store=None,
    vector_store=None,
    keep_source: bool = False,
) -> dict:
    """Delete a revision's blobs and vectors. Returns what was removed.

    Neo4j is NOT touched here: supersede already deletes those nodes inside
    the same transaction that installs the new revision, and this runs after
    that transaction commits so a slow object-store call cannot hold a write
    lock open.

    Best-effort by design. A purge failure must never fail an ingest that has
    already succeeded -- the cost of a leaked blob is storage, the cost of a
    failed ingest is the document. Failures are logged and counted so a
    reconcile pass can find what was left behind.
    """
    removed = {"blobs": 0, "vectors": 0, "errors": []}

    if blob_store is None:
        from ...shared.storage.blob.factory import get_blob_store

        blob_store = get_blob_store()
    if vector_store is None:
        from ...shared.storage.vector.factory import get_vector_store

        vector_store = get_vector_store()

    for prefix in revision_blob_prefixes(
        tenant_id=tenant_id,
        logical_id=logical_id,
        revision_id=revision_id,
        keep_source=keep_source,
    ):
        try:
            removed["blobs"] += blob_store.delete_prefix(prefix) or 0
        except Exception as exc:  # noqa: BLE001 - best-effort, see docstring
            removed["errors"].append(f"blob prefix {prefix}: {exc}")
            logger.warning("Purge failed for blob prefix %s: %s", prefix, exc)

    for key in revision_blob_keys(
        tenant_id=tenant_id, logical_id=logical_id, revision_id=revision_id
    ):
        try:
            if blob_store.exists(key):
                blob_store.delete(key)
                removed["blobs"] += 1
        except Exception as exc:  # noqa: BLE001
            removed["errors"].append(f"blob key {key}: {exc}")
            logger.warning("Purge failed for blob key %s: %s", key, exc)

    try:
        vector_store.delete_by_filter({"revision_id": revision_id})
        removed["vectors"] = _count_or_none(vector_store, revision_id)
    except Exception as exc:  # noqa: BLE001
        removed["errors"].append(f"vectors {revision_id}: {exc}")
        logger.warning("Purge failed for vectors of %s: %s", revision_id, exc)

    logger.info(
        "Purged revision %s: %s blobs, vectors remaining=%s%s",
        revision_id,
        removed["blobs"],
        removed["vectors"],
        f", {len(removed['errors'])} errors" if removed["errors"] else "",
    )
    return removed


def delete_document(
    driver,
    *,
    logical_id: str,
    tenant_id: str,
    blob_store=None,
    vector_store=None,
    keep_source: bool = False,
) -> Optional[dict]:
    """Delete a whole logical document -- every revision, from every store.

    `purge_revision` clears one revision's blobs and vectors; this is the
    operator-facing whole-document delete, so it also removes the content
    nodes, the DocRevision audit nodes and the DocumentLogical node itself.
    After it returns, the id is absent from Neo4j, the blob store and the
    vector store alike -- there is no remaining trace to make the document
    reappear in a picker or a retrieval hit.

    Returns None when the id matches nothing, so the caller can 404 rather
    than reporting a successful delete of something that never existed.

    Neo4j is deleted LAST, on purpose. It is the index of which revisions
    exist, so losing it first would strand the blobs and vectors with
    nothing left to enumerate them by -- exactly the state that left 50,642
    orphaned objects behind. If this fails midway the document is still
    listed and the delete can simply be retried.
    """
    from ..graph.constants import DOC_REVISION_LABEL, DOCUMENT_LOGICAL_LABEL

    with driver.session() as session:
        revisions = [
            r["id"]
            for r in session.run(
                f"""
                MATCH (dl:{DOCUMENT_LOGICAL_LABEL} {{logical_id: $logical_id}})
                      -[:HAS_REVISION]->(rev:{DOC_REVISION_LABEL})
                WHERE coalesce(dl.tenant_id, 'default') = $tenant_id
                RETURN rev.id AS id
                """,
                logical_id=logical_id,
                tenant_id=tenant_id,
            )
            if r["id"]
        ]
        # A revision whose DocumentLogical node was lost still owns storage,
        # so fall back to matching revisions by their own logical id.
        if not revisions:
            revisions = [
                r["id"]
                for r in session.run(
                    f"""
                    MATCH (rev:{DOC_REVISION_LABEL})
                    WHERE rev.logical_id = $logical_id
                      AND coalesce(rev.tenant_id, 'default') = $tenant_id
                    RETURN rev.id AS id
                    """,
                    logical_id=logical_id,
                    tenant_id=tenant_id,
                )
                if r["id"]
            ]
        if not revisions:
            return None

    totals = {"revisions": revisions, "blobs": 0, "nodes": 0, "errors": []}
    for revision_id in revisions:
        result = purge_revision(
            tenant_id=tenant_id,
            logical_id=logical_id,
            revision_id=revision_id,
            blob_store=blob_store,
            vector_store=vector_store,
            keep_source=keep_source,
        )
        totals["blobs"] += result["blobs"]
        totals["errors"].extend(result["errors"])

    with driver.session() as session:
        totals["nodes"] = session.run(
            f"""
            MATCH (n)
            WHERE n.revision_id IN $revisions
              AND NOT n:{DOCUMENT_LOGICAL_LABEL}
            DETACH DELETE n
            RETURN count(n) AS c
            """,
            revisions=revisions,
        ).single()["c"]
        session.run(
            f"MATCH (dl:{DOCUMENT_LOGICAL_LABEL} {{logical_id: $logical_id}}) DETACH DELETE dl",
            logical_id=logical_id,
        )

    logger.info(
        "Deleted document %s: %d revision(s), %d node(s), %d blob(s)%s",
        logical_id,
        len(revisions),
        totals["nodes"],
        totals["blobs"],
        f", {len(totals['errors'])} error(s)" if totals["errors"] else "",
    )
    return totals


def _count_or_none(vector_store, revision_id: str) -> Optional[int]:
    """Points still carrying this revision_id, for verifying a purge.

    Backend-specific and entirely optional: a store with no count API simply
    reports None rather than making the purge itself conditional on one.
    """
    client = getattr(vector_store, "_client", None)
    collection = getattr(vector_store, "collection_name", None)
    if client is None or collection is None:
        return None
    try:
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        return client.count(
            collection_name=collection,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="revision_id", match=MatchValue(value=revision_id)
                    )
                ]
            ),
        ).count
    except Exception:  # noqa: BLE001
        return None
