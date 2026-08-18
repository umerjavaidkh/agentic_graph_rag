#!/usr/bin/env python
"""Delete blobs and vectors belonging to revisions Neo4j no longer has.

Supersede purges all three stores now (`document.purge.purge_revision`), but
nothing purged the blob or vector stores before that, and a Neo4j wipe still
leaves both untouched. Everything ingested up to that point is still sitting
there: on the dev instance Neo4j held 2 revisions while the blob store held
51,406 objects across 23 and Qdrant 6,509 points across 18.

Nothing reads those leftovers -- a vector hit whose node is gone is dropped
when the id is resolved against Neo4j -- so this reclaims space rather than
fixing an answer. Neo4j is the authority on what exists; anything else is
garbage.

Dry-run by default. Pass --apply to delete.

    python scripts/gc_orphaned_storage.py            # report only
    python scripts/gc_orphaned_storage.py --apply
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.document.purge import purge_revision  # noqa: E402
from src.graph.constants import DOC_REVISION_LABEL  # noqa: E402
from src.graph.driver import get_neo4j_driver  # noqa: E402
from src.storage.blob.factory import get_blob_store  # noqa: E402
from src.storage.vector.factory import get_vector_store  # noqa: E402


def live_revisions() -> set[str]:
    """Revision ids Neo4j still knows about, in any lifecycle state.

    EXPIRED revisions are kept as live for this purpose: the DocRevision node
    survives supersede on purpose, as the audit trail of what was replaced,
    and its content is purged by that same supersede. Treating it as garbage
    here would be harmless but would misreport what this script reclaimed.
    """
    with get_neo4j_driver().session() as session:
        return {
            r["id"]
            for r in session.run(
                f"MATCH (n:{DOC_REVISION_LABEL}) RETURN n.id AS id"
            )
            if r["id"]
        }


def blob_revisions(blob_store) -> dict[tuple[str, str, str], int]:
    """(tenant, logical_id, revision_id) -> object count, from the keys.

    Read back from the key layout rather than from any index, because the
    orphans this reclaims are precisely the ones nothing indexes.
    """
    found: dict[tuple[str, str, str], int] = collections.Counter()
    for obj in blob_store._client.list_objects(blob_store.bucket, recursive=True):
        parts = obj.object_name.split("/")
        # `t/l/r/...` (node text) or `root/t/l/r...` (source file, snapshots,
        # page reports) -- page reports end in `<revision>.json`, not a
        # directory, so the suffix is stripped to recover the revision id.
        if parts[0] in ("documents", "graph_snapshots", "page_reports"):
            if len(parts) < 4:
                continue
            tenant, logical, rev = parts[1], parts[2], parts[3]
            if rev.endswith(".json"):
                rev = rev[: -len(".json")]
        else:
            if len(parts) < 4:
                continue
            tenant, logical, rev = parts[0], parts[1], parts[2]
        found[(tenant, logical, rev)] += 1
    return found


def vector_revisions(vector_store) -> dict[str, int]:
    """revision_id -> point count, by scrolling the whole collection."""
    counts: collections.Counter = collections.Counter()
    offset = None
    while True:
        points, offset = vector_store._client.scroll(
            collection_name=vector_store.collection_name,
            limit=1000,
            offset=offset,
            with_payload=True,
        )
        for point in points:
            counts[(point.payload or {}).get("revision_id")] += 1
        if offset is None:
            return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="actually delete")
    parser.add_argument(
        "--include-sources",
        action="store_true",
        help="also delete the original uploaded files under documents/ "
        "(they cannot be rebuilt -- everything else here can, by re-parsing)",
    )
    args = parser.parse_args()

    live = live_revisions()
    blob_store, vector_store = get_blob_store(), get_vector_store()
    blobs, vectors = blob_revisions(blob_store), vector_revisions(vector_store)

    print(f"Neo4j knows {len(live)} revision(s): {', '.join(sorted(live)) or '(none)'}\n")

    orphans = {
        (t, l, r): n for (t, l, r), n in blobs.items() if r not in live
    }
    orphan_vectors = {
        r: n for r, n in vectors.items() if r and r not in live
    }
    unattributed = vectors.get(None, 0)

    print("%-46s %8s %8s" % ("orphaned revision", "blobs", "vectors"))
    keys = sorted({(t, l, r) for t, l, r in orphans} |
                  {("?", "?", r) for r in orphan_vectors if
                   not any(rr == r for _, _, rr in orphans)})
    for tenant, logical, rev in keys:
        print("%-46s %8s %8s" % (
            rev[:46], orphans.get((tenant, logical, rev), 0), orphan_vectors.get(rev, 0)))
    print("%-46s %8d %8d" % (
        "TOTAL", sum(orphans.values()), sum(orphan_vectors.values())))
    if unattributed:
        print(f"\n{unattributed} vector point(s) carry no revision_id -- left alone "
              "(nothing identifies which revision they belong to).")
    if not args.include_sources:
        print("\nOriginal uploaded files under documents/ are KEPT (they cannot be "
              "rebuilt). Pass --include-sources to delete those too.")
    print(f"\nKept: {sum(n for (_, _, r), n in blobs.items() if r in live)} blob(s), "
          f"{sum(n for r, n in vectors.items() if r in live)} vector(s).")

    if not orphans and not orphan_vectors:
        print("\nNothing to reclaim.")
        return 0
    if not args.apply:
        print("\nDry run. Re-run with --apply to delete.")
        return 0

    print()
    for (tenant, logical, rev) in sorted({(t, l, r) for t, l, r in orphans}):
        result = purge_revision(
            tenant_id=tenant,
            logical_id=logical,
            revision_id=rev,
            blob_store=blob_store,
            vector_store=vector_store,
            keep_source=not args.include_sources,
        )
        print("purged %-40s blobs=%-6s errors=%s"
              % (rev[:40], result["blobs"], len(result["errors"])))
    # Vectors for a revision whose blobs are already gone leave no (tenant,
    # logical) to derive, so they are cleared by revision_id alone.
    for rev in sorted(orphan_vectors):
        if not any(r == rev for _, _, r in orphans):
            vector_store.delete_by_filter({"revision_id": rev})
            print("purged %-40s vectors=%s" % (rev[:40], orphan_vectors[rev]))
    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
