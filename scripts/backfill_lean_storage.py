#!/usr/bin/env python3
"""
Phase 5 backfill (docs/DESIGN_unstructured_graph_v2.md): migrate nodes
ingested before the phase-3 storage split -- still carrying inline
`.text`/`.embedding` and no `blob_key_text`/`search_text`/`vector_id` --
into the lean shape, in place. Idempotent: only touches nodes matching
`n.text IS NOT NULL AND n.blob_key_text IS NULL`, so re-running is safe
and a partial run can be resumed.

Does NOT re-run the parser/GraphConstructionService (that would need the
original PDF and is what a real re-ingest already does via doc_key).
Instead: blob-writes the existing `.text` verbatim (same blob_key scheme
Neo4jExporter._dual_write_chunk already uses), derives `search_text` as a
title+budget-capped truncation of that same text (an approximation --
the real chunk-bucket-derived version a fresh ingest produces may differ
slightly, see Axis1StructuralBuilder._derive_search_text), and computes
`vector_id` via the same deterministic VectorStore.point_id_for(node.id)
already used for new ingests (no re-embedding needed -- the vector already
exists in the store from this node's original dual-write).

Usage:
    python scripts/backfill_lean_storage.py --dry-run
    python scripts/backfill_lean_storage.py --batch-size 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.neo4j.driver import get_neo4j_driver  # noqa: E402
from src.shared.storage.blob.factory import get_blob_store  # noqa: E402
from src.shared.storage.vector.factory import get_vector_store  # noqa: E402

_SEARCH_TEXT_CHAR_BUDGET = 2000  # matches Axis1StructuralBuilder._SEARCH_TEXT_CHAR_BUDGET


def _derive_search_text(title: str, text: str) -> str:
    title = title or ""
    body = (text or "")[:_SEARCH_TEXT_CHAR_BUDGET]
    return f"{title}\n\n{body}".strip() if body else title


def backfill(dry_run: bool, batch_size: int) -> None:
    driver = get_neo4j_driver()
    blob_store = get_blob_store()
    vector_store = get_vector_store()

    total_migrated = 0
    with driver.session() as session:
        while True:
            rows = session.run(
                """
                MATCH (n)
                WHERE n.text IS NOT NULL AND n.blob_key_text IS NULL
                RETURN n.id AS id, n.title AS title, n.text AS text,
                       n.embedding AS embedding, n.tenant_id AS tenant_id,
                       n.logical_doc_id AS logical_doc_id, n.revision_id AS revision_id
                LIMIT $batch_size
                """,
                batch_size=batch_size,
            )
            batch = [dict(r) for r in rows]
            if not batch:
                break

            for row in batch:
                node_id = row["id"]
                text = row.get("text") or ""
                tenant_id = row.get("tenant_id") or "default"
                logical_id = row.get("logical_doc_id") or "unknown"
                revision_id = row.get("revision_id") or "unknown"

                blob_key_text = f"{tenant_id}/{logical_id}/{revision_id}/{node_id}/text"
                search_text = _derive_search_text(row.get("title"), text)
                vector_id = (
                    vector_store.point_id_for(node_id) if row.get("embedding") else None
                )

                if dry_run:
                    print(f"[dry-run] would migrate {node_id} (blob_key={blob_key_text})")
                    continue

                blob_store.put(blob_key_text, text)
                session.run(
                    """
                    MATCH (n {id: $id})
                    SET n.blob_key_text = $blob_key_text,
                        n.search_text = $search_text,
                        n.vector_id = $vector_id
                    REMOVE n.text, n.embedding
                    """,
                    id=node_id,
                    blob_key_text=blob_key_text,
                    search_text=search_text,
                    vector_id=vector_id,
                )

            total_migrated += len(batch)
            print(f"  migrated {total_migrated} node(s) so far...")

    verb = "would migrate" if dry_run else "migrated"
    print(f"Done: {verb} {total_migrated} node(s) total.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="report what would migrate, write nothing")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    backfill(dry_run=args.dry_run, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
