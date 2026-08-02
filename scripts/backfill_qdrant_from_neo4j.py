#!/usr/bin/env python3
"""
Backfill Qdrant from embeddings already sitting in Neo4j node properties.

Context: VECTOR_STORE_BACKEND has been "memory" (the default) for this
deployment's whole history, so every vector_store.upsert_batch() call
during ingestion wrote to an in-process store that doesn't survive across
the worker/API process boundary -- Qdrant, while running as a container,
has never actually received a single vector. This script populates it
from the one place the embeddings do reliably exist today: Neo4j's own
n.embedding property. One-time step, cheap (no re-embedding, no LLM
calls) -- read + batch-upsert only.

Run AFTER setting VECTOR_STORE_BACKEND=qdrant (or pass --force-qdrant to
construct the Qdrant client directly regardless of the current setting).

Examples:
  python scripts/backfill_qdrant_from_neo4j.py
  python scripts/backfill_qdrant_from_neo4j.py --batch-size 500
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.driver import get_neo4j_driver  # noqa: E402


def _fetch_embedded_nodes_sync(driver, batch_size: int):
    """Yields batches of (id, embedding, metadata) from every node that
    carries a non-null embedding, regardless of document/tenant -- a full
    one-time sweep, not scoped to one document."""
    with driver.session() as session:
        result = session.run(
            """
            MATCH (n) WHERE n.embedding IS NOT NULL
            RETURN n.id AS id, n.embedding AS embedding,
                   n.logical_doc_id AS logical_doc_id,
                   n.revision_id AS revision_id,
                   n.tenant_id AS tenant_id
            """
        )
        batch = []
        for record in result:
            if not record["id"] or not record["embedding"]:
                continue
            batch.append((
                record["id"],
                record["embedding"],
                {
                    "logical_doc_id": record["logical_doc_id"] or "",
                    "revision_id": record["revision_id"] or "",
                    "tenant_id": record["tenant_id"] or "default",
                },
            ))
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument(
        "--force-qdrant", action="store_true",
        help="Construct QdrantVectorStore directly, ignoring the current VECTOR_STORE_BACKEND setting.",
    )
    args = parser.parse_args()

    if args.force_qdrant:
        from src.config.settings import QDRANT_API_KEY, QDRANT_COLLECTION, QDRANT_URL, VECTOR_DIM
        from src.storage.vector.qdrant_store import QdrantVectorStore

        vector_store = QdrantVectorStore(
            url=QDRANT_URL, collection_name=QDRANT_COLLECTION,
            api_key=QDRANT_API_KEY or None, dim=VECTOR_DIM,
        )
    else:
        from src.config.settings import VECTOR_STORE_BACKEND
        from src.storage.vector.factory import get_vector_store

        if VECTOR_STORE_BACKEND != "qdrant":
            print(
                f"VECTOR_STORE_BACKEND={VECTOR_STORE_BACKEND!r}, not 'qdrant' -- "
                "this would backfill the in-memory store, which is pointless "
                "(it doesn't persist). Set VECTOR_STORE_BACKEND=qdrant first, "
                "or pass --force-qdrant to backfill Qdrant directly regardless."
            )
            sys.exit(1)
        vector_store = get_vector_store()

    driver = get_neo4j_driver()
    total = 0
    for batch in _fetch_embedded_nodes_sync(driver, args.batch_size):
        vector_store.upsert_batch(batch)
        total += len(batch)
        print(f"  upserted {total} vectors so far…")

    print(f"Done. {total} vectors backfilled into Qdrant.")


if __name__ == "__main__":
    main()
