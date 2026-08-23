#!/usr/bin/env python3
"""Embed the content nodes that were never embedded.

Audited on the 998-document corpus:

    Section  19,594 / 19,594  100%
    Chapter  10,488 / 10,488  100%
    Page        178 / 18,466    1.0%
    Region        0 / 12,818    0%   (8,068 tables, 4,750 figures)

So the dense channel cannot see 51% of the corpus, including every table and
every figure. No amount of ranking compensates for content that is not in
the index: a question about Figure 1 was answered from Section prose that
mentions the figure, never from the figure, and "how is a risk exposure
score calculated" found nothing because the formula lives in an appendix
table.

Resumable: nodes already carrying `vector_id` are skipped, so an
interrupted run continues rather than re-paying for what it did.

    python scripts/backfill_missing_embeddings.py --estimate
    python scripts/backfill_missing_embeddings.py --labels Page Region
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from neo4j import GraphDatabase  # noqa: E402

from src.shared.config.settings import EMBEDDING_MODEL  # noqa: E402
from src.shared.model_providers.factory import get_embedding_provider  # noqa: E402
from src.shared.storage.hydrator import get_hydrator  # noqa: E402
from src.shared.storage.vector.factory import get_vector_store  # noqa: E402

NEO4J_URI = "bolt://localhost:17687"
# text-embedding-3-small's window is 8191 tokens; this stays well inside it
# without a tokenizer round trip.
MAX_CHARS = 8000
USD_PER_1M_TOKENS = 0.02


def _fetch(session, label: str, limit: int, skip_ids: bool = True) -> list[dict]:
    where = "n.vector_id IS NULL AND " if skip_ids else ""
    rows = session.run(
        f"""
        MATCH (n:{label})
        WHERE {where}n.logical_doc_id IS NOT NULL
          AND coalesce(n.search_text, '') <> ''
        RETURN n.id AS id, coalesce(n.title, '') AS title,
               n.search_text AS search_text, n.blob_key_text AS blob_key_text,
               n.logical_doc_id AS doc, n.revision_id AS rev,
               coalesce(n.tenant_id, '') AS tenant
        LIMIT $limit
        """,
        limit=limit,
    )
    return [dict(r) for r in rows]


def _text_for(row: dict, hydrator) -> str:
    """What actually gets embedded.

    The title is prepended because a table's or figure's own text is often a
    bare grid or a caption fragment; the heading is what carries the topic,
    and it is what a question will resemble.
    """
    body = hydrator.hydrate(row.get("blob_key_text"), row.get("search_text") or "")
    title = (row.get("title") or "").strip()
    text = f"{title}\n{body}" if title else body
    return " ".join(text.split())[:MAX_CHARS]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs="+", default=["Page", "Region"])
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--workers", type=int, default=4,
                    help="concurrent embedding calls; the API is the bottleneck, "
                         "not Neo4j or Qdrant")
    ap.add_argument("--limit", type=int, default=1_000_000)
    ap.add_argument("--estimate", action="store_true", help="count and cost only")
    args = ap.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", "password123"))
    hydrator = get_hydrator()

    with driver.session() as s:
        pending = {
            lbl: s.run(
                f"MATCH (n:{lbl}) WHERE n.vector_id IS NULL "
                f"AND n.logical_doc_id IS NOT NULL "
                f"AND coalesce(n.search_text,'') <> '' RETURN count(n) AS c"
            ).single()["c"]
            for lbl in args.labels
        }
    total = sum(pending.values())
    for lbl, c in pending.items():
        print(f"  {lbl:<10} {c:>7,} unembedded")
    approx_tokens = total * (MAX_CHARS / 4) * 0.35  # most nodes are far shorter
    print(f"  total {total:,} nodes  ~${approx_tokens / 1e6 * USD_PER_1M_TOKENS:.2f} "
          f"at {USD_PER_1M_TOKENS}/1M tokens")
    if args.estimate or not total:
        return

    provider = get_embedding_provider()
    store = get_vector_store()
    done = 0
    t0 = time.perf_counter()

    pool = ThreadPoolExecutor(max_workers=args.workers)

    def embed(chunk):
        """Embed one chunk, waiting out the token-per-minute ceiling.

        Fixing the unlabelled `MATCH` in the write path made this fast
        enough to saturate the 1M TPM limit, so 429 is now the expected
        steady state rather than an error -- the corpus is ~22M tokens and
        the account allows 1M per minute, which sets the floor at roughly
        twenty minutes however the batching is arranged.
        """
        for attempt in range(8):
            try:
                resp = provider.embeddings(
                    model=EMBEDDING_MODEL, input=[t for _, t in chunk]
                )
                return chunk, [d.embedding for d in resp.data]
            except Exception as e:
                if "rate_limit" not in str(e).lower() or attempt == 7:
                    raise
                time.sleep(min(2 ** attempt, 30))
        raise RuntimeError("unreachable")

    for label in args.labels:
        while done < args.limit:
            # One read of many rows, then several embedding calls in flight
            # at once. Serial batches ran at 6 nodes/s -- 86 minutes for this
            # corpus -- and the wait is entirely the API round trip, so the
            # fix is concurrency rather than a bigger batch.
            with driver.session() as s:
                rows = _fetch(s, label, args.batch_size * args.workers)
            if not rows:
                break
            texts = [_text_for(r, hydrator) for r in rows]
            keep = [(r, t) for r, t in zip(rows, texts) if t.strip()]
            if not keep:
                break
            chunks = [keep[i:i + args.batch_size]
                      for i in range(0, len(keep), args.batch_size)]
            for chunk, vectors in pool.map(embed, chunks):
                store.upsert_batch([
                    (r["id"], list(v), {
                        "logical_doc_id": r["doc"],
                        "revision_id": r["rev"],
                        "tenant_id": r["tenant"],
                    })
                    for (r, _), v in zip(chunk, vectors)
                ])
                # vector_id marks a node done, so it is written only after
                # the vector is safely stored -- a crash between the two
                # costs a repeat, never a silent gap.
                with driver.session() as s:
                    # Labelled. Constraints are per label, so an unlabelled
                    # `MATCH (n) WHERE n.id = ...` scans the whole database
                    # per row -- the same mistake this corpus already paid
                    # for twice in the read path.
                    s.run(
                        f"UNWIND $rows AS row MATCH (n:{label}) "
                        "WHERE n.id = row.id SET n.vector_id = row.vid",
                        rows=[{"id": r["id"], "vid": store.point_id_for(r["id"])}
                              for r, _ in chunk],
                    ).consume()
                done += len(chunk)
            el = time.perf_counter() - t0
            print(f"  {label}: {done:,}/{total:,}  {el:.0f}s  "
                  f"({done / max(el, 1):.0f}/s)", flush=True)

    print(f"embedded {done:,} nodes in {time.perf_counter() - t0:.0f}s")


if __name__ == "__main__":
    main()
