#!/usr/bin/env python3
"""
Phase 1 backfill (docs/DESIGN_language_independence.md): give every
document ingested before language scoping existed the default language,
and hang it under its `:Language` parent.

Backfill, not a null-guard. The design is explicit about why, and the
repository has already paid for the alternative: a null-guard in a scope
predicate cost this project a 611,815-node scan on a single query. Once
this has run, `n.language = $language` is an index-backed equality with
nothing to coalesce.

Correct by the assignment rule rather than by assumption: English is what
a document is when no other profile claims it, so every pre-existing
document IS English by definition -- there is nothing to detect and no
document text has to be re-read. A document that should have been Arabic
would have to have been ingested before Arabic was enabled, which cannot
have happened. This is why enabling a second language does not mean
re-ingesting the first one's corpus.

Touches the DOCUMENT TREE only. The structured business graph shares this
database and is the larger part of it; it has no language dimension and
must not be given one.

Idempotent: only touches nodes matching `n.language IS NULL`, so a
partial run resumes and re-running is free.

Usage:
    python scripts/backfill_language.py --dry-run
    python scripts/backfill_language.py --batch-size 10000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.config.settings import DEFAULT_LANGUAGE  # noqa: E402
from src.shared.neo4j.driver import get_neo4j_driver  # noqa: E402
from src.shared.storage.vector.factory import get_vector_store  # noqa: E402
from src.unstructured.graph.constants import (  # noqa: E402
    DOC_REVISION_LABEL,
    DOCUMENT_LOGICAL_LABEL,
    INDEXED_NODE_CYPHER,
)

# The document tree, and nothing else. Derived from the constants the
# ingestion path already uses rather than listed again here, so a new
# document label is backfilled without anyone remembering this file.
#
# Scoping this matters more than it looks. This database holds 611,814
# nodes and 547,416 of them are the structured business graph -- Orders,
# Payments, Customers. An unscoped `MATCH (n) WHERE n.language IS NULL`
# stamps a language on every one of them: 8.5x the writes, and every one
# of them a lie. The business graph has no language dimension; its labels
# and properties are schema, not prose. Leaving it unstamped also makes
# the boundary physical -- splice language_filter() into a structured
# query by mistake and it matches nothing loudly, instead of quietly
# working in English and failing in Arabic.
DOCUMENT_LABELS = ":" + "|".join(
    sorted(set(INDEXED_NODE_CYPHER.split("|")) | {DOCUMENT_LOGICAL_LABEL, DOC_REVISION_LABEL})
)


def _count_unstamped(session) -> int:
    row = session.run(
        f"MATCH (n{DOCUMENT_LABELS}) WHERE n.language IS NULL RETURN count(n) AS n"
    ).single()
    return (row or {}).get("n", 0)


def _stamp_batch(session, language: str, batch_size: int) -> int:
    """One batch of nodes, so a corpus-sized write is not one transaction.

    A single `MATCH (n) WHERE n.language IS NULL SET n.language = ...` over
    600k nodes builds one transaction the heap has to hold entire. Batching
    keeps each write bounded and lets the run be interrupted without losing
    what it has already done.
    """
    row = session.run(
        f"""
        MATCH (n{DOCUMENT_LABELS}) WHERE n.language IS NULL
        WITH n LIMIT $batch_size
        SET n.language = $language
        RETURN count(n) AS stamped
        """,
        batch_size=batch_size,
        language=language,
    ).single()
    return (row or {}).get("stamped", 0)


def _languages_present(session) -> set:
    """Languages already stamped on documents, so a second run migrates
    the vectors of documents ingested since the first one."""
    rows = session.run(
        "MATCH (dl:DocumentLogical) WHERE dl.language IS NOT NULL "
        "RETURN DISTINCT dl.language AS code"
    )
    return {r["code"] for r in rows if r["code"]}


def _attach_language_parent(session, language: str) -> int:
    """The `(:Language)-[:HAS_DOCUMENT]->(:DocumentLogical)` edge.

    One edge per logical document -- 998 of them today, +0.1% on the
    relationship count. It deliberately stops at DocumentLogical: attaching
    Language to every node instead would put 611,814 relationships on a
    single supernode.
    """
    row = session.run(
        """
        MERGE (lang:Language {code: $language})
        WITH lang
        MATCH (dl:DocumentLogical)
        WHERE NOT (:Language)-[:HAS_DOCUMENT]->(dl)
        MERGE (lang)-[:HAS_DOCUMENT]->(dl)
        RETURN count(dl) AS attached
        """,
        language=language,
    ).single()
    return (row or {}).get("attached", 0)


def _backfill_vector_payloads(session, language: str) -> int:
    """Put the language on the vector points too, not only on the nodes.

    candidate_docs filters the ANN query itself, so a point with no
    `language` in its payload is invisible to a language-scoped probe.
    Backfilling Neo4j alone would therefore leave retrieval finding
    nothing at all -- worse than before the scoping existed.

    Grouped by language rather than by document: one call naming every
    document of a language, instead of 998 calls naming one each.
    """
    rows = session.run(
        """
        MATCH (dl:DocumentLogical)
        WHERE coalesce(dl.language, $language) = $language
        RETURN collect(dl.logical_id) AS ids
        """,
        language=language,
    ).single()
    doc_ids = (rows or {}).get("ids") or []
    if not doc_ids:
        return 0
    get_vector_store().set_payload_by_filter(
        {"logical_doc_id": doc_ids}, {"language": language}
    )
    return len(doc_ids)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--language", default=DEFAULT_LANGUAGE)
    parser.add_argument(
        "--skip-vectors",
        action="store_true",
        help="Neo4j only. The vector payloads must be migrated too before "
             "language-scoped retrieval returns anything.",
    )
    args = parser.parse_args()

    driver = get_neo4j_driver()
    with driver.session() as session:
        remaining = _count_unstamped(session)
        print(f"nodes with no language: {remaining}")
        if args.dry_run:
            print(f"dry run: would stamp them '{args.language}' and attach the parent")
            return 0

        stamped = 0
        while True:
            done = _stamp_batch(session, args.language, args.batch_size)
            if not done:
                break
            stamped += done
            print(f"  stamped {stamped}/{remaining}", flush=True)

        attached = _attach_language_parent(session, args.language)
        print(f"stamped {stamped} nodes '{args.language}'")
        print(f"attached {attached} documents to (:Language {{code:'{args.language}'}})")

        if not args.skip_vectors:
            for code in sorted({args.language} | _languages_present(session)):
                docs = _backfill_vector_payloads(session, code)
                print(f"vector payloads: tagged points of {docs} documents '{code}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
