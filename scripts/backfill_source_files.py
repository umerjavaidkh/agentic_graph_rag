"""
scripts/backfill_source_files.py — persist original source files for
documents ingested before source-file persistence existed.

For each ACTIVE revision in Neo4j, locates the matching local PDF under
sample_data_to_test/ (by ticker folder + title, or the flat unstructured/
root for non-SEC-EDGAR demo docs) and writes it to blob storage under the
same key ingestion now writes going forward (source_file_blob_key), so the
GET /documents/{id}/file viewer works for documents ingested before this
feature existed. Idempotent — skips any key already present.

Run inside the app container (env already points at Neo4j/MinIO):
    docker exec graphrag-app python scripts/backfill_source_files.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.shared.config.settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER
from src.unstructured.document.versioning import source_file_blob_key
from src.unstructured.graph.constants import DOC_REVISION_LABEL, DOCUMENT_LOGICAL_LABEL
from src.shared.storage.blob.factory import get_blob_store

SAMPLE_DATA_ROOT = Path(__file__).resolve().parents[1] / "sample_data_to_test"


def _find_local_source(title: str) -> Path | None:
    """Best-effort local file lookup: SEC-EDGAR ticker folder first, then
    the flat unstructured/ root for the small non-SEC demo docs."""
    ticker = title.split("_", 1)[0]
    candidate = SAMPLE_DATA_ROOT / "unstructured" / "sec_edgar" / ticker / f"{title}.pdf"
    if candidate.exists():
        return candidate
    candidate = SAMPLE_DATA_ROOT / "unstructured" / f"{title}.pdf"
    if candidate.exists():
        return candidate
    return None


def main() -> None:
    from neo4j import GraphDatabase

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    blob_store = get_blob_store()

    with driver.session() as session:
        rows = list(
            session.run(
                f"""
                MATCH (dl:{DOCUMENT_LOGICAL_LABEL})-[:ACTIVE_REVISION]->(rev:{DOC_REVISION_LABEL})
                RETURN dl.logical_id AS logical_id, rev.revision_id AS revision_id,
                       rev.tenant_id AS tenant_id, rev.source_filename AS source_filename,
                       coalesce(dl.title, dl.logical_id) AS title
                ORDER BY logical_id
                """
            )
        )

    print(f"Found {len(rows)} ACTIVE revision(s).\n")
    done = skipped = missing = 0
    for row in rows:
        logical_id = row["logical_id"]
        revision_id = row["revision_id"]
        tenant_id = row["tenant_id"] or "default"
        source_filename = row["source_filename"] or ""
        title = row["title"] or logical_id

        key = source_file_blob_key(
            tenant_id=tenant_id,
            logical_id=logical_id,
            revision_id=revision_id,
            source_filename=source_filename,
        )
        if blob_store.exists(key):
            print(f"  SKIP  (already backfilled) {logical_id} -> {key}")
            skipped += 1
            continue

        local_path = _find_local_source(title)
        if local_path is None:
            print(f"  MISS  (no local file found) {logical_id} (title={title!r})")
            missing += 1
            continue

        import mimetypes

        content_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"
        blob_store.put_bytes(key, local_path.read_bytes(), content_type=content_type)
        print(f"  DONE  {logical_id} <- {local_path.relative_to(SAMPLE_DATA_ROOT.parent)} -> {key}")
        done += 1

    print(f"\nBackfilled {done}, skipped {skipped} (already present), missing {missing}.")
    driver.close()


if __name__ == "__main__":
    main()
