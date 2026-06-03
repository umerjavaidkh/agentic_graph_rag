#!/usr/bin/env bash
# Verify document versioning end-to-end (API + Neo4j).
# Usage:
#   ./scripts/verify_document_versioning.sh /path/to/test.pdf
# Env (optional): API_BASE=http://localhost:8000  NEO4J_URI=bolt://localhost:17687
set -euo pipefail

PDF="${1:-}"
API_BASE="${API_BASE:-http://localhost:8000}"
DOC_KEY="${DOC_KEY:-versioning-smoke-test}"
NEO4J_URI="${NEO4J_URI:-bolt://localhost:17687}"
NEO4J_USER="${NEO4J_USER:-neo4j}"
NEO4J_PASSWORD="${NEO4J_PASSWORD:-password123}"

if [[ -z "$PDF" || ! -f "$PDF" ]]; then
  echo "Usage: $0 /path/to/document.pdf"
  exit 1
fi

echo "== 1) API health =="
curl -sf "$API_BASE/health" | head -c 200
echo ""

poll_job() {
  local jid="$1"
  local i=0
  while [[ $i -lt 120 ]]; do
    resp=$(curl -sf "$API_BASE/ingest/jobs/$jid")
    status=$(echo "$resp" | python3 -c "import sys,json; print(json.load(sys.stdin)['status'])")
    if [[ "$status" == "completed" || "$status" == "failed" ]]; then
      echo "$resp" | python3 -m json.tool
      return 0
    fi
    sleep 5
    i=$((i + 1))
  done
  echo "Timed out waiting for job $jid"
  exit 1
}

echo ""
echo "== 2) First ingest (expect version_number=1, skipped_duplicate=false) =="
J1=$(curl -sf -X POST "$API_BASE/ingest/unstructured" \
  -F "file=@${PDF}" \
  -F "doc_key=${DOC_KEY}" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "job_id=$J1"
poll_job "$J1"

echo ""
echo "== 3) Re-ingest same file (expect skipped_duplicate=true, fast finish) =="
J2=$(curl -sf -X POST "$API_BASE/ingest/unstructured" \
  -F "file=@${PDF}" \
  -F "doc_key=${DOC_KEY}" | python3 -c "import sys,json; print(json.load(sys.stdin)['job_id'])")
echo "job_id=$J2"
poll_job "$J2"

echo ""
echo "== 4) Neo4j graph checks =="
export NEO4J_URI NEO4J_USER NEO4J_PASSWORD DOC_KEY
python3 <<'PY'
import os
import sys

try:
    from neo4j import GraphDatabase
except ImportError:
    print("Install neo4j driver: pip install neo4j", file=sys.stderr)
    sys.exit(1)

doc_key = os.environ["DOC_KEY"]
uri = os.environ["NEO4J_URI"]
auth = (os.environ["NEO4J_USER"], os.environ["NEO4J_PASSWORD"])

driver = GraphDatabase.driver(uri, auth=auth)
with driver.session() as s:
    row = s.run(
        """
        MATCH (dl:DocumentLogical {logical_id: $lid})-[:ACTIVE_REVISION]->(rev:DocRevision)
        RETURN dl.logical_id AS logical_id, rev.id AS revision_id,
               rev.version_number AS version, rev.status AS status,
               rev.content_hash AS hash
        """,
        lid=doc_key,
    ).single()
    if not row:
        print(f"FAIL: no DocumentLogical/ACTIVE_REVISION for {doc_key!r}")
        sys.exit(1)
    print("ACTIVE revision:", dict(row))

    active_sections = s.run(
        """
        MATCH (s:Section)
        WHERE s.logical_doc_id = $lid AND coalesce(s.lifecycle_status,'ACTIVE') = 'ACTIVE'
        RETURN count(s) AS n
        """,
        lid=doc_key,
    ).single()["n"]
    expired_sections = s.run(
        """
        MATCH (s:Section)
        WHERE s.logical_doc_id = $lid AND s.lifecycle_status = 'EXPIRED'
        RETURN count(s) AS n
        """,
        lid=doc_key,
    ).single()["n"]
    print(f"ACTIVE sections: {active_sections}, EXPIRED sections: {expired_sections}")

    revs = s.run(
        """
        MATCH (dl:DocumentLogical {logical_id: $lid})-[:HAS_REVISION]->(rev:DocRevision)
        RETURN rev.id AS id, rev.version_number AS v, rev.status AS status
        ORDER BY rev.version_number
        """,
        lid=doc_key,
    )
    print("All revisions:")
    for r in revs:
        print(" ", dict(r))

driver.close()
print("Neo4j checks OK")
PY

echo ""
echo "== 5) Optional: second ingest after editing PDF =="
echo "Change one byte in the PDF, re-run step 2 with a new doc_key or same key;"
echo "Expect version_number=2, one EXPIRED revision, new ACTIVE sections."
