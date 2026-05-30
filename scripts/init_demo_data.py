#!/usr/bin/env python3
"""Load Northwind sample data once (Docker demo bootstrap)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.ingestion.service import IngestionManager  # noqa: E402

URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "password123")
CYPHER_FILE = PROJECT_ROOT / "docker" / "northwind-docker.cypher"
LOAD_DEMO = os.environ.get("LOAD_NORTHWIND_DEMO", "true").lower() in ("1", "true", "yes")


def _product_count(session) -> int:
    row = session.run("MATCH (p:Product) RETURN count(p) AS c").single()
    return int(row["c"]) if row else 0


def _run_cypher_file(session, path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    statements, params = IngestionManager()._parse_cypher_script(text)
    for idx, stmt in enumerate(statements, start=1):
        preview = " ".join(stmt.split())[:100]
        print(f"  [{idx}/{len(statements)}] {preview}…")
        try:
            session.run(stmt, **params).consume()
        except ClientError as exc:
            code = getattr(exc, "code", "") or ""
            if code in {
                "Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists",
                "Neo.ClientError.Schema.IndexAlreadyExists",
                "Neo.ClientError.Schema.ConstraintAlreadyExists",
            }:
                print(f"    (skip schema: {code})")
                continue
            raise


def main() -> int:
    if not LOAD_DEMO:
        print("LOAD_NORTHWIND_DEMO=false — skipping Northwind bootstrap")
        return 0

    if not CYPHER_FILE.is_file():
        print(f"Missing {CYPHER_FILE}", file=sys.stderr)
        return 1

    driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
    try:
        with driver.session() as session:
            count = _product_count(session)
            if count > 0:
                print(f"Northwind already loaded ({count} products) — skip")
                return 0
            print(f"Loading Northwind demo from {CYPHER_FILE.name}…")
            _run_cypher_file(session, CYPHER_FILE)
            count = _product_count(session)
            print(f"Northwind loaded ({count} products)")
    finally:
        driver.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
