#!/usr/bin/env python3
"""Block until Neo4j accepts Bolt connections."""
from __future__ import annotations

import os
import sys
import time

from neo4j import GraphDatabase

URI = os.environ.get("NEO4J_URI", "bolt://neo4j:7687")
USER = os.environ.get("NEO4J_USER", "neo4j")
PASSWORD = os.environ.get("NEO4J_PASSWORD", "password123")
MAX_WAIT = int(os.environ.get("NEO4J_WAIT_SECONDS", "120"))


def main() -> int:
    deadline = time.time() + MAX_WAIT
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            driver = GraphDatabase.driver(URI, auth=(USER, PASSWORD))
            try:
                driver.verify_connectivity()
                print(f"Neo4j ready at {URI}")
                return 0
            finally:
                driver.close()
        except Exception as exc:
            last_err = exc
            print(f"Waiting for Neo4j ({URI})… {exc}")
            time.sleep(3)
    print(f"Neo4j not ready after {MAX_WAIT}s: {last_err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
