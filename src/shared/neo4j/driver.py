"""
Shared Neo4j driver for the process.

The official driver keeps a connection pool behind one Driver instance.
Create it once, reuse it, and open a short-lived session per request/unit of work.
Do not create a new Driver per query — that adds TCP + auth overhead.
"""
from __future__ import annotations

import threading
from typing import Optional

from neo4j import Driver, GraphDatabase

from ..config.settings import NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

_lock = threading.Lock()
_driver: Optional[Driver] = None


def get_neo4j_driver(
    uri: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> Driver:
    """Return the process-wide Neo4j driver (lazy singleton, thread-safe)."""
    global _driver
    if _driver is not None:
        return _driver
    with _lock:
        if _driver is None:
            _driver = GraphDatabase.driver(
                uri or NEO4J_URI,
                auth=(user or NEO4J_USER, password or NEO4J_PASSWORD),
            )
    return _driver


def close_neo4j_driver() -> None:
    """Close the shared driver (tests / graceful shutdown)."""
    global _driver
    with _lock:
        if _driver is not None:
            _driver.close()
            _driver = None
