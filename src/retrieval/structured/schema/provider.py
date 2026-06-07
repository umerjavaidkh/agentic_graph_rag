"""Neo4j graph schema introspection with in-memory cache."""
from __future__ import annotations

from typing import Optional

from neo4j import Driver


class SchemaProvider:
    """
    In-memory cache for Neo4j schema introspection.

    The graph schema string is built once per process (first fetch) and reused for
    every Text-to-Cypher / multistep call. It is still embedded in each LLM prompt
    (token cost), but Neo4j is not re-queried on every request.
    """

    def __init__(self, driver: Driver):
        self._driver = driver
        self._cache: Optional[str] = None

    def fetch(self) -> str:
        if self._cache:
            return self._cache
        with self._driver.session() as session:
            labels_result = session.run(
                """
                CALL db.schema.nodeTypeProperties()
                YIELD nodeType, propertyName, propertyTypes
                RETURN nodeType, collect(propertyName + ': ' + propertyTypes[0]) AS properties
                """
            )
            nodes = [f"{r['nodeType']} {{{', '.join(r['properties'])}}}" for r in labels_result]

            patterns_result = session.run(
                """
                MATCH (a)-[r]->(b)
                RETURN DISTINCT labels(a)[0] AS from, type(r) AS rel, labels(b)[0] AS to
                """
            )
            patterns = [f"(:{r['from']})-[:{r['rel']}]->(:{r['to']})" for r in patterns_result]

            rel_lines: list[str] = []
            try:
                rel_props = session.run(
                    """
                    CALL db.schema.relTypeProperties()
                    YIELD relType, propertyName, propertyTypes
                    RETURN relType, collect(propertyName + ': ' + propertyTypes[0]) AS properties
                    """
                )
                rel_lines = [f"{r['relType']} {{{', '.join(r['properties'])}}}" for r in rel_props]
            except Exception:
                rel_lines = ["(relationship properties unavailable)"]

        schema = (
            "NODE TYPES:\n" + "\n".join(nodes) +
            "\n\nRELATIONSHIP TYPES:\n" + "\n".join(sorted(set(patterns))) +
            "\n\nRELATIONSHIP PROPERTIES:\n" + "\n".join(rel_lines)
        )
        self._cache = schema
        return schema

    def clear_cache(self) -> None:
        self._cache = None
