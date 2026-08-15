"""Neo4j graph schema introspection with in-memory cache."""
from __future__ import annotations

from typing import Optional

from neo4j import Driver


def _parse_node_type_labels(node_type: str) -> list[str]:
    """"`:Order`" -> ["Order"]; "`:Label1`:`Label2`" (multi-label nodes) -> both."""
    labels: list[str] = []
    for part in (node_type or "").split(":"):
        part = part.strip().strip("`")
        if part:
            labels.append(part)
    return labels


# Enough to show the shape of a value, few enough to keep the prompt small
# and to avoid pasting a whole free-text column into it.
_EXAMPLES_PER_PROPERTY = 3
_ROWS_SAMPLED_PER_LABEL = 25
_MAX_VALUE_CHARS = 40


def _sample_values(session, labels: list[str]) -> list[str]:
    """A few real string values per property, for the prompt.

    Property NAMES alone cannot say which vocabulary a property holds. A
    category stored as `name` in Portuguese beside `name_english` looks
    identical in a types-only schema, so a question naming an English
    category was matched against the Portuguese column and returned "no
    sellers" for a category with 196 of them. Values disambiguate that, and
    the same applies to any code-vs-label or enum column.

    Only strings are sampled: numbers and dates say nothing a type has not
    already said. Long values are skipped rather than truncated, so a
    free-text column contributes nothing instead of a misleading fragment.
    """
    lines: list[str] = []
    for label in labels:
        try:
            rows = session.run(
                f"MATCH (n:`{label}`) RETURN n LIMIT $limit",
                limit=_ROWS_SAMPLED_PER_LABEL,
            )
            seen: dict[str, list[str]] = {}
            for row in rows:
                for key, value in dict(row["n"]).items():
                    if not isinstance(value, str) or len(value) > _MAX_VALUE_CHARS:
                        continue
                    bucket = seen.setdefault(key, [])
                    if value not in bucket and len(bucket) < _EXAMPLES_PER_PROPERTY:
                        bucket.append(value)
        except Exception:
            continue
        for key, values in sorted(seen.items()):
            if values:
                shown = ", ".join(repr(v) for v in values)
                lines.append(f":{label}.{key} e.g. {shown}")
    return lines or ["(no sampled values)"]


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
        self._labels_cache: Optional[set[str]] = None
        self._props_cache: Optional[dict[str, set[str]]] = None

    def known_labels(self) -> set[str]:
        """Every node label actually present in the graph -- used to catch a
        generated Cypher query referencing a label that doesn't exist (an LLM
        hallucination of a schema shape it expects rather than the one it was
        given, e.g. assuming a first-class Employee node when this graph only
        has an employeeID property on Order). Populated as a side effect of
        fetch() the first time either is called, so this never costs a second
        DB round trip."""
        if self._labels_cache is None:
            self.fetch()
        return self._labels_cache or set()

    def known_properties(self) -> dict[str, set[str]]:
        """Property names per label AND per relationship type.

        The same introspection that builds the prompt schema, kept as data so
        a generated query can be checked against it rather than only described
        to the model. Labels and relationship types share one mapping: a name
        collision between the two is rare and merely unions the two property
        sets, which can only make the check more permissive -- the safe
        direction for something that rejects queries.
        """
        if self._props_cache is None:
            self.fetch()
        return self._props_cache or {}

    def fetch(self) -> str:
        if self._cache is not None and self._labels_cache is not None:
            return self._cache
        with self._driver.session() as session:
            labels_result = session.run(
                """
                CALL db.schema.nodeTypeProperties()
                YIELD nodeType, propertyName, propertyTypes
                RETURN nodeType, collect(propertyName + ': ' + propertyTypes[0]) AS properties
                """
            )
            rows = [dict(r) for r in labels_result]
            nodes = [f"{r['nodeType']} {{{', '.join(r['properties'])}}}" for r in rows]
            labels: set[str] = set()
            props: dict[str, set[str]] = {}
            for r in rows:
                names = {p.split(":", 1)[0].strip() for p in r["properties"]}
                for label in _parse_node_type_labels(r["nodeType"]):
                    labels.add(label)
                    props.setdefault(label, set()).update(names)
            self._labels_cache = labels

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
                for r in rel_props:
                    rel_lines.append(f"{r['relType']} {{{', '.join(r['properties'])}}}")
                    names = {p.split(":", 1)[0].strip() for p in r["properties"]}
                    for rel in _parse_node_type_labels(r["relType"]):
                        props.setdefault(rel, set()).update(names)
            except Exception:
                rel_lines = ["(relationship properties unavailable)"]

            examples = _sample_values(session, sorted(labels))
        self._props_cache = props

        schema = (
            "NODE TYPES:\n" + "\n".join(nodes) +
            "\n\nRELATIONSHIP TYPES:\n" + "\n".join(sorted(set(patterns))) +
            "\n\nRELATIONSHIP PROPERTIES:\n" + "\n".join(rel_lines) +
            "\n\nEXAMPLE VALUES (real values sampled from this graph -- when a\n"
            "value in the question resembles one of these, filter on THAT property,\n"
            "not on a same-named property holding a different vocabulary):\n"
            + "\n".join(examples)
        )
        self._cache = schema
        return schema

    def clear_cache(self) -> None:
        self._cache = None
        self._labels_cache = None
        self._props_cache = None
