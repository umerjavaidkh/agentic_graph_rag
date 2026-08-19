"""Neo4j graph schema introspection with in-memory cache."""
from __future__ import annotations

from typing import Optional

from neo4j import Driver

from ....unstructured.graph.constants import NON_BUSINESS_LABELS, SCHEMA_DOC_LABEL


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
# As reported by db.schema.nodeTypeProperties().
_NUMERIC_TYPES = frozenset({"Long", "Double", "Integer", "Float"})

_EXAMPLES_PER_PROPERTY = 3
_ROWS_SAMPLED_PER_LABEL = 25
_MAX_VALUE_CHARS = 40

# A property with few enough distinct values is an enum, and for an enum the
# COMPLETE set belongs in the prompt rather than a sample. Sampling rows only
# ever surfaces common values: order status 'canceled' is 0.6% of rows, so a
# 25-row sample essentially never contains it, and a question about cancelled
# orders was answered "there were no orders cancelled" from a filter matching
# the British spelling against American data. 80 covers a status list, a
# payment-type list, a country's states and a category list without
# meaningfully enlarging the prompt.
_ENUM_MAX_VALUES = 80
# Cardinality needs a scan per label. Skipping the largest labels keeps
# introspection bounded on a big graph; those are id-like columns anyway.
_MAX_NODES_FOR_CARDINALITY = 2_000_000


def _descriptions(session) -> list[str]:
    """Human-written notes on what each field MEANS.

    Types and sample values say what a column holds, never what it is for,
    and several wrong answers came from exactly that gap: "which customer
    state pays the highest freight" grouped by `seller.state` because both
    are a two-letter code on a node the query already touched, and revenue
    per seller was computed as avg(price) because nothing said price is a
    per-line amount that must be summed first.

    Stored as nodes rather than a dict in this file so the notes belong to
    the dataset, not to the code -- a different graph brings its own, and a
    graph with none is unaffected.
    """
    try:
        rows = session.run(
            "MATCH (d:`%s`) RETURN d.target AS target, d.text AS text ORDER BY d.target"
            % SCHEMA_DOC_LABEL
        )
        return [f"{r['target']} — {r['text']}" for r in rows if r["target"] and r["text"]]
    except Exception:
        return []


def _value_sets(session, labels: list[str], props: dict[str, set[str]]) -> list[str]:
    """Complete value lists for enum-like properties, cardinality for the rest.

    Two things the model cannot get from a sample. First, the full set of an
    enum, so it filters on a value that exists rather than the wording the
    question happened to use. Second, how many distinct values a property has
    relative to the node count -- which is the only way to tell an identifier
    from a grouping key. Olist gives every order a fresh `customer_id`, so
    counting distinct customer_id returns the order count, and every
    retention question answers "nobody bought twice". `unique_id` having
    fewer distinct values than there are nodes is exactly that signal.

    One scan per label: all the counts come from a single aggregate.
    """
    lines: list[str] = []
    for label in labels:
        # The document tree shares this database but is never the subject of a
        # structured question, and enumerating its values tripled the prompt.
        if label in NON_BUSINESS_LABELS:
            continue
        names = sorted(props.get(label, set()))
        if not names:
            continue
        try:
            total = session.run("MATCH (n:`%s`) RETURN count(n) AS c" % label).single()["c"]
            if not total or total > _MAX_NODES_FOR_CARDINALITY:
                continue
            counts = ", ".join(
                "count(DISTINCT n.`%s`) AS `%s`" % (n, n) for n in names
            )
            row = session.run("MATCH (n:`%s`) RETURN %s" % (label, counts)).single()
        except Exception:
            continue
        for name in names:
            d = row[name]
            if d is None or d == 0:
                continue
            if d <= _ENUM_MAX_VALUES:
                try:
                    vals = [
                        r["v"] for r in session.run(
                            "MATCH (n:`%s`) WHERE n.`%s` IS NOT NULL "
                            "RETURN DISTINCT n.`%s` AS v ORDER BY v" % (label, name, name)
                        )
                    ]
                except Exception:
                    continue
                shown = ", ".join(repr(v)[:40] for v in vals)
                lines.append(":%s.%s — all %d values: %s" % (label, name, d, shown))
            elif d < total:
                lines.append(
                    ":%s.%s — %s distinct across %s nodes (repeats, so it groups)"
                    % (label, name, format(d, ","), format(total, ","))
                )
            else:
                lines.append(
                    ":%s.%s — %s distinct across %s nodes (unique per node, an identifier)"
                    % (label, name, format(d, ","), format(total, ","))
                )
    return lines


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
        self._numeric_cache: Optional[set[tuple[str, str]]] = None

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

    def numeric_properties(self) -> set[tuple[str, str]]:
        """(label, property) pairs whose values are numeric.

        Used to decide whether a question about an aggregate is genuinely
        ambiguous in THIS graph, rather than assuming a fixed set of metrics.
        """
        if self._numeric_cache is None:
            self.fetch()
        return self._numeric_cache or set()

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
            numeric: set[tuple[str, str]] = set()
            for r in rows:
                names = {p.split(":", 1)[0].strip() for p in r["properties"]}
                numeric_names = {
                    p.split(":", 1)[0].strip()
                    for p in r["properties"]
                    if p.split(":", 1)[-1].strip() in _NUMERIC_TYPES
                }
                for label in _parse_node_type_labels(r["nodeType"]):
                    labels.add(label)
                    props.setdefault(label, set()).update(names)
                    if label not in NON_BUSINESS_LABELS:
                        numeric.update((label, n) for n in numeric_names)
            self._labels_cache = labels
            self._numeric_cache = numeric

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
            value_sets = _value_sets(session, sorted(labels), props)
            described = _descriptions(session)
        self._props_cache = props

        schema = (
            "NODE TYPES:\n" + "\n".join(nodes) +
            "\n\nRELATIONSHIP TYPES:\n" + "\n".join(sorted(set(patterns))) +
            "\n\nRELATIONSHIP PROPERTIES:\n" + "\n".join(rel_lines) +
            "\n\nEXAMPLE VALUES (real values sampled from this graph -- when a\n"
            "value in the question resembles one of these, filter on THAT property,\n"
            "not on a same-named property holding a different vocabulary):\n"
            + "\n".join(examples)
            + ("\n\nWHAT EACH FIELD MEANS (read this before choosing which node or\n"
               "property to group, filter or aggregate on):\n" + "\n".join(described)
               if described else "")
            + ("\n\nVALUE SETS AND CARDINALITY (complete lists where a property is an\n"
               "enum; for the rest, how many distinct values exist -- a property with\n"
               "fewer distinct values than nodes repeats, so it groups rather than\n"
               "identifies):\n" + "\n".join(value_sets) if value_sets else "")
        )
        self._cache = schema
        return schema

    def clear_cache(self) -> None:
        self._cache = None
        self._labels_cache = None
        self._props_cache = None
        self._numeric_cache = None
