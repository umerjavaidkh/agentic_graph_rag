"""
Generate the field notes the Cypher generator reads, for any loaded dataset.

A type and a sample value say what a column holds and never what it is for,
and that gap produced real wrong answers: "which customer state pays the most
freight" grouped by `seller.state`, because both are a two-letter code on a
node the query already touched; "average revenue per seller" was computed as
`avg(price)`, because nothing said price is a per-line amount that must be
summed before it means revenue.

Hand-written notes fix that but only for a dataset someone has sat down with.
This fills the gap for everything else: at ingestion, any label or property
with no note gets one written from what the graph can show about it -- its
type, real sample values, how many distinct values it has against the node
count, and the relationships its label takes part in.

One LLM call per label, not per property, so a wide table costs one request.
Notes already present are never overwritten: a human who has written a better
one keeps it.
"""
from __future__ import annotations

import logging
from typing import Any, Iterable, Optional

from ..config.settings import STRUCTURED_FALLBACK_MODEL
from ..graph.constants import NON_BUSINESS_LABELS, SCHEMA_DOC_LABEL
from ..model_providers.factory import get_chat_provider

logger = logging.getLogger(__name__)

# Enough rows to show what values look like without pulling a column's whole
# vocabulary into a prompt.
_SAMPLE_ROWS = 20
_MAX_VALUE_CHARS = 40

_PROMPT = """You are documenting a Neo4j graph so a text-to-Cypher model can query it correctly.

LABEL: {label}  ({count:,} nodes)

RELATIONSHIPS THIS LABEL PARTICIPATES IN:
{patterns}

PROPERTIES (name, type, distinct values vs node count, real examples):
{properties}

For EACH property, and one for the label itself, write a single line:

  <target>|<description>

where <target> is `{label}` for the label, or `{label}.<property>` for a
property, and <description> is ONE sentence that states:
  - what the value actually represents in the real world
  - its type
  - how to use it correctly in a query

That last part is what matters most. Say things a query author could get
wrong and would not learn from the type:
  - if a value is one line of a larger whole, say that a total means SUM and
    that averaging it answers a different question
  - if "average X per Y" needs grouping by Y and aggregating first, say so
  - if two labels both carry a similarly-named property, say which question
    each one answers
  - if a property is unique per node it is an identifier; if it repeats it
    groups
  - if a value is a physical measure, a code or an id, say it is never an
    answer about money or names

Output only those lines, nothing else."""


def _existing_targets(session) -> set[str]:
    try:
        return {
            r["t"]
            for r in session.run(f"MATCH (d:{SCHEMA_DOC_LABEL}) RETURN d.target AS t")
            if r["t"]
        }
    except Exception:
        return set()


def _label_context(session, label: str) -> Optional[dict[str, Any]]:
    """Everything the model needs about one label, in a single pass per query."""
    count = session.run(f"MATCH (n:`{label}`) RETURN count(n) AS c").single()["c"]
    if not count:
        return None
    rows = [
        dict(r["n"])
        for r in session.run(
            f"MATCH (n:`{label}`) RETURN n LIMIT $k", k=_SAMPLE_ROWS
        )
    ]
    if not rows:
        return None

    names = sorted({k for row in rows for k in row})
    counts_clause = ", ".join(f"count(DISTINCT n.`{n}`) AS `{n}`" for n in names)
    distinct = session.run(f"MATCH (n:`{label}`) RETURN {counts_clause}").single()

    lines = []
    for name in names:
        samples, seen = [], set()
        for row in rows:
            v = row.get(name)
            if v is None:
                continue
            text = str(v)[:_MAX_VALUE_CHARS]
            if text not in seen:
                seen.add(text)
                samples.append(text)
            if len(samples) >= 3:
                break
        type_name = type(next((r[name] for r in rows if r.get(name) is not None), "")).__name__
        lines.append(
            f"  {name} ({type_name}) — {distinct[name]:,} distinct of {count:,} — "
            f"e.g. {', '.join(repr(s) for s in samples) or 'n/a'}"
        )

    patterns = [
        f"  (:{label})-[:{r['t']}]->(:{r['o']})"
        for r in session.run(
            f"MATCH (:`{label}`)-[e]->(b) RETURN DISTINCT type(e) AS t, labels(b)[0] AS o"
        )
    ] + [
        f"  (:{r['o']})-[:{r['t']}]->(:{label})"
        for r in session.run(
            f"MATCH (a)-[e]->(:`{label}`) RETURN DISTINCT type(e) AS t, labels(a)[0] AS o"
        )
    ]
    return {
        "label": label,
        "count": count,
        "properties": "\n".join(lines),
        "patterns": "\n".join(patterns) or "  (none)",
    }


def ensure_field_docs(
    session,
    *,
    labels: Optional[Iterable[str]] = None,
    provider=None,
    model: Optional[str] = None,
    overwrite: bool = False,
) -> int:
    """Write a note for every label/property that does not already have one.

    Returns how many were written. Failure is logged and swallowed: missing
    documentation degrades query quality, but a description service that
    cannot reach its provider must never stop a data load from finishing.
    """
    provider = provider or get_chat_provider()
    model = model or STRUCTURED_FALLBACK_MODEL
    have = set() if overwrite else _existing_targets(session)

    if labels is None:
        labels = [
            r["l"]
            for r in session.run("MATCH (n) UNWIND labels(n) AS l RETURN DISTINCT l AS l ORDER BY l")
            if r["l"] not in NON_BUSINESS_LABELS and r["l"] != SCHEMA_DOC_LABEL
        ]

    written = 0
    for label in labels:
        try:
            context = _label_context(session, label)
            if context is None:
                continue
            # Skip the label entirely when everything about it is already
            # documented -- re-describing costs a request and changes nothing.
            props = [ln.strip().split(" ")[0] for ln in context["properties"].splitlines()]
            if not overwrite and all(
                f"{label}.{p}" in have for p in props
            ) and label in have:
                continue

            reply = provider.chat_completion(
                model=model,
                temperature=0,
                messages=[{"role": "user", "content": _PROMPT.format(**context)}],
                max_tokens=900,
            )
            text = (reply.choices[0].message.content or "").strip()
            rows = []
            for line in text.splitlines():
                if "|" not in line:
                    continue
                target, _, description = line.partition("|")
                target, description = target.strip().lstrip("-").strip(), description.strip()
                if not target or not description or (target in have and not overwrite):
                    continue
                rows.append({"target": target, "text": description})
            if rows:
                session.run(
                    f"UNWIND $rows AS r MERGE (d:{SCHEMA_DOC_LABEL} {{target: r.target}}) "
                    f"SET d.text = r.text, d.generated = true",
                    rows=rows,
                )
                written += len(rows)
        except Exception as exc:
            logger.warning("schema docs: %s skipped (%s)", label, exc)
    return written
