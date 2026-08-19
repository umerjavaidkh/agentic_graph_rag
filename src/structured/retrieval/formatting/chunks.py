"""Convert Neo4j rows into retrieval chunks and API responses."""
from __future__ import annotations

from typing import Any

from ..neo4j_sanitize import sanitize_row


# Suffixes that mark a column as the human-readable one, in preference
# order. Generic English, not any schema's column list -- `productName` and
# `category_label` both match without being named here.
_TITLE_SUFFIXES = ("name", "title", "label")


def row_title(row: dict) -> str:
    """Best human-readable value in a result row.

    Prefers a name/title/label column, then any other text, and falls back to
    the first value. Matching on suffix rather than on a fixed list of column
    names keeps this working when the graph is swapped for another dataset --
    the old list only recognised one schema's columns and returned an opaque
    id for every other.
    """
    for suffix in _TITLE_SUFFIXES:
        for key, value in row.items():
            if value is not None and key.lower().endswith(suffix):
                return str(value)
    for key, value in row.items():
        if isinstance(value, str) and value.strip():
            return value
    return str(list(row.values())[0]) if row else "Result"


def row_to_text(row: dict) -> str:
    return "\n".join(f"{k}: {v}" for k, v in row.items() if v is not None)


def rows_to_chunks(rows: list[dict], cypher: str) -> list[dict]:
    out: list[dict] = []
    for i, row in enumerate(rows):
        clean = sanitize_row(row)
        out.append({
            "id": f"row_{i}",
            "title": row_title(clean),
            "text": row_to_text(clean),
            "raw": clean,
            "score": 1.0,
            "cypher": cypher,
            "related": [],
        })
    return out


def format_response(query: str, items: list, strategy: str) -> dict[str, Any]:
    return {
        "query": query,
        "strategy": strategy,
        "chunks": items,
        "total_available": len(items),
    }
