"""Convert Neo4j rows into retrieval chunks and API responses."""
from __future__ import annotations

from typing import Any

from ..neo4j_sanitize import sanitize_row


def row_title(row: dict) -> str:
    for key in ("productName", "name", "title", "companyName", "categoryName", "customerID", "orderID"):
        if key in row and row[key] is not None:
            return str(row[key])
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
