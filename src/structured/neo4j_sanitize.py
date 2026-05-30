"""
Convert Neo4j driver types (Date, DateTime, etc.) to JSON-serializable values.
"""
from __future__ import annotations

from datetime import date, datetime, time
from typing import Any


def sanitize_neo4j_value(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (str, int, float, bool)):
        return val
    if isinstance(val, (date, datetime, time)):
        return val.isoformat()
    mod = getattr(type(val), "__module__", "") or ""
    if mod.startswith("neo4j"):
        if hasattr(val, "iso_format"):
            return val.iso_format()
        if hasattr(val, "to_native"):
            return sanitize_neo4j_value(val.to_native())
        return str(val)
    if isinstance(val, dict):
        return {k: sanitize_neo4j_value(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [sanitize_neo4j_value(v) for v in val]
    return val


def sanitize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {k: sanitize_neo4j_value(v) for k, v in row.items()}
