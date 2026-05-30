"""
Fast path: build a short answer from tabular Cypher rows (no LLM).
"""
from __future__ import annotations

from typing import Optional

from ..presentation.structured_planner import (
    _cell,
    _human_key,
    _pick_columns,
    extract_rows_from_sources,
)


def try_tabular_answer(chunks: list[dict]) -> Optional[str]:
    if any(c.get("id") in ("error", "access_denied") for c in chunks):
        return None

    sources = [{"raw": c["raw"]} for c in chunks if isinstance(c.get("raw"), dict)]
    rows = extract_rows_from_sources(sources)
    if not rows:
        return None

    label_key, value_key, _ = _pick_columns(rows)
    if not label_key or not value_key:
        return None

    lines = ["Here are the results from the database:\n"]
    for i, row in enumerate(rows[:10], 1):
        name = _cell(row.get(label_key)) or "—"
        metric = _cell(row.get(value_key))
        lines.append(f"{i}. **{name}** — {_human_key(value_key)}: {metric}")

    return "\n".join(lines)
