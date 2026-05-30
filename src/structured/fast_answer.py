"""
Fast path: build a short answer from tabular Cypher rows (no LLM).
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..presentation.structured_planner import (
    _cell,
    _human_key,
    _is_number,
    _pick_columns,
    _to_float,
    extract_rows_from_sources,
)
from .query_intent import is_singular_best_query

_SKIP_METRIC = re.compile(
    r"(^|_)(id|uuid|key)$|productid|customerid|orderid",
    re.I,
)


def try_tabular_answer(
    chunks: list[dict],
    question: str = "",
) -> Optional[str]:
    if any(c.get("id") in ("error", "access_denied") for c in chunks):
        return None

    sources = [{"raw": c["raw"]} for c in chunks if isinstance(c.get("raw"), dict)]
    rows = extract_rows_from_sources(sources)
    if not rows:
        return None

    if is_singular_best_query(question) or (
        len(rows) == 1 and re.search(r"\b(?:best|one|single|top)\b", question, re.I)
    ):
        singular = _format_singular_best_answer(rows[0], question)
        if singular:
            return singular

    label_key, value_key, _ = _pick_columns(rows)
    if not label_key or not value_key:
        return None

    lines = ["Here are the results from the database:\n"]
    for i, row in enumerate(rows[:10], 1):
        name = _cell(row.get(label_key)) or "—"
        metric = _cell(row.get(value_key))
        lines.append(f"{i}. **{name}** — {_human_key(value_key)}: {metric}")

    return "\n".join(lines)


def _format_singular_best_answer(row: dict, question: str) -> Optional[str]:
    label_key, value_key, keys = _pick_columns([row])
    name = _cell(row.get(label_key)) if label_key else None
    if not name:
        for k in ("productName", "companyName", "name", "title"):
            if row.get(k):
                name = _cell(row[k])
                break
    if not name:
        return None

    metrics: list[tuple[str, str]] = []
    for k in keys:
        if k == label_key or _SKIP_METRIC.search(k.replace(" ", "")):
            continue
        val = row.get(k)
        if _is_number(val):
            metrics.append((_human_key(k), _cell(val)))

    q = question.lower()
    wants_profit = "profit" in q
    lines = [f"The **best-performing product** in the database is **{name}**.", ""]

    for label, val in metrics:
        lines.append(f"- **{label}:** {val}")

    if wants_profit and not any("profit" in m[0].lower() for m in metrics):
        lines.append(
            "\n_Profit is not stored in this graph; **total revenue** "
            "(line-item sales after discount) is shown instead._"
        )

    if not metrics and value_key:
        lines.append(f"- **{_human_key(value_key)}:** {_cell(row.get(value_key))}")

    return "\n".join(lines)
