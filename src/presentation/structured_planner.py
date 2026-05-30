"""
Build presentation blocks for structured Neo4j queries (Northwind, analytics, top-N, etc.).
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..structured.query_intent import is_singular_best_query

_ANALYTICS = re.compile(
    r"\b(top|bottom|best|worst|highest|lowest|most|least|ranking|rank|"
    r"sales|sold|revenue|profit|count|sum|total|average|compare|trend|"
    r"products?|orders?|customers?)\b",
    re.I,
)
_LABEL_PRIORITY = (
    "month",
    "productName",
    "companyName",
    "categoryName",
    "customerName",
    "name",
    "title",
    "label",
)
_VALUE_PRIORITY = (
    "ordervolume",
    "order_volume",
    "unitssold",
    "units_sold",
    "totalrevenue",
    "total_revenue",
    "revenue",
    "sales",
    "amount",
    "total",
    "sum",
    "count",
    "quantity",
    "units",
    "orders",
    "value",
)
_SKIP_VALUE_KEYS = re.compile(
    r"(^|_)(id|uuid|key)$|productid|customerid|orderid|employeeid|supplierid|categoryid",
    re.I,
)


def is_structured_analytics_query(question: str) -> bool:
    return bool(_ANALYTICS.search(question))


def _is_number(val: Any) -> bool:
    if val is None:
        return False
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return True
    try:
        float(str(val).replace(",", ""))
        return True
    except (TypeError, ValueError):
        return False


def _to_float(val: Any) -> float:
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return float(str(val).replace(",", ""))


def extract_rows_from_sources(sources: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for src in sources or []:
        raw = src.get("raw")
        if isinstance(raw, dict) and raw:
            rows.append(dict(raw))
    return rows


def _pick_columns(rows: list[dict]) -> tuple[Optional[str], Optional[str], list[str]]:
    if not rows:
        return None, None, []
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)

    numeric = [
        k for k in keys
        if not _SKIP_VALUE_KEYS.search(k.replace(" ", ""))
        and sum(1 for r in rows if _is_number(r.get(k))) >= min(2, len(rows))
    ]
    stringish = [
        k for k in keys
        if not all(_is_number(r.get(k)) for r in rows)
        and any(r.get(k) is not None and str(r.get(k)).strip() for r in rows)
    ]

    label_key = None
    for pref in _LABEL_PRIORITY:
        if pref in stringish:
            label_key = pref
            break
    if not label_key and stringish:
        label_key = stringish[0]

    value_key = None
    for pref in _VALUE_PRIORITY:
        for k in numeric:
            kn = k.lower().replace(" ", "").replace("_", "")
            if pref in kn:
                value_key = k
                break
        if value_key:
            break
    if not value_key and numeric:
        value_key = max(
            numeric,
            key=lambda k: sum(_to_float(r.get(k)) for r in rows if _is_number(r.get(k))),
        )

    display_keys = [k for k in keys if k in (label_key, value_key) or k in numeric]
    if label_key and label_key not in display_keys:
        display_keys.insert(0, label_key)
    if value_key and value_key not in display_keys:
        display_keys.append(value_key)
    if not display_keys:
        display_keys = keys[:8]
    return label_key, value_key, display_keys


def build_structured_presentation(
    question: str,
    answer: str,
    sources: list[dict],
) -> Optional[dict]:
    """
    Returns { kind, blocks } or None if not suitable for rich structured UI.
    """
    rows = extract_rows_from_sources(sources)
    if not rows:
        return None
    if not is_structured_analytics_query(question):
        return None

    if is_singular_best_query(question) or len(rows) == 1:
        return {
            "kind": "plain",
            "blocks": [{"type": "markdown", "content": (answer or "").strip()}],
        }

    if len(rows) < 2:
        return None

    label_key, value_key, display_keys = _pick_columns(rows)
    if not display_keys:
        return None

    headers = [_human_key(k) for k in display_keys]
    table_rows: list[list[str]] = []
    labels: list[str] = []
    values: list[float] = []

    chart_rows: list[tuple[str, float]] = []
    for row in rows[:25]:
        table_rows.append([_cell(row.get(k)) for k in display_keys])
        if label_key and value_key and _is_number(row.get(value_key)):
            lbl = _cell(row.get(label_key)) or "—"
            chart_rows.append((lbl[:40], _to_float(row.get(value_key))))

    if chart_rows:
        chart_rows.sort(key=lambda x: x[1], reverse=True)
    labels = [r[0] for r in chart_rows]
    values = [r[1] for r in chart_rows]

    blocks: list[dict] = []

    if value_key and len(values) >= 2 and _values_vary_enough(values):
        chart_title = _chart_title(question, value_key)
        blocks.append({
            "type": "chart",
            "chartType": "bar",
            "interactive": True,
            "valueFormat": "number",
            "valueKey": value_key,
            "title": chart_title,
            "labels": labels[:12],
            "values": values[:12],
        })

    blocks.append({
        "type": "table",
        "title": "Query results",
        "headers": headers,
        "rows": table_rows,
        "interactive": True,
    })

    blocks.append({"type": "markdown", "content": answer or ""})

    kinds = {b["type"] for b in blocks}
    kind = "mixed" if len(kinds) > 1 else "table"
    return {"kind": kind, "blocks": blocks}


def _human_key(key: str) -> str:
    return re.sub(r"([a-z])([A-Z])", r"\1 \2", key).replace("_", " ").title()


def _cell(val: Any) -> str:
    if val is None:
        return ""
    if hasattr(val, "iso_format"):
        return val.iso_format()[:7]
    if hasattr(val, "year") and hasattr(val, "month"):
        return f"{val.year}-{val.month:02d}"
    if isinstance(val, float):
        return f"{val:,.2f}" if abs(val) < 1e9 else f"{val:.2e}"
    s = str(val)
    if len(s) >= 10 and s[4] == "-" and s[7] == "-":
        return s[:7]
    return s


def _values_vary_enough(values: list[float]) -> bool:
    if len(values) < 2:
        return False
    lo, hi = min(values), max(values)
    if hi <= 0:
        return False
    if lo == hi:
        return False
    return (hi - lo) / max(abs(hi), 1.0) >= 0.01


def _chart_title(question: str, value_key: str) -> str:
    m = re.search(r"\btop\s+(\d+)\b", question, re.I)
    if m:
        return f"Top {m.group(1)} by {_human_key(value_key)}"
    return f"By {_human_key(value_key)}"
