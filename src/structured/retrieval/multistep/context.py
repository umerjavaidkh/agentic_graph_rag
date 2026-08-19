"""Helpers for chaining multistep Cypher via UNWIND parameters."""
from __future__ import annotations

import re
from typing import Any

_PARAM_RE = re.compile(r"\$(\w+)")


def extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object from an LLM response."""
    t = (text or "").strip()
    if not t:
        return "{}"
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        return t[start : end + 1]
    return t


def find_param_names(cypher: str) -> list[str]:
    return sorted(set(_PARAM_RE.findall(cypher or "")))


def collect_values_from_ctx(ctx: dict[str, Any], key: str) -> list[Any]:
    """
    Collect values for a key from previous step rows.

    ctx holds {step_id: {"rows": [...]}} entries.
    """
    values: list[Any] = []
    for v in ctx.values():
        if not isinstance(v, dict):
            continue
        rows = v.get("rows")
        if not isinstance(rows, list):
            continue
        for r in rows:
            if isinstance(r, dict) and key in r and r[key] is not None:
                values.append(r[key])
    return list(dict.fromkeys(values))


def normalize_row_keys(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize keys so they can be referenced as row.<key> in later UNWIND steps."""
    out: dict[str, Any] = {}
    for k, v in (row or {}).items():
        nk = str(k).replace(".", "_").strip()
        out[nk] = v
    return out
