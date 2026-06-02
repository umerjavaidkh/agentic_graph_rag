"""Conversation clarification helpers (generic, deterministic).

This module is intentionally small and domain-agnostic: it matches a short user
reply (e.g. "1", "freight", "order total") to an option list emitted by a retriever.
"""

from __future__ import annotations

import re
from typing import Any, Optional


def format_clarification_answer(
    prompt: str,
    options: list[dict[str, Any]],
    *,
    footer: str = "Reply with an option number or name (e.g. `1` or `freight`).",
) -> str:
    lines = [prompt.strip(), ""]
    for i, opt in enumerate(options, 1):
        label = opt.get("label") or opt.get("id") or f"Option {i}"
        detail = opt.get("detail") or ""
        lines.append(f"{i}. **{label}**" + (f" — {detail}" if detail else ""))
    if footer:
        lines.extend(["", footer.strip()])
    return "\n".join(lines)


def normalize_clarification_reply(text: str) -> str:
    """Light normalization so 'go with option 1' still matches '1'."""
    q = (text or "").strip().lower()
    q = re.sub(r"^(?:go\s+with|choose|pick|option)\s+", "", q).strip()
    return q.strip(" ,.")


def match_clarification_choice(
    reply: str,
    options: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not reply or not options:
        return None

    q = normalize_clarification_reply(reply)
    q_compact = re.sub(r"[^a-z0-9]", "", q)

    ord_m = re.match(r"^(?:#\s*)?(\d{1,2})\s*\.?$", q)
    if ord_m:
        idx = int(ord_m.group(1)) - 1
        if 0 <= idx < len(options):
            return options[idx]

    for opt in options:
        oid = str(opt.get("id") or "").strip().lower()
        label = str(opt.get("label") or "").strip().lower()
        aliases = [str(a).strip().lower() for a in (opt.get("aliases") or [])]

        for cand in [oid, label, *aliases]:
            if not cand:
                continue
            cand_compact = re.sub(r"[^a-z0-9]", "", cand)
            # Be strict to avoid hijacking new full questions. Only match:
            # - exact equality, or
            # - compact equality (ignoring punctuation/spaces), or
            # - whole-word match for multi-word candidates (e.g. "order total").
            if cand == q:
                return opt
            if cand_compact and cand_compact == q_compact:
                return opt
            # Whole word / phrase boundary match (prevents matching "total" inside "total order count").
            if len(cand) >= 4 and re.search(rf"(^|\\b){re.escape(cand)}(\\b|$)", q):
                return opt

    return None

