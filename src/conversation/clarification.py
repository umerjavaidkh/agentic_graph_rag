"""Deterministic clarification prompts when a query slot is ambiguous."""
from __future__ import annotations

import re
from typing import Any, Optional


def format_clarification_answer(
    prompt: str,
    options: list[dict[str, Any]],
    *,
    footer: str = "Reply with the document or option name (e.g. **Stratec**).",
) -> str:
    lines = [prompt.strip(), ""]
    for i, opt in enumerate(options, 1):
        label = opt.get("label") or opt.get("id") or f"Option {i}"
        detail = opt.get("detail") or ""
        lines.append(f"{i}. **{label}**" + (f" — {detail}" if detail else ""))
    if footer:
        lines.extend(["", footer.strip()])
    return "\n".join(lines)


def match_clarification_choice(
    question: str,
    options: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    """Match a short user reply to one clarification option."""
    if not question or not options:
        return None

    q = question.strip().lower()
    q_compact = re.sub(r"[^a-z0-9]", "", q)

    ord_m = re.match(r"^(?:#|option\s+)?(\d{1,2})\s*\.?$", q)
    if ord_m:
        idx = int(ord_m.group(1)) - 1
        if 0 <= idx < len(options):
            return options[idx]

    for opt in options:
        oid = (opt.get("id") or "").lower()
        label = (opt.get("label") or "").lower()
        label_compact = re.sub(r"[^a-z0-9]", "", label)
        if oid and (oid in q or q in oid or q_compact in re.sub(r"[^a-z0-9]", "", oid)):
            return opt
        if label_compact and (label_compact in q_compact or q_compact in label_compact):
            return opt
        if label and (label in q or q in label):
            return opt
        for alias in opt.get("aliases") or []:
            al = alias.lower()
            ac = re.sub(r"[^a-z0-9]", "", al)
            if ac and (ac == q_compact or ac in q_compact or q_compact in ac):
                return opt
            if al and (al == q or al in q or q in al):
                return opt
    return None


def document_option(doc_id: str, title: str, detail: str = "") -> dict[str, Any]:
    label = _friendly_doc_title(title, doc_id)
    aliases = [label.lower()]
    if "stratec" in label.lower():
        aliases.extend(["stratec", "setratec", "STRATEC"])
    if "go.data" in label.lower() or "godata" in re.sub(r"[^a-z0-9]", "", label.lower()):
        aliases.extend(["go data", "godata", "go.data", "Go.Data"])
    return {
        "id": doc_id,
        "label": label,
        "detail": detail,
        "aliases": list(dict.fromkeys(aliases)),
    }


def _friendly_doc_title(title: str, doc_id: str) -> str:
    t = (title or "").strip()
    opaque = (
        not t
        or t == doc_id
        or re.search(r"^[a-f0-9]{32}", t, re.I)
        or "_rag_document" in t
    )
    if not opaque:
        return t
    for key, name in (("stratec", "Stratec"), ("godata", "Go.Data"), ("go.data", "Go.Data")):
        if key in doc_id.lower():
            return name
    cleaned = re.sub(r"^doc_", "", doc_id)
    cleaned = re.sub(r"_rag_document.*$", "", cleaned)
    cleaned = cleaned.replace("_", " ").strip()
    if re.search(r"^[a-f0-9]{32}", cleaned, re.I):
        return cleaned[:48]
    return cleaned or doc_id[:48]


def normalize_clarification_reply(question: str) -> str:
    """Strip switch phrasing so 'now for Go.Data' matches option Go.Data."""
    q = question.strip()
    patterns = (
        r"^(?:now|ok|okay|instead|rather|switch|change)\s+(?:to\s+)?(?:for\s+)?",
        r"^(?:what\s+about|how\s+about)\s+",
        r"^(?:show\s+(?:me\s+)?(?:the\s+)?(?:toc|table\s+of\s+contents?|contents?)\s+(?:for|from|of)\s+)",
        r"^(?:give\s+me\s+)(?:the\s+)?(?:toc|table\s+of\s+contents?)\s+(?:for|from|of)\s+",
        r"^(?:and\s+)?(?:for\s+)",
    )
    for pat in patterns:
        q = re.sub(pat, "", q, flags=re.I).strip()
    q = re.sub(r"\s+document\s*$", "", q, flags=re.I)
    return q.strip(" ,.")
