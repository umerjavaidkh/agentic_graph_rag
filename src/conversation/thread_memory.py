"""
Remember only the last critical document turn per thread_id (in-memory).

Used to resolve short follow-ups ("Development Timeline", "show image on that page").
"""
from __future__ import annotations

import re
from typing import Any, Optional

from ..retrieval.unstructured.visual_retrieval import extract_visual_focus_terms
from .clarification import match_clarification_choice

# thread_id -> last critical turn snapshot
_store: dict[str, dict[str, Any]] = {}


def get_turn(thread_id: str) -> Optional[dict[str, Any]]:
    if not thread_id:
        return None
    return _store.get(thread_id)


def save_turn(thread_id: str, user_question: str, result: dict) -> None:
    if not thread_id:
        return
    snapshot = extract_critical_from_result(user_question, result)
    if snapshot:
        _store[thread_id] = snapshot


def clear_turn(thread_id: str) -> None:
    if thread_id:
        _store.pop(thread_id, None)


def extract_critical_from_result(user_question: str, result: dict) -> Optional[dict]:
    """Persist turns that help the next follow-up reply."""
    agent = result.get("agent")
    rc = result.get("retrieved_context") or {}
    mode = rc.get("mode") or ""

    if mode == "needs_clarification":
        return {
            "question": user_question.strip(),
            "agent": agent,
            "mode": mode,
            "pending_clarification": {
                "kind": rc.get("clarification_kind"),
                "options": rc.get("clarification_options") or [],
                "original_question": rc.get("original_question") or user_question.strip(),
            },
        }

    if agent not in (None, "unstructured", "hybrid", "structured"):
        return None

    sources = result.get("sources") or []
    query_type = result.get("query_type") or ""

    snapshot: dict[str, Any] = {
        "question": user_question.strip(),
        "query_type": query_type,
        "mode": mode,
        "parent_id": rc.get("parent_id"),
        "parent_title": rc.get("parent_title"),
        "pdf_page": rc.get("pdf_page"),
        "document_page": rc.get("document_page"),
        "agent": agent,
    }

    if not any([mode, query_type, snapshot["parent_id"], snapshot["pdf_page"]]):
        return None

    if mode == "subsection_tree" and snapshot.get("parent_title"):
        parent_title = snapshot["parent_title"]
        snapshot["children"] = [
            {"id": c["id"], "title": c.get("title", "")}
            for c in sources
            if c.get("id") and c.get("title") and c.get("title") != parent_title
        ]

    if mode in ("page_lookup", "page_text", "page_visual_list") and sources:
        top = sources[0]
        if snapshot.get("pdf_page") is None:
            snapshot["pdf_page"] = top.get("pdf_page")
        snapshot["last_page_title"] = top.get("title")
        if mode == "page_visual_list":
            snapshot["figures"] = [
                {
                    "id": c.get("id"),
                    "title": c.get("title"),
                    "image_key": c.get("image_key"),
                    "region_kind": c.get("region_kind"),
                }
                for c in sources
                if c.get("image_key")
            ]

    if mode == "section_detail" and sources:
        snapshot["focus_section_id"] = sources[0].get("id")
        snapshot["focus_section_title"] = sources[0].get("title")

    return snapshot


def resolve_follow_up(question: str, prior: Optional[dict[str, Any]]) -> dict[str, Any]:
    """
    If the new message is a short follow-up, rewrite the question.
    """
    base = {
        "question": question,
        "focus_section_id": None,
        "parent_section_id": None,
        "document_id": None,
        "use_prior": False,
        "follow_up_kind": None,
    }
    if not prior:
        return base

    q = question.strip()
    q_lower = q.lower()

    pending = prior.get("pending_clarification")
    if pending:
        # Only treat very short messages as clarification replies.
        # This prevents hijacking new questions like: '"Gnocchi..." total order count'
        # which might contain words like "total".
        q_short = len(q) <= 40 and len(q.split()) <= 4
        options = pending.get("options") or []
        choice = match_clarification_choice(question, options) if q_short else None
        if choice:
            kind = pending.get("kind") or ""
            orig = pending.get("original_question") or base["question"]
            # Keep rewrite simple and explicit; avoid stuffing definitions into the question
            # because it can confuse Text-to-Cypher generation.
            if kind == "structured_order_price":
                cid = (choice.get("id") or "").strip()

                # Try to preserve the user's country filter in a robust way.
                # Example: "avg order price for Germany" -> country_phrase="Germany"
                country_phrase = None
                m = re.search(r"\bfor\s+([a-zA-Z][a-zA-Z\s\.\-]{1,60})\??\s*$", orig, re.I)
                if m:
                    country_phrase = m.group(1).strip()
                if not country_phrase:
                    m2 = re.search(r"\bin\s+([a-zA-Z][a-zA-Z\s\.\-]{1,60})\??\s*$", orig, re.I)
                    if m2:
                        country_phrase = m2.group(1).strip()

                where = f" for {country_phrase}" if country_phrase else ""

                if cid == "order_total":
                    rewritten = f"Calculate the average order total{where}. Define order total as sum of line items per order (unitPrice × quantity × (1 - discount))."
                elif cid == "freight":
                    rewritten = f"Calculate the average freight (shipping cost) per order{where}."
                elif cid == "unit_price":
                    rewritten = f"Calculate the average line-item unit price{where} (avg of ORDER_CONTAINS.unitPrice for matching orders)."
                else:
                    rewritten = f"Clarify and answer the original question{where}: {orig}"
                return {
                    **base,
                    "question": rewritten,
                    "use_prior": True,
                    "follow_up_kind": "structured_clarification",
                }
            if kind == "document_choice":
                label = (choice.get("label") or choice.get("id") or "").strip()
                rewritten = f"{orig} from {label} document" if label else orig
                return {
                    **base,
                    "question": rewritten,
                    "document_id": choice.get("id"),
                    "use_prior": True,
                    "follow_up_kind": "clarification_document",
                }

    children = prior.get("children") or []
    ord_m = re.match(r"^(?:#|item\s+)?(\d{1,2})\s*\.?$", q_lower)
    if ord_m and children:
        idx = int(ord_m.group(1)) - 1
        if 0 <= idx < len(children):
            child = children[idx]
            return _subsection_detail_resolution(prior, child)

    if children and len(q) < 100 and not _looks_like_new_topic(q_lower):
        for child in children:
            title = (child.get("title") or "").strip()
            if not title or len(title) < 3:
                continue
            tl = title.lower()
            if tl == q_lower or tl in q_lower or q_lower in tl:
                return _subsection_detail_resolution(prior, child)

    pdf_p = prior.get("pdf_page")
    focus_terms = extract_visual_focus_terms(question)
    if pdf_p is not None and len(q) < 100 and focus_terms and _is_page_follow_up(q_lower):
        label = focus_terms[0]
        rewritten = f"Show only the {label} image on PDF page {pdf_p}"
        return {
            **base,
            "question": rewritten,
            "use_prior": True,
            "follow_up_kind": "page_visual_focus",
            "pdf_page": pdf_p,
        }

    if pdf_p is not None and len(q) < 80 and _is_page_follow_up(q_lower):
        if re.search(r"\b(image|picture|photo|figure)\b", q_lower):
            rewritten = f"Show the image from PDF page {pdf_p}"
        elif re.search(r"\b(text|content|words)\b", q_lower):
            rewritten = f"Give me all the text from PDF page {pdf_p}"
        else:
            rewritten = f"Tell me about PDF page {pdf_p}: {question}"
        return {
            **base,
            "question": rewritten,
            "use_prior": True,
            "follow_up_kind": "page",
            "pdf_page": pdf_p,
        }

    return base


def _subsection_detail_resolution(prior: dict, child: dict) -> dict:
    title = child.get("title", "")
    parent_title = prior.get("parent_title") or "the parent section"
    rewritten = (
        f'Provide a detailed explanation of the document section "{title}" '
        f'under "{parent_title}". Use only the text from that subsection in the document.'
    )
    return {
        "question": rewritten,
        "focus_section_id": child.get("id"),
        "parent_section_id": prior.get("parent_id"),
        "document_id": None,
        "use_prior": True,
        "follow_up_kind": "subsection_detail",
    }


def _looks_like_new_topic(q_lower: str) -> bool:
    """Avoid hijacking a clearly new full question."""
    if re.search(r"\b\d+(?:\.\d+)+\.?\s+\w", q_lower):
        return True
    if re.search(r"\btop\s+\d+\b", q_lower) and "product" in q_lower:
        return True
    if len(q_lower) > 120:
        return True
    return False


def _is_page_follow_up(q_lower: str) -> bool:
    if re.search(r"\bpage\s+\d+", q_lower):
        return False
    if re.search(r"\b(?:only|just)\b", q_lower) and re.search(
        r"\b(logo|logos|icon|icons|image|picture|photo)\b", q_lower
    ):
        return True
    return bool(
        re.search(
            r"\b(that|this|same)\s+page\b"
            r"|\b(on|from)\s+(that|this)\s+page\b"
            r"|\b(the\s+)?(image|text|page)\b",
            q_lower,
        )
    )
