"""Stable question-pattern keys for feedback bucketing."""
from __future__ import annotations

import hashlib
import re


def question_hash(question: str) -> str:
    normalized = re.sub(r"\s+", " ", (question or "").strip().lower())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def retrieval_pattern(question: str, *, agent: str = "") -> str:
    """Bucket key from intent flags (not verbatim question text)."""
    from ..retrieval.structured.query_intent import likely_needs_multistep_plan
    from ..retrieval.unstructured.query_intent import (
        is_enumeration_question,
        is_fact_lookup_question,
        is_page_question,
        is_synthesis_question,
        is_toc_question,
        is_visual_page_question,
    )

    q = question or ""
    flags: list[str] = []
    if agent:
        flags.append(f"agent:{agent}")
    if is_toc_question(q):
        flags.append("toc")
    if is_page_question(q):
        flags.append("page")
    if is_visual_page_question(q):
        flags.append("visual_page")
    if is_synthesis_question(q):
        flags.append("synthesis")
    if is_enumeration_question(q):
        flags.append("enumeration")
    if is_fact_lookup_question(q):
        flags.append("fact_lookup")
    if likely_needs_multistep_plan(q):
        flags.append("structured_multistep")
    if not flags:
        flags.append("general")
    return "|".join(flags)


def pattern_hash(pattern: str) -> str:
    return hashlib.sha256((pattern or "general").encode("utf-8")).hexdigest()[:16]
