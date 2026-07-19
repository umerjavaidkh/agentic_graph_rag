"""
retrieval/unstructured/verification.py — Correctness signal for the document RAG path.

Mirrors retrieval/structured/verification.py's shape:

  1. check_answer_sanity — free, deterministic, always runs.
  2. verify_with_llm      — gated by DOCUMENT_VERIFY_ENABLED, only invoked
                             when check_answer_sanity found nothing.

compute_confidence() is the single gate both retrieval/unstructured/graph.py
and streaming/document.py call, so the gating logic lives in one place.

Chunk relevance `score` is an unbounded, uncalibrated composite heuristic
(can exceed 1.0) — deliberately not used here as an absolute threshold.
Signals used instead: chunk count, retrieval `mode` (lexical-only fallback),
question-shape mismatches, and self-reported hedging in the answer text.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ...config.prompts import load_prompt
from ...config.settings import DOCUMENT_VERIFY_ENABLED, DOCUMENT_VERIFY_MAX_TOKENS
from .query_intent import is_enumeration_question

MAX_VERIFY_CHUNKS = 8  # sample cap sent to the LLM cross-check
MAX_CELL_CHARS = 300  # per-chunk truncation to bound prompt size

NO_CHUNKS_NOTE = "No relevant passages were found in the ingested documents for this question."

_ENUMERATION_MIN_CHUNKS = 3  # below this, a "list all X" answer is likely incomplete

_HEDGE_RE = re.compile(
    r"\b(i (?:could not|couldn't|did not|don't|do not) find|"
    r"not (?:mentioned|found|present|available) in the (?:provided|document|ingested)|"
    r"no (?:relevant )?information (?:is |was )?(?:available|found)|"
    r"the (?:document|text|provided sections?) does(?:n'?t| not) (?:mention|contain|include)|"
    r"unable to (?:find|locate|determine)|"
    r"not in the document corpus|"
    r"use structured data access|"
    r"there (?:is|are) no .{0,80} (?:mentioned|found|provided|available|specified) in|"
    r"(?:no|not any) .{0,40} (?:is|are|was|were) mentioned in the (?:provided|document|ingested))\b",
    re.I,
)


def check_answer_sanity(question: str, answer: str, chunks: list[dict], mode: str) -> Optional[str]:
    """Cheap regex/count-only pattern-mismatch checks. Returns a reason string or None.

    No LLM. Assumes chunks is non-empty (compute_confidence special-cases
    the zero-chunk case before calling this).
    """
    n = len(chunks)

    if is_enumeration_question(question) and n < _ENUMERATION_MIN_CHUNKS:
        return (
            f"Question asks to list/enumerate all items, but only {n} supporting "
            "passage(s) were retrieved — the list may be incomplete."
        )

    if mode == "graph_rag_lexical" and n < _ENUMERATION_MIN_CHUNKS:
        return (
            "Retrieval fell back to keyword-only matching (no semantic match found) "
            "with few results — the answer may be incomplete."
        )

    if _HEDGE_RE.search(answer or ""):
        return "The generated answer itself indicates the information may not be fully present in the retrieved sections."

    return None


def _extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object from an LLM response.

    Duplicated from structured/verification.py rather than shared — that
    module deliberately avoids importing from a neo4j-dependent sibling
    package, and this module has no other reason to depend on it either.
    """
    t = (text or "").strip()
    if not t:
        return "{}"
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        return t[start : end + 1]
    return t


def _truncate_chunks_for_prompt(chunks: list[dict], max_chunks: int = MAX_VERIFY_CHUNKS) -> str:
    sample = chunks[:max_chunks]
    truncated = []
    for c in sample:
        text = str(c.get("text") or "")
        clipped = text if len(text) <= MAX_CELL_CHARS else text[:MAX_CELL_CHARS] + "…"
        truncated.append({"title": c.get("title"), "text": clipped})
    return json.dumps(truncated, default=str, ensure_ascii=False)


def verify_with_llm(
    question: str,
    answer: str,
    chunks: list[dict],
    *,
    provider,
    model: str,
    max_tokens: int,
) -> tuple[bool, Optional[str]]:
    """One small LLM call judging whether the answer is well-supported by the chunks.

    Fails OPEN — (True, None) — on any error. Cost-conscious add-on signal;
    a flaky LLM call must never manufacture a false "low confidence".
    """
    prompt = load_prompt(
        "document_verify",
        question=question or "",
        answer=answer or "",
        chunks=_truncate_chunks_for_prompt(chunks),
    )
    try:
        resp = provider.chat_completion(
            model=model,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
        )
        raw = (resp.choices[0].message.content or "").strip()
        data = json.loads(_extract_json(raw))
        is_valid = bool(data.get("valid", True))
        reason = data.get("reason") if not is_valid else None
        return is_valid, (str(reason) if reason else None)
    except Exception:
        return True, None


def compute_confidence(
    question: str,
    answer: str,
    chunks: list[dict],
    mode: str,
    *,
    provider,
    model: str,
) -> tuple[bool, Optional[str]]:
    """low_confidence, confidence_note — the one gate document answer paths call."""
    if not chunks:
        return True, NO_CHUNKS_NOTE

    confidence_note = check_answer_sanity(question, answer, chunks, mode)
    if confidence_note is None and DOCUMENT_VERIFY_ENABLED:
        is_valid, llm_reason = verify_with_llm(
            question,
            answer,
            chunks,
            provider=provider,
            model=model,
            max_tokens=DOCUMENT_VERIFY_MAX_TOKENS,
        )
        if not is_valid:
            confidence_note = llm_reason or "The generated answer may not be fully supported by the retrieved sections."

    return confidence_note is not None, confidence_note
