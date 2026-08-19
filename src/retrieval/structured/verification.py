"""
retrieval/structured/verification.py — Correctness signal for the Text-to-Cypher path.

Two independent, additive checks — neither ever overrides the synthesized
`answer` text, they only annotate confidence:

  1. check_answer_sanity — free, deterministic, always runs.
  2. verify_with_llm      — gated by STRUCTURED_VERIFY_ENABLED, only invoked
                             when check_answer_sanity found nothing (catches
                             what regex patterns can't; avoids double LLM spend).

compute_confidence() is the single gate both retrieval/structured/graph.py
and streaming/structured.py call, so the gating logic lives in one place.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from ...shared.config.prompts import load_prompt
from ...shared.config.settings import STRUCTURED_VERIFY_ENABLED, STRUCTURED_VERIFY_MAX_TOKENS

MAX_VERIFY_ROWS = 8  # sample cap sent to the LLM cross-check
MAX_CELL_CHARS = 200  # per-value truncation to bound prompt size

_AVG_WORDS = re.compile(r"\b(average|avg|mean)\b", re.I)
_COUNT_WORDS = re.compile(r"\b(how many|count|number of)\b", re.I)
_RANK_WORDS = re.compile(
    r"\b(top|highest|lowest|most|least|best|worst|largest|smallest)\b", re.I
)

_HAS_AVG = re.compile(r"\bavg\s*\(", re.I)
_HAS_COUNT = re.compile(r"\bcount\s*\(|\bsize\s*\(", re.I)
_HAS_ORDER_BY = re.compile(r"\border\s+by\b", re.I)
_HAS_LIMIT = re.compile(r"\blimit\b", re.I)


def _is_blank_value(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, (int, float)):
        return value == 0
    if isinstance(value, str):
        return value.strip() == ""
    return False


def _is_empty_aggregate(rows: list[dict]) -> bool:
    """True for a single row where every value is zero/None/blank.

    Aggregating (SUM/AVG/COUNT-like) over zero matched base rows still
    yields one row with a zero/null aggregate, not zero rows — Cypher (like
    SQL) can't distinguish "nothing matched" from "genuinely zero" at that
    point. A named-entity lookup ("Acme's total sales") that silently
    matched nothing looks identical to a real zero unless flagged here.
    """
    if len(rows) != 1:
        return False
    row = rows[0]
    if not row:
        return False
    return all(_is_blank_value(v) for v in row.values())


def check_answer_sanity(question: str, cypher: str, rows: list[dict]) -> Optional[str]:
    """Cheap regex-only pattern-mismatch checks. Returns a reason string or None.

    No Neo4j, no LLM — pure string matching on the question and Cypher text.
    Deliberately excludes a "percentages sum to ~100%" check: rows are
    routinely a filtered/top-N subset, which would make that rule fire
    constantly against exactly the ranking queries below already target.
    """
    q = question or ""
    c = cypher or ""

    if _AVG_WORDS.search(q) and not _HAS_AVG.search(c):
        return "Question asks for an average/mean, but the query has no AVG(...)."

    if _COUNT_WORDS.search(q) and not _HAS_COUNT.search(c):
        return "Question asks for a count, but the query has no COUNT(...)/size(...)."

    if _RANK_WORDS.search(q) and not (_HAS_ORDER_BY.search(c) and _HAS_LIMIT.search(c)):
        return "Question asks for a top/highest/lowest result, but the query has no ORDER BY + LIMIT."

    # Skip for explicit count questions — a literal "how many X" legitimately
    # answering zero is common and not a sign anything went wrong.
    if not _COUNT_WORDS.search(q) and _is_empty_aggregate(rows):
        return (
            "The query returned a single row with every value zero/empty — "
            "likely no matching entity was found in the structured data, "
            "not a genuine zero result."
        )

    return None


def _extract_json(text: str) -> str:
    """Best-effort extraction of a JSON object from an LLM response.

    Duplicated from multistep/context.py rather than imported — that
    module's package pulls in a neo4j-dependent sibling (executor.py) via
    its __init__.py, which this module has no other reason to depend on.
    """
    t = (text or "").strip()
    if not t:
        return "{}"
    start = t.find("{")
    end = t.rfind("}")
    if start >= 0 and end > start:
        return t[start : end + 1]
    return t


def _truncate_rows_for_prompt(rows: list[dict], max_rows: int = MAX_VERIFY_ROWS) -> str:
    sample = rows[:max_rows]
    truncated = []
    for row in sample:
        clipped = {}
        for k, v in (row or {}).items():
            s = str(v)
            clipped[k] = s if len(s) <= MAX_CELL_CHARS else s[:MAX_CELL_CHARS] + "…"
        truncated.append(clipped)
    return json.dumps(truncated, default=str, ensure_ascii=False)


def verify_with_llm(
    question: str,
    cypher: str,
    rows: list[dict],
    *,
    provider,
    model: str,
    max_tokens: int,
) -> tuple[bool, Optional[str]]:
    """One small LLM call judging Cypher-vs-question logic.

    Fails OPEN — (True, None) — on any error. This is a cost-conscious
    add-on signal; a flaky LLM call must never manufacture a false
    "low confidence" and erode trust in the feature itself.
    """
    prompt = load_prompt(
        "structured_verify",
        question=question or "",
        cypher=cypher or "",
        rows=_truncate_rows_for_prompt(rows),
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
    chunks: list[dict],
    *,
    provider,
    model: str,
) -> tuple[bool, Optional[str]]:
    """low_confidence, confidence_note — the one gate structured answer paths call.

    Text2Cypher and multistep shape `chunks` differently: Text2Cypher makes
    one chunk per result row (`raw` is a row dict); multistep makes one
    chunk per *plan step* (`raw` is that step's own full row list). Both are
    normalized to a single `rows: list[dict]` representing the actual
    answer data before running the shared sanity checks — using the last
    chunk's cypher/rows for multistep, since the final step is the one that
    actually produced the answer (the first step is typically just an ID
    lookup feeding the rest of the plan).
    """
    cypher = chunks[-1].get("cypher") if chunks else None
    raw_values = [c.get("raw") for c in chunks if c.get("raw") is not None]

    is_multistep_shape = bool(raw_values) and isinstance(raw_values[0], list)
    if is_multistep_shape:
        rows = raw_values[-1] if isinstance(raw_values[-1], list) else []
    else:
        rows = raw_values

    confidence_note = check_answer_sanity(question, cypher or "", rows)

    # Multistep's UNWIND-chained steps short-circuit to zero rows (rather
    # than a fake single zero/null aggregate row) when an earlier step found
    # no matching entity — check_answer_sanity can't see this shape, so
    # check it here instead of teaching that shared, directly-tested
    # function about multistep's row semantics.
    if (
        confidence_note is None
        and is_multistep_shape
        and not rows
        and not _COUNT_WORDS.search(question or "")
    ):
        confidence_note = (
            "The final planned step returned no rows — likely no matching "
            "entity was found in the structured data, not a genuine empty result."
        )

    if confidence_note is None and STRUCTURED_VERIFY_ENABLED:
        is_valid, llm_reason = verify_with_llm(
            question,
            cypher or "",
            rows,
            provider=provider,
            model=model,
            max_tokens=STRUCTURED_VERIFY_MAX_TOKENS,
        )
        if not is_valid:
            confidence_note = llm_reason or "The generated query may not match the question's intent."

    return confidence_note is not None, confidence_note
