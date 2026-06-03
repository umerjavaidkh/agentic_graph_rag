"""Heuristic validators for document and structured RAG eval cases."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


_UNCERTAINTY_PHRASES = (
    "not mentioned",
    "not stated",
    "not specified",
    "not provided",
    "not found",
    "not in the",
    "no information",
    "no mention",
    "does not mention",
    "does not state",
    "does not specify",
    "does not provide",
    "doesn't mention",
    "don't have",
    "do not have",
    "not discussed",
    "not covered",
    "cannot find",
    "can't find",
    "unable to find",
    "unable to determine",
    "cannot determine",
    "no evidence",
    "not described",
    "not included",
    "not available",
    "not listed",
    "not documented",
    "there is no",
    "there are no",
    "outside the scope",
    "beyond the scope",
    "annex",
    "main body does not",
    "i don't",
    "i do not",
    "i cannot",
    "i can't",
)


@dataclass
class ValidationResult:
    passed: bool
    checks: list[str] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def add_pass(self, msg: str) -> None:
        self.checks.append(msg)

    def add_fail(self, msg: str) -> None:
        self.failures.append(msg)
        self.passed = False


def _norm(text: str) -> str:
    return (text or "").lower()


def _response_haystack(response: dict[str, Any], *, include_sources: bool) -> str:
    parts = [response.get("answer") or ""]
    if include_sources:
        for src in response.get("sources") or []:
            if not isinstance(src, dict):
                continue
            for key in ("text", "title", "snippet", "content"):
                val = src.get(key)
                if val:
                    parts.append(str(val))
    return "\n".join(parts)


def _count_matching_groups(text: str, groups: list[list[str]]) -> int:
    t = _norm(text)
    matched = 0
    for group in groups:
        if all(_norm(token) in t for token in group):
            matched += 1
    return matched


def _contains_any(text: str, groups: list[list[str]]) -> bool:
    return _count_matching_groups(text, groups) > 0


def _contains_uncertainty(text: str) -> bool:
    t = _norm(text)
    return any(p in t for p in _UNCERTAINTY_PHRASES)


def _presentation_chart_types(presentation: dict[str, Any] | None) -> list[str]:
    if not presentation:
        return []
    types: list[str] = []
    for block in presentation.get("blocks") or []:
        if block.get("type") == "chart":
            ct = block.get("chartType")
            if ct:
                types.append(str(ct))
    return types


def _presentation_table_rows(presentation: dict[str, Any] | None) -> int:
    if not presentation:
        return 0
    total = 0
    for block in presentation.get("blocks") or []:
        if block.get("type") == "table":
            rows = block.get("rows") or []
            total += len(rows)
    return total


def validate_response(case: dict[str, Any], response: dict[str, Any]) -> ValidationResult:
    """Validate one API response against a suite case's expect block."""
    expect = case.get("expect") or {}
    result = ValidationResult(passed=True)
    answer = response.get("answer") or ""
    route = response.get("route_tool")
    agent = response.get("agent")
    sources = response.get("sources") or []
    total_chunks = response.get("total_chunks")
    if total_chunks is None:
        total_chunks = len(sources)

    keyword_scope = _response_haystack(
        response,
        include_sources=expect.get("keywords_in_sources", True),
    )

    exp_route = expect.get("route_tool")
    if exp_route and exp_route != "any":
        if route == exp_route:
            result.add_pass(f"route_tool={route}")
        else:
            result.add_fail(f"route_tool expected {exp_route!r}, got {route!r}")

    exp_agent = expect.get("agent")
    if exp_agent and exp_agent != "any":
        if agent == exp_agent:
            result.add_pass(f"agent={agent}")
        else:
            result.add_fail(f"agent expected {exp_agent!r}, got {agent!r}")

    min_chars = expect.get("min_answer_chars", 0)
    if len(answer.strip()) >= min_chars:
        result.add_pass(f"answer length>={min_chars}")
    else:
        result.add_fail(f"answer too short ({len(answer.strip())} < {min_chars})")

    min_sources = expect.get("min_sources")
    if min_sources is not None:
        if total_chunks >= min_sources:
            result.add_pass(f"sources>={min_sources}")
        else:
            result.add_fail(f"sources expected>={min_sources}, got {total_chunks}")

    any_kw = expect.get("any_of_keywords")
    if any_kw:
        min_groups = int(expect.get("min_keyword_groups", 1))
        matched = _count_matching_groups(keyword_scope, any_kw)
        if matched >= min_groups:
            scope = "answer+sources" if expect.get("keywords_in_sources", True) else "answer"
            result.add_pass(f"keywords matched {matched}/{min_groups} groups ({scope})")
        else:
            result.add_fail(
                f"keywords matched {matched}/{min_groups} groups; expected {any_kw}"
            )

    none_of = expect.get("none_of_keywords") or []
    for bad in none_of:
        if _norm(bad) in _norm(answer):
            result.add_fail(f"forbidden phrase in answer: {bad!r}")
        else:
            result.add_pass(f"none_of: {bad!r}")

    if expect.get("anti_hallucination"):
        forbidden_hit = any(_norm(b) in _norm(answer) for b in none_of)
        if forbidden_hit:
            result.add_fail("anti_hallucination: forbidden content in answer")
        elif _contains_uncertainty(answer):
            result.add_pass("anti_hallucination: uncertainty in answer")
        elif any_kw and _count_matching_groups(keyword_scope, any_kw) >= int(
            expect.get("min_keyword_groups", 1)
        ):
            result.add_pass("anti_hallucination: scope keywords matched")
        elif none_of and not forbidden_hit:
            result.add_pass("anti_hallucination: no forbidden invented entities")
        else:
            result.add_fail(
                "anti_hallucination: answer should hedge, deny, or avoid inventing facts"
            )

    max_chars = expect.get("max_answer_chars")
    if max_chars is not None and len(answer) > max_chars:
        result.add_fail(f"answer too long ({len(answer)} > {max_chars})")

    regex = expect.get("answer_regex")
    if regex and not re.search(regex, answer, re.I | re.S):
        result.add_fail(f"answer did not match regex: {regex}")

    presentation = response.get("presentation")
    chart_types = _presentation_chart_types(presentation)
    if expect.get("forbid_chart"):
        if chart_types:
            result.add_fail(f"chart not expected, got chartType={chart_types}")
        else:
            result.add_pass("no chart in presentation")

    exp_charts = expect.get("expect_chart_types")
    if exp_charts:
        if any(ct in chart_types for ct in exp_charts):
            result.add_pass(f"chart type in {exp_charts!r} (got {chart_types})")
        else:
            result.add_fail(
                f"expected chart type one of {exp_charts!r}, got {chart_types or 'none'}"
            )

    min_rows = expect.get("min_table_rows")
    if min_rows is not None:
        row_count = _presentation_table_rows(presentation)
        if row_count >= min_rows:
            result.add_pass(f"table rows>={min_rows}")
        else:
            result.add_fail(f"table rows expected>={min_rows}, got {row_count}")

    max_rows = expect.get("max_table_rows")
    if max_rows is not None:
        row_count = _presentation_table_rows(presentation)
        if row_count <= max_rows:
            result.add_pass(f"table rows<={max_rows}")
        else:
            result.add_fail(f"table rows expected<={max_rows}, got {row_count}")

    return result
