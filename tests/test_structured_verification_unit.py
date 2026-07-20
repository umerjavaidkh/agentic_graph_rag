"""
tests/test_structured_verification_unit.py — structured answer verification.

Covers check_answer_sanity (free, rule-based), verify_with_llm (gated LLM
cross-check, fail-open), and compute_confidence (the shared gate).

Run with:
    python -m pytest tests/test_structured_verification_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.model_providers.base import ModelProvider
from src.retrieval.structured.verification import (
    check_answer_sanity,
    compute_confidence,
    verify_with_llm,
)


class FakeModelProvider(ModelProvider):
    def __init__(self, content: str):
        self._content = content
        self.calls: list[dict] = []

    def chat_completion(self, model, messages, **kwargs):
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return MagicMock(choices=[MagicMock(message=MagicMock(content=self._content))])

    def embeddings(self, model, input, **kwargs):
        raise AssertionError("embeddings should not be called")


# ── check_answer_sanity ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "question,cypher,expect_hit",
    [
        ("What is the average order value?", "MATCH (o:Order) RETURN o.total", True),
        ("What is the average order value?", "MATCH (o:Order) RETURN avg(o.total)", False),
        ("What's the mean delivery time?", "MATCH (o) RETURN AVG(o.days)", False),
        ("How many customers do we have?", "MATCH (c:Customer) RETURN c.name", True),
        ("How many customers do we have?", "MATCH (c:Customer) RETURN count(c)", False),
        ("Number of orders placed in 1997?", "MATCH (o:Order) RETURN size(collect(o))", False),
        (
            "What are the top 5 products by revenue?",
            "MATCH (p:Product) RETURN p.name",
            True,
        ),
        (
            "What are the top 5 products by revenue?",
            "MATCH (p:Product) RETURN p.name ORDER BY p.revenue DESC LIMIT 5",
            False,
        ),
        (
            "What are the top 5 products by revenue?",
            "MATCH (p:Product) RETURN p.name ORDER BY p.revenue DESC",
            True,
        ),
        ("List all suppliers.", "MATCH (s:Supplier) RETURN s.name", False),
    ],
)
def test_check_answer_sanity(question, cypher, expect_hit):
    reason = check_answer_sanity(question, cypher, [])
    if expect_hit:
        assert reason is not None
    else:
        assert reason is None


def test_check_answer_sanity_empty_inputs_no_crash():
    assert check_answer_sanity("", "", []) is None
    assert check_answer_sanity(None, None, []) is None


# ── verify_with_llm ───────────────────────────────────────────────────────────


def test_verify_with_llm_valid_response():
    provider = FakeModelProvider('{"valid": true}')
    is_valid, reason = verify_with_llm(
        "how many orders", "MATCH (o) RETURN count(o)", [],
        provider=provider, model="gpt-4o-mini", max_tokens=120,
    )
    assert is_valid is True
    assert reason is None


def test_verify_with_llm_invalid_response_returns_reason():
    provider = FakeModelProvider('{"valid": false, "reason": "wrong join direction"}')
    is_valid, reason = verify_with_llm(
        "how many orders per customer", "MATCH (o)-[:PLACED]->(c) RETURN c", [],
        provider=provider, model="gpt-4o-mini", max_tokens=120,
    )
    assert is_valid is False
    assert reason == "wrong join direction"


def test_verify_with_llm_fails_open_on_malformed_json():
    provider = FakeModelProvider("not json at all")
    is_valid, reason = verify_with_llm(
        "question", "MATCH (n) RETURN n", [],
        provider=provider, model="gpt-4o-mini", max_tokens=120,
    )
    assert is_valid is True
    assert reason is None


def test_verify_with_llm_fails_open_on_provider_exception():
    class RaisingProvider(ModelProvider):
        def chat_completion(self, model, messages, **kwargs):
            raise RuntimeError("API down")

        def embeddings(self, model, input, **kwargs):
            raise AssertionError

    is_valid, reason = verify_with_llm(
        "question", "MATCH (n) RETURN n", [],
        provider=RaisingProvider(), model="gpt-4o-mini", max_tokens=120,
    )
    assert is_valid is True
    assert reason is None


def test_verify_with_llm_truncates_row_sample():
    provider = FakeModelProvider('{"valid": true}')
    many_rows = [{"id": i, "text": "x" * 500} for i in range(20)]
    verify_with_llm(
        "question", "MATCH (n) RETURN n", many_rows,
        provider=provider, model="gpt-4o-mini", max_tokens=120,
    )
    prompt_sent = provider.calls[0]["messages"][0]["content"]
    # Only the first 8 rows should appear (row ids 8..19 must not).
    assert '"id": "7"' in prompt_sent
    assert '"id": "8"' not in prompt_sent
    assert "x" * 500 not in prompt_sent  # cell truncated well below 500 chars


# ── compute_confidence ────────────────────────────────────────────────────────


def test_compute_confidence_clean_query_no_verify_enabled(monkeypatch):
    import src.retrieval.structured.verification as v

    monkeypatch.setattr(v, "STRUCTURED_VERIFY_ENABLED", False)
    chunks = [{"cypher": "MATCH (c:Customer) RETURN count(c)", "raw": {"count": 5}}]
    low_confidence, note = compute_confidence(
        "how many customers", chunks, provider=None, model="gpt-4o-mini"
    )
    assert low_confidence is False
    assert note is None


def test_compute_confidence_rule_hit_skips_llm_call(monkeypatch):
    import src.retrieval.structured.verification as v

    monkeypatch.setattr(v, "STRUCTURED_VERIFY_ENABLED", True)
    provider = FakeModelProvider('{"valid": true}')
    chunks = [{"cypher": "MATCH (o:Order) RETURN o.total", "raw": {"total": 5}}]

    low_confidence, note = compute_confidence(
        "what is the average order value", chunks, provider=provider, model="gpt-4o-mini"
    )

    assert low_confidence is True
    assert "AVG" in note
    assert provider.calls == []  # rule already flagged it — no LLM call needed


def test_compute_confidence_llm_gate_flags_when_enabled(monkeypatch):
    import src.retrieval.structured.verification as v

    monkeypatch.setattr(v, "STRUCTURED_VERIFY_ENABLED", True)
    provider = FakeModelProvider('{"valid": false, "reason": "filters do not match"}')
    chunks = [{"cypher": "MATCH (c:Customer) RETURN c.name", "raw": {"name": "Acme"}}]

    low_confidence, note = compute_confidence(
        "list customers in Germany", chunks, provider=provider, model="gpt-4o-mini"
    )

    assert low_confidence is True
    assert note == "filters do not match"
    assert len(provider.calls) == 1


def test_compute_confidence_empty_chunks():
    low_confidence, note = compute_confidence("anything", [], provider=None, model="m")
    assert low_confidence is False
    assert note is None
