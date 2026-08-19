"""
tests/test_document_verification_unit.py — document (unstructured) answer verification.

Covers check_answer_sanity (free, rule-based), verify_with_llm (gated LLM
cross-check, fail-open), and compute_confidence (the shared gate) for the
document RAG path — mirrors test_structured_verification_unit.py.

Run with:
    python -m pytest tests/test_document_verification_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


from src.shared.model_providers.base import ModelProvider
from src.unstructured.retrieval.verification import (
    NO_CHUNKS_NOTE,
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


def _chunks(n: int, text: str = "Some supporting passage text.") -> list[dict]:
    return [{"id": str(i), "title": f"Section {i}", "text": text} for i in range(n)]


# ── check_answer_sanity ──────────────────────────────────────────────────────


def test_check_answer_sanity_enumeration_question_few_chunks():
    reason = check_answer_sanity("List all suppliers mentioned.", "Here are the suppliers: A, B.", _chunks(2), "graph_rag")
    assert reason is not None
    assert "list/enumerate" in reason


def test_check_answer_sanity_enumeration_question_enough_chunks():
    reason = check_answer_sanity("List all suppliers mentioned.", "Here are the suppliers: A, B, C.", _chunks(5), "graph_rag")
    assert reason is None


def test_check_answer_sanity_lexical_mode_few_chunks():
    reason = check_answer_sanity("What is the policy on X?", "The policy states X.", _chunks(1), "graph_rag_lexical")
    assert reason is not None
    assert "keyword-only" in reason


def test_check_answer_sanity_lexical_mode_enough_chunks():
    reason = check_answer_sanity("What is the policy on X?", "The policy states X.", _chunks(5), "graph_rag_lexical")
    assert reason is None


@pytest.mark.parametrize(
    "answer",
    [
        "I could not find that information in the provided sections.",
        "This is not mentioned in the ingested documents.",
        "No relevant information is available for this question.",
        "The document does not mention that detail.",
        "I am unable to locate the exact figure requested.",
        "That data is not in the document corpus; use structured data access.",
        # Regression: "does not cover" (and other synonymous hedge verbs)
        # slipped through the original mention/contain/include-only verb
        # list, and "This document..." (not "The document...") slipped
        # through the original the-only determiner -- together, this let
        # a hedge get treated as a confident, on-topic answer instead.
        # Verified live: this exact phrasing let a structured-data
        # question's RBAC-denial fallback silently substitute this hedge
        # for the clear "you don't have permission" message it should
        # have shown instead.
        "This document does not cover the specific topic of the top 5 products by sales revenue.",
        "This document does not cover Example 2.8 Direction of Motion.",
        "The document does not address that topic.",
        "The provided sections do not discuss this subject.",
    ],
)
def test_check_answer_sanity_hedge_language_flagged(answer):
    reason = check_answer_sanity("What is the exact revenue figure?", answer, _chunks(5), "graph_rag")
    assert reason is not None


def test_check_answer_sanity_healthy_case_no_flag():
    reason = check_answer_sanity(
        "What does the compliance policy say about whistleblowing?",
        "The compliance policy requires employees to report concerns to the ethics officer.",
        _chunks(5),
        "graph_rag",
    )
    assert reason is None


def test_check_answer_sanity_empty_inputs_no_crash():
    assert check_answer_sanity("", "", [], "") is None
    assert check_answer_sanity(None, None, [], None) is None


# ── verify_with_llm ───────────────────────────────────────────────────────────


def test_verify_with_llm_valid_response():
    provider = FakeModelProvider('{"valid": true}')
    is_valid, reason = verify_with_llm(
        "what does the policy say", "The policy says X.", _chunks(2),
        provider=provider, model="gpt-4o-mini", max_tokens=120,
    )
    assert is_valid is True
    assert reason is None


def test_verify_with_llm_invalid_response_returns_reason():
    provider = FakeModelProvider('{"valid": false, "reason": "answer invents a figure not in the passages"}')
    is_valid, reason = verify_with_llm(
        "what is the total budget", "The total budget is $5M.", _chunks(2),
        provider=provider, model="gpt-4o-mini", max_tokens=120,
    )
    assert is_valid is False
    assert reason == "answer invents a figure not in the passages"


def test_verify_with_llm_fails_open_on_malformed_json():
    provider = FakeModelProvider("not json at all")
    is_valid, reason = verify_with_llm(
        "question", "answer", _chunks(1),
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
        "question", "answer", _chunks(1),
        provider=RaisingProvider(), model="gpt-4o-mini", max_tokens=120,
    )
    assert is_valid is True
    assert reason is None


def test_verify_with_llm_truncates_chunk_sample():
    provider = FakeModelProvider('{"valid": true}')
    many_chunks = _chunks(20, text="x" * 500)
    verify_with_llm(
        "question", "answer", many_chunks,
        provider=provider, model="gpt-4o-mini", max_tokens=120,
    )
    prompt_sent = provider.calls[0]["messages"][0]["content"]
    # Only the first 8 chunks should appear (title "Section 8" onward must not).
    assert '"Section 7"' in prompt_sent
    assert '"Section 8"' not in prompt_sent
    assert "x" * 500 not in prompt_sent  # cell truncated well below 500 chars


# ── compute_confidence ────────────────────────────────────────────────────────


def test_compute_confidence_empty_chunks():
    low_confidence, note = compute_confidence(
        "anything", "I could not find relevant information.", [], "graph_rag", provider=None, model="m"
    )
    assert low_confidence is True
    assert note == NO_CHUNKS_NOTE


def test_compute_confidence_clean_answer_no_verify_enabled(monkeypatch):
    import src.unstructured.retrieval.verification as v

    monkeypatch.setattr(v, "DOCUMENT_VERIFY_ENABLED", False)
    low_confidence, note = compute_confidence(
        "What does the compliance policy say about whistleblowing?",
        "The compliance policy requires employees to report concerns to the ethics officer.",
        _chunks(5),
        "graph_rag",
        provider=None,
        model="gpt-4o-mini",
    )
    assert low_confidence is False
    assert note is None


def test_compute_confidence_rule_hit_skips_llm_call(monkeypatch):
    import src.unstructured.retrieval.verification as v

    monkeypatch.setattr(v, "DOCUMENT_VERIFY_ENABLED", True)
    provider = FakeModelProvider('{"valid": true}')

    low_confidence, note = compute_confidence(
        "List all suppliers mentioned.", "Here are the suppliers: A, B.", _chunks(2),
        "graph_rag", provider=provider, model="gpt-4o-mini",
    )

    assert low_confidence is True
    assert "list/enumerate" in note
    assert provider.calls == []  # rule already flagged it — no LLM call needed


def test_compute_confidence_llm_gate_flags_when_enabled(monkeypatch):
    import src.unstructured.retrieval.verification as v

    monkeypatch.setattr(v, "DOCUMENT_VERIFY_ENABLED", True)
    provider = FakeModelProvider('{"valid": false, "reason": "answer not backed by passages"}')

    low_confidence, note = compute_confidence(
        "What does the compliance policy say about whistleblowing?",
        "The compliance policy requires employees to report concerns to the ethics officer.",
        _chunks(5),
        "graph_rag",
        provider=provider,
        model="gpt-4o-mini",
    )

    assert low_confidence is True
    assert note == "answer not backed by passages"
    assert len(provider.calls) == 1


def test_compute_confidence_llm_gate_disabled_by_default(monkeypatch):
    import src.unstructured.retrieval.verification as v

    monkeypatch.setattr(v, "DOCUMENT_VERIFY_ENABLED", False)
    provider = FakeModelProvider('{"valid": false, "reason": "should never be seen"}')

    low_confidence, note = compute_confidence(
        "What does the compliance policy say about whistleblowing?",
        "The compliance policy requires employees to report concerns to the ethics officer.",
        _chunks(5),
        "graph_rag",
        provider=provider,
        model="gpt-4o-mini",
    )

    assert low_confidence is False
    assert note is None
    assert provider.calls == []
