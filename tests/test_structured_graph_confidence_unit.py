"""
tests/test_structured_graph_confidence_unit.py — _generate_structured_answer
verification wiring.

Proves: a clean question gets low_confidence=False/confidence_note=None; a
question whose Cypher doesn't match an implied aggregation gets flagged
low_confidence=True with a note — and, critically, the returned `answer`
text is identical either way (verification never rewrites the answer).

Run with:
    python -m pytest tests/test_structured_graph_confidence_unit.py -v
"""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# graph.py does `from langgraph.graph import END, StateGraph` and builds a
# real StateGraph/compile() at module level — stub both, permissively, so
# importing the real graph.py doesn't need the actual langgraph package.
for _n in ["langgraph", "langgraph.graph"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["langgraph.graph"].StateGraph = MagicMock()
sys.modules["langgraph.graph"].END = MagicMock()

# state.py needs auth.roles.UserContext -> real src.shared.auth.__init__ pulls in
# rbac_setup.py -> real `neo4j` (not installed in this env). Stub minimally.
if "neo4j" not in sys.modules:
    _stub_module("neo4j")
sys.modules["neo4j"].GraphDatabase = MagicMock()
sys.modules["neo4j"].Driver = MagicMock

for _n in ["src.shared.auth", "src.shared.auth.rbac_setup", "src.shared.auth.roles"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["src.shared.auth.rbac_setup"].GraphRBAC = MagicMock()
sys.modules["src.shared.auth.rbac_setup"].initialize_rbac_schema = MagicMock()
sys.modules["src.shared.auth.roles"].UserContext = MagicMock

# graph.py also does `retriever = StructuredRetriever()` at module level,
# which would otherwise need a live Neo4j driver, RBAC, schema provider,
# Cypher generator, and multistep planner/executor to construct. None of
# that is exercised by _generate_structured_answer (only retrieve_node uses
# `retriever`), so stub the class to a no-op.
#
# Only clear/re-stub the two modules this file actually needs control over
# (graph.py itself, and its retriever dependency) — NOT the whole
# src.retrieval.structured.* namespace. Other test files (e.g.
# test_structured_verification_unit.py) import sibling modules like
# verification.py directly and hold references to that module object;
# blanket-deleting it here would force a second, distinct module instance
# into existence, silently breaking monkeypatch targeting in those files
# once pytest's collection phase (which imports every test file before
# running any test) reaches this one.
for _mod_name in ("src.retrieval.structured.graph", "src.retrieval.structured.retriever"):
    if _mod_name in sys.modules:
        del sys.modules[_mod_name]
_retriever_stub = _stub_module("src.retrieval.structured.retriever")
_retriever_stub.StructuredRetriever = lambda *a, **k: MagicMock()

from src.retrieval.structured.graph import _generate_structured_answer
import src.retrieval.structured.graph as graph_mod


class FakeChatProvider:
    def __init__(self, content: str):
        self._content = content

    def chat_completion(self, model, messages, **kwargs):
        return MagicMock(choices=[MagicMock(message=MagicMock(content=self._content))])


def _chunks(cypher: str, rows: list[dict]) -> list[dict]:
    return [
        {
            "id": f"row_{i}",
            "title": f"Row {i}",
            "text": "\n".join(f"{k}: {v}" for k, v in row.items()),
            "raw": row,
            "score": 1.0,
            "cypher": cypher,
            "related": [],
        }
        for i, row in enumerate(rows)
    ]


def test_clean_question_is_high_confidence(monkeypatch):
    monkeypatch.setattr(graph_mod, "STRUCTURED_FAST_ANSWER", True)
    chunks = _chunks("MATCH (c:Customer) RETURN count(c) AS n", [{"n": 42}])
    state = {"strategy": "text2cypher"}

    result = _generate_structured_answer(state, {}, chunks, "how many customers do we have")

    assert result["low_confidence"] is False
    assert result["confidence_note"] is None


def test_mismatched_aggregation_is_flagged_but_answer_unchanged(monkeypatch):
    monkeypatch.setattr(graph_mod, "STRUCTURED_FAST_ANSWER", True)
    chunks = _chunks("MATCH (o:Order) RETURN o.total AS total", [{"total": 42.0}])
    state = {"strategy": "text2cypher"}
    question = "what is the average order value"

    flagged = _generate_structured_answer(state, {}, chunks, question)

    assert flagged["low_confidence"] is True
    assert flagged["confidence_note"] is not None
    assert "AVG" in flagged["confidence_note"]

    # Verification must never rewrite the answer text — compare against the
    # same fast-path build with confidence stripped out of the equation.
    from src.retrieval.structured.graph import _build_fast_structured_answer

    expected_answer = _build_fast_structured_answer(chunks, "text2cypher", question)
    assert flagged["answer"] == expected_answer


def test_llm_synthesis_path_carries_confidence(monkeypatch):
    monkeypatch.setattr(graph_mod, "STRUCTURED_FAST_ANSWER", False)  # force LLM synthesis branch
    monkeypatch.setattr(graph_mod, "provider", FakeChatProvider("Here are your results."))
    chunks = _chunks("MATCH (c:Customer) RETURN c.name AS name", [{"name": "Acme"}])
    state = {"strategy": "text2cypher"}

    result = _generate_structured_answer(state, {}, chunks, "list customers")

    assert result["answer"] == "Here are your results."
    assert result["low_confidence"] is False
    assert result["confidence_note"] is None


def test_error_branch_sets_confidence_note():
    chunks = [
        {
            "id": "error",
            "title": "Query Error",
            "text": "Generated Cypher failed: syntax error\nCypher: MATCH (n RETURN n",
            "score": 0.0,
            "related": [],
            "cypher": "MATCH (n RETURN n",
        }
    ]
    state = {"strategy": "text2cypher"}

    result = _generate_structured_answer(state, {}, chunks, "anything")

    assert result["low_confidence"] is True
    assert result["confidence_note"]


def test_clarification_branch_is_high_confidence():
    # Never reaches the verification step (no Cypher/rows to check).
    result = _generate_structured_answer(
        {}, {"mode": "needs_clarification"}, [], "ambiguous question"
    )
    assert result["low_confidence"] is False
    assert result.get("confidence_note") is None


def test_no_chunks_branch_flags_low_confidence_except_count_questions():
    # Zero rows for a non-aggregate query could mean genuinely empty data,
    # or (same root cause as the multistep/aggregate confidence check) a
    # named entity that only exists in ingested documents, not the
    # structured graph — flagged so the router's document fallback gets a
    # chance, except for a literal count question where zero is legitimate.
    result = _generate_structured_answer({}, {}, [], "no data question")
    assert result["low_confidence"] is True

    result2 = _generate_structured_answer({}, {}, [], "how many orders were placed")
    assert result2["low_confidence"] is False
