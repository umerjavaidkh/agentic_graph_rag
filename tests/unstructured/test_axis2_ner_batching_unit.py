"""
tests/test_axis2_ner_batching_unit.py — Axis2's batched NER extraction.

Regression: _extract_entities made one LLM call per Section/Page node, so a
single 7,165-node document (a physics textbook, with thousands of such
nodes) burned an entire 10,000-request daily OpenAI quota by itself —
reproduced live twice, each time exhausting the quota mid-ingestion with no
error surfaced (the per-call `except Exception: return node.id, []` silently
swallows 429s, so the job just crawls forward instead of failing loudly).

Fix: batch AXIS2_NER_BATCH_SIZE nodes' text into a single call, prompting for
a JSON object keyed by each excerpt's local index. These tests guard: call
count actually drops by ~batch size, entities map back to the correct node
(not misaligned), a batch that fails entirely degrades to empty entities for
just that batch (not a crash), and a response missing some indices still
assigns empty (not KeyError) for those.

Run with:
    python -m pytest tests/test_axis2_ner_batching_unit.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest


from src.shared.config.settings import AXIS2_NER_BATCH_SIZE
from src.unstructured.models import DKGNode, NodeType
from src.unstructured.semantic.axis2 import Axis2Builder


def _section_node(node_id: str, text: str = "some physics text") -> DKGNode:
    node = DKGNode(id=node_id, type=NodeType.SECTION, title=node_id, text=text, order=0)
    node.embedding = None
    node.entities = []
    return node


def _builder() -> Axis2Builder:
    builder = Axis2Builder.__new__(Axis2Builder)
    builder.client = MagicMock()
    return builder


def _batch_response(entity_map: dict) -> MagicMock:
    return MagicMock(
        choices=[MagicMock(message=MagicMock(content=json.dumps(entity_map)))]
    )


def test_call_count_drops_by_roughly_batch_size():
    """The whole point of batching: N nodes should cost ceil(N/batch) calls,
    not N calls. This is what actually fixes the daily-quota exhaustion."""
    n = AXIS2_NER_BATCH_SIZE * 5  # exactly 5 full batches
    nodes = [_section_node(f"n{i}") for i in range(n)]

    builder = _builder()
    builder.client.chat_completion.return_value = _batch_response(
        {str(i): ["entity"] for i in range(AXIS2_NER_BATCH_SIZE)}
    )

    builder._extract_entities(nodes)

    assert builder.client.chat_completion.call_count == 5
    assert builder.client.chat_completion.call_count < n


def test_entities_map_back_to_the_correct_node_not_misaligned():
    # Text must actually contain the mocked entities -- _extract_entities
    # now grounds each returned entity against its own node's text before
    # keeping it (deterministic substring check, catches hallucination).
    nodes = [
        _section_node("n0", text="alpha particle decay"),
        _section_node("n1", text="beta decay and gamma rays"),
        _section_node("n2", text="no entities on this page"),
    ]
    builder = _builder()
    builder.client.chat_completion.return_value = _batch_response({
        "0": ["alpha"],
        "1": ["beta", "gamma"],
        "2": [],
    })

    result = builder._extract_entities(nodes)

    by_id = {n.id: n.entities for n in result}
    assert by_id["n0"] == ["alpha"]
    assert by_id["n1"] == ["beta", "gamma"]
    assert by_id["n2"] == []


def test_batch_call_failure_yields_empty_entities_not_a_crash():
    nodes = [_section_node(f"n{i}") for i in range(3)]
    builder = _builder()
    builder.client.chat_completion.side_effect = RuntimeError("429 rate limited")

    result = builder._extract_entities(nodes)  # must not raise

    assert all(n.entities == [] for n in result)


def test_response_missing_an_index_assigns_empty_not_keyerror():
    nodes = [
        _section_node("n0", text="alpha particle decay"),
        _section_node("n1", text="no entities on this page"),
        _section_node("n2", text="gamma ray burst"),
    ]
    builder = _builder()
    # Model only returned entries for indices 0 and 2, skipping 1.
    builder.client.chat_completion.return_value = _batch_response({
        "0": ["alpha"],
        "2": ["gamma"],
    })

    result = builder._extract_entities(nodes)

    by_id = {n.id: n.entities for n in result}
    assert by_id["n0"] == ["alpha"]
    assert by_id["n1"] == []  # missing index -> empty, not an error
    assert by_id["n2"] == ["gamma"]


def test_malformed_json_response_yields_empty_entities():
    nodes = [_section_node("n0")]
    builder = _builder()
    builder.client.chat_completion.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="not valid json"))]
    )

    result = builder._extract_entities(nodes)

    assert result[0].entities == []


# ── long-node chunking ───────────────────────────────────────────────────────
# Regression: a real 264-page 10-K's page (4,074 chars) lost every entity past
# its first ~1,200 chars under a flat per-node truncation ("OPEC", "Russia",
# "Chevron's Strategic Direction" never reached the model at all). Fixed by
# splitting long node text into _NER_CHUNK_CHARS-sized chunks, each its own
# NER unit, merged back together per node afterward.


def test_long_node_text_is_chunked_not_truncated():
    """A node whose text spans two chunks must get entities grounded in
    BOTH halves, not just the first _NER_CHUNK_CHARS characters. Both
    chunks fit in one batch call here (small AXIS2_NER_BATCH_SIZE headroom)
    -- the point under test is that the second half is sent to the model at
    all, not how many calls that takes."""
    chunk_size = Axis2Builder._NER_CHUNK_CHARS
    first_half = ("alpha particle decay. " * 100)[:chunk_size]
    second_half = "beta decay and gamma ray emission follow shortly after."
    text = first_half + " " + second_half

    node = _section_node("n0", text=text)
    builder = _builder()

    def _dispatch(*, messages, **_kwargs):
        user_content = messages[1]["content"]
        if "gamma" in user_content:
            return _batch_response({"0": ["alpha"], "1": ["gamma"]})
        return _batch_response({"0": ["alpha"]})

    builder.client.chat_completion.side_effect = _dispatch

    result = builder._extract_entities([node])

    sent = " ".join(
        call.kwargs["messages"][1]["content"] for call in builder.client.chat_completion.call_args_list
    )
    assert "gamma ray emission" in sent  # the second half actually reached the model
    assert set(result[0].entities) == {"alpha", "gamma"}


def test_chunk_results_from_different_batches_are_merged_not_overwritten(monkeypatch):
    """Chunks of the same long node can land in different batches (each
    processed and merged independently) -- force that by capping the batch
    size to 1 chunk/call, then confirm the second batch's result for this
    node doesn't silently overwrite the first's."""
    monkeypatch.setattr("src.unstructured.semantic.axis2.AXIS2_NER_BATCH_SIZE", 1)
    chunk_size = Axis2Builder._NER_CHUNK_CHARS
    long_text = ("word " * (chunk_size // 4)) + "uniqueterm"
    node = _section_node("n0", text=long_text)
    assert len(long_text) > chunk_size  # sanity: this really does chunk

    builder = _builder()

    def _dispatch(*, messages, **_kwargs):
        user_content = messages[1]["content"]
        if "uniqueterm" in user_content:
            return _batch_response({"0": ["uniqueterm"]})
        return _batch_response({"0": ["word"]})

    builder.client.chat_completion.side_effect = _dispatch

    result = builder._extract_entities([node])

    assert builder.client.chat_completion.call_count == 2
    assert "uniqueterm" in result[0].entities
    assert "word" in result[0].entities


def test_short_node_text_produces_a_single_chunk_as_before():
    # Below _NER_CHUNK_CHARS: exactly the pre-fix behavior, one call.
    node = _section_node("n0", text="alpha particle decay")
    builder = _builder()
    builder.client.chat_completion.return_value = _batch_response({"0": ["alpha"]})

    builder._extract_entities([node])

    assert builder.client.chat_completion.call_count == 1


def test_merged_entities_deduped_case_insensitively_across_chunks():
    chunk_size = Axis2Builder._NER_CHUNK_CHARS
    text = ("Alpha decay context. " * 100)[:chunk_size] + " more text about alpha decay again"
    node = _section_node("n0", text=text)
    builder = _builder()
    builder.client.chat_completion.return_value = _batch_response({"0": ["Alpha decay"]})

    result = builder._extract_entities([node])

    entities_lower = [e.lower() for e in result[0].entities]
    assert entities_lower.count("alpha decay") == 1


def test_batch_json_failure_recovers_via_split_retry_instead_of_losing_everything(monkeypatch):
    """The overflow/truncation failure mode this fix targets: a batch call's
    JSON response is malformed (simulating a max_tokens overflow from
    several entity-dense chunks combined), but the two halves succeed on
    retry -- entities must be recovered from the successful halves, not
    lost for the whole original batch. Pins batch size to fit all 4 nodes
    in the initial call regardless of the current AXIS2_NER_BATCH_SIZE
    default -- the scenario under test is the split-retry itself, not the
    initial batching."""
    monkeypatch.setattr("src.unstructured.semantic.axis2.AXIS2_NER_BATCH_SIZE", 4)
    nodes = [
        _section_node("n0", text="alpha particle decay"),
        _section_node("n1", text="beta decay process"),
        _section_node("n2", text="gamma ray burst"),
        _section_node("n3", text="delta wave pattern"),
    ]
    builder = _builder()
    call_batches: list[list[str]] = []

    def _dispatch(*, messages, **_kwargs):
        content = messages[1]["content"]
        excerpt_count = content.count("[")
        call_batches.append(content)
        if excerpt_count == 4:
            # The full batch: simulate a truncated/malformed response.
            return MagicMock(choices=[MagicMock(message=MagicMock(content="{\"0\": [{\"text\": \"alpha\""))])
        # Both halves (2 excerpts each) succeed.
        if "alpha" in content:
            return _batch_response({"0": ["alpha"], "1": ["beta"]})
        return _batch_response({"0": ["gamma"], "1": ["delta"]})

    builder.client.chat_completion.side_effect = _dispatch

    result = builder._extract_entities(nodes)

    by_id = {n.id: n.entities for n in result}
    assert by_id["n0"] == ["alpha"]
    assert by_id["n1"] == ["beta"]
    assert by_id["n2"] == ["gamma"]
    assert by_id["n3"] == ["delta"]
    # 1 failed full-batch call + 2 successful half-batch retries.
    assert builder.client.chat_completion.call_count == 3


def test_batch_split_retry_terminates_at_single_unit_without_crash():
    # Every call fails, all the way down to batch size 1 -- must terminate
    # (not infinite-recurse) and degrade to empty entities, not crash.
    nodes = [_section_node(f"n{i}") for i in range(4)]
    builder = _builder()
    builder.client.chat_completion.side_effect = RuntimeError("persistent failure")

    result = builder._extract_entities(nodes)  # must not raise or hang

    assert all(n.entities == [] for n in result)


def test_merged_entities_capped_per_node():
    chunk_size = Axis2Builder._NER_CHUNK_CHARS
    cap = Axis2Builder._NER_MAX_ENTITIES_PER_NODE
    # Enough distinct chunks that, uncapped, entities would exceed the cap.
    n_chunks = (cap // 10) + 3
    text = " ".join(f"term{i}chunk padding filler words to force a split" for i in range(n_chunks * 40))
    node = _section_node("n0", text=text)
    builder = _builder()

    def _dispatch(*, messages, **_kwargs):
        # Return a fresh, grounded, unique entity per call so each chunk
        # contributes something new to the merged total.
        excerpt = messages[1]["content"]
        first_word = excerpt.split("[0]\n", 1)[-1].split()[0]
        return _batch_response({"0": [first_word]})

    builder.client.chat_completion.side_effect = _dispatch

    result = builder._extract_entities([node])

    assert len(result[0].entities) <= cap
