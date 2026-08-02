"""
tests/test_backfill_lean_storage_unit.py — Phase 5 backfill script
(scripts/backfill_lean_storage.py).

Run with:
    python -m pytest tests/test_backfill_lean_storage_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
_scripts = _root / "scripts"
if str(_scripts) not in sys.path:
    sys.path.insert(0, str(_scripts))

import backfill_lean_storage as backfill_mod  # noqa: E402


def test_derive_search_text_caps_body_and_prefixes_title():
    result = backfill_mod._derive_search_text("Title", "x" * 5000)
    assert result.startswith("Title\n\n")
    assert len(result) - len("Title\n\n") <= backfill_mod._SEARCH_TEXT_CHAR_BUDGET


def test_derive_search_text_empty_text_falls_back_to_title():
    assert backfill_mod._derive_search_text("Title", "") == "Title"


class _FakeSession:
    def __init__(self, batches):
        self._batches = list(batches)
        self.queries: list[str] = []
        self.set_calls: list[dict] = []

    def run(self, cypher, **kwargs):
        self.queries.append(cypher)
        if "MATCH (n)\n" in cypher and "REMOVE" not in cypher:
            return self._batches.pop(0) if self._batches else []
        self.set_calls.append(kwargs)
        return []


class _FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self):
        return self

    def __enter__(self):
        return self._session

    def __exit__(self, *a):
        return False


class _FakeBlobStore:
    def __init__(self):
        self.puts: dict[str, str] = {}

    def put(self, key, content, *, content_type="text/plain"):
        self.puts[key] = content
        return key


class _FakeVectorStore:
    def point_id_for(self, node_id):
        return f"vec_{node_id}"


def test_backfill_migrates_node_and_strips_text(monkeypatch):
    node = {
        "id": "n1", "title": "Sec", "text": "full body text",
        "embedding": [0.1], "tenant_id": "t1", "logical_doc_id": "doc1", "revision_id": "doc1:r1",
    }
    session = _FakeSession([[node], []])
    driver = _FakeDriver(session)
    blob_store = _FakeBlobStore()
    vector_store = _FakeVectorStore()

    monkeypatch.setattr(backfill_mod, "get_neo4j_driver", lambda: driver)
    monkeypatch.setattr(backfill_mod, "get_blob_store", lambda: blob_store)
    monkeypatch.setattr(backfill_mod, "get_vector_store", lambda: vector_store)

    backfill_mod.backfill(dry_run=False, batch_size=500)

    assert blob_store.puts["t1/doc1/doc1:r1/n1/text"] == "full body text"
    assert len(session.set_calls) == 1
    call = session.set_calls[0]
    assert call["id"] == "n1"
    assert call["blob_key_text"] == "t1/doc1/doc1:r1/n1/text"
    assert call["vector_id"] == "vec_n1"
    assert call["search_text"].startswith("Sec\n\n")


def test_backfill_dry_run_writes_nothing(monkeypatch):
    node = {
        "id": "n1", "title": "Sec", "text": "body",
        "embedding": None, "tenant_id": "t1", "logical_doc_id": "doc1", "revision_id": "doc1:r1",
    }
    session = _FakeSession([[node], []])
    driver = _FakeDriver(session)
    blob_store = _FakeBlobStore()
    vector_store = _FakeVectorStore()

    monkeypatch.setattr(backfill_mod, "get_neo4j_driver", lambda: driver)
    monkeypatch.setattr(backfill_mod, "get_blob_store", lambda: blob_store)
    monkeypatch.setattr(backfill_mod, "get_vector_store", lambda: vector_store)

    backfill_mod.backfill(dry_run=True, batch_size=500)

    assert blob_store.puts == {}
    assert session.set_calls == []


def test_backfill_no_embedding_leaves_vector_id_none(monkeypatch):
    node = {
        "id": "n1", "title": "Sec", "text": "body",
        "embedding": None, "tenant_id": "t1", "logical_doc_id": "doc1", "revision_id": "doc1:r1",
    }
    session = _FakeSession([[node], []])
    driver = _FakeDriver(session)
    blob_store = _FakeBlobStore()
    vector_store = _FakeVectorStore()

    monkeypatch.setattr(backfill_mod, "get_neo4j_driver", lambda: driver)
    monkeypatch.setattr(backfill_mod, "get_blob_store", lambda: blob_store)
    monkeypatch.setattr(backfill_mod, "get_vector_store", lambda: vector_store)

    backfill_mod.backfill(dry_run=False, batch_size=500)

    assert session.set_calls[0]["vector_id"] is None
