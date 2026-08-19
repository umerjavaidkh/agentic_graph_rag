"""Deleting a document removes it from Neo4j, blobs and vectors alike.

Deleting only the Neo4j half is what left 50,642 orphaned blob objects and
6,195 orphaned vector points on the dev instance.
"""
from pathlib import Path

from src.unstructured.document.purge import delete_document


class _FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def __iter__(self):
        return iter(self._rows)

    def single(self):
        return self._rows[0] if self._rows else None


class _FakeSession:
    def __init__(self, driver):
        self._driver = driver

    def run(self, cypher, **params):
        self._driver.queries.append((cypher, params))
        if "RETURN rev.id AS id" in cypher:
            return _FakeResult([{"id": r} for r in self._driver.revisions])
        if "DETACH DELETE n" in cypher and "count(n)" in cypher:
            self._driver.deleted_revisions = params.get("revisions")
            return _FakeResult([{"c": 42}])
        return _FakeResult([])

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _FakeDriver:
    def __init__(self, revisions):
        self.revisions = revisions
        self.queries: list = []
        self.deleted_revisions = None

    def session(self):
        return _FakeSession(self)


class _FakeBlobStore:
    def __init__(self):
        self.deleted_prefixes: list[str] = []

    def delete_prefix(self, prefix: str) -> int:
        self.deleted_prefixes.append(prefix)
        return 1

    def exists(self, key: str) -> bool:
        return False

    def delete(self, key: str) -> None:  # pragma: no cover - exists() is False
        raise AssertionError("should not be called")


class _FakeVectorStore:
    def __init__(self):
        self.filters: list[dict] = []

    def delete_by_filter(self, filters: dict) -> None:
        self.filters.append(filters)


def _delete(revisions, **kwargs):
    driver = _FakeDriver(revisions)
    blob, vec = _FakeBlobStore(), _FakeVectorStore()
    result = delete_document(
        driver,
        logical_id="doc_report",
        tenant_id="default",
        blob_store=blob,
        vector_store=vec,
        **kwargs,
    )
    return result, driver, blob, vec


def test_unknown_id_returns_none_so_the_caller_can_404():
    """Reporting a successful delete of something that never existed hides
    a typo in the id."""
    result, driver, blob, vec = _delete([])
    assert result is None
    assert blob.deleted_prefixes == []
    assert vec.filters == []


def test_every_revision_is_purged_from_both_stores():
    result, driver, blob, vec = _delete(["doc_report:r1", "doc_report:r2"])
    assert result["revisions"] == ["doc_report:r1", "doc_report:r2"]
    assert vec.filters == [
        {"revision_id": "doc_report:r1"},
        {"revision_id": "doc_report:r2"},
    ]
    for revision in ("doc_report:r1", "doc_report:r2"):
        assert f"default/doc_report/{revision}/" in blob.deleted_prefixes
        assert f"documents/default/doc_report/{revision}/" in blob.deleted_prefixes
        assert f"graph_snapshots/default/doc_report/{revision}/" in blob.deleted_prefixes


def test_neo4j_is_deleted_last():
    """Neo4j is the index of which revisions exist. Dropping it first strands
    the blobs and vectors with nothing left to enumerate them by."""
    result, driver, blob, vec = _delete(["doc_report:r1"])
    kinds = [
        "delete_nodes" if "DETACH DELETE n" in q else "read_revisions"
        for q, _ in driver.queries
        if "RETURN rev.id AS id" in q or "DETACH DELETE n" in q
    ]
    assert kinds[0] == "read_revisions"
    assert kinds[-1] == "delete_nodes"
    assert driver.deleted_revisions == ["doc_report:r1"]
    assert result["nodes"] == 42


def test_keep_source_spares_the_uploaded_file():
    _, _, blob, _ = _delete(["doc_report:r1"], keep_source=True)
    assert not any(p.startswith("documents/") for p in blob.deleted_prefixes)
    assert "default/doc_report/doc_report:r1/" in blob.deleted_prefixes


def test_delete_is_tenant_scoped():
    _, driver, _, _ = _delete(["doc_report:r1"])
    reads = [p for q, p in driver.queries if "RETURN rev.id AS id" in q]
    assert reads and all(p.get("tenant_id") == "default" for p in reads)


def test_endpoint_is_registered_and_admin_gated():
    """A DELETE that anyone could call is a denial-of-service on the corpus.

    Read as source rather than imported: src.api pulls in the whole retrieval
    stack, and other tests in this suite stub parts of it in sys.modules, so
    importing it here passes or fails depending on collection order.
    """
    source = (Path(__file__).parent.parent / "src" / "api.py").read_text()
    decorator = '@app.delete("/documents/{logical_doc_id}")'
    assert decorator in source, "DELETE /documents/{logical_doc_id} is not registered"
    body = source.split(decorator, 1)[1].split("\n@app.", 1)[0]
    assert "resolve_admin_session" in body, "the delete endpoint is not admin-gated"
    assert "delete_document" in body
