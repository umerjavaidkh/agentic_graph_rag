"""The same file must never become two documents.

Three copies of one PDF once sat in the graph at the same time, all titled
from the same filename and so indistinguishable in the document picker. Each
had been ingested under a different doc_key, and supersede only ever fires
within one logical id, so nothing expired anything.
"""
from pathlib import Path

from src.unstructured.document.versioning import build_revision_plan


def _plan(tmp_path: Path, doc_key: str, *, logical_id: str | None = None):
    f = tmp_path / "report.pdf"
    if not f.exists():
        f.write_bytes(b"identical bytes")
    return build_revision_plan(
        f, tenant_id="default", doc_key=doc_key, logical_id=logical_id
    )


def test_different_doc_keys_diverge_without_an_explicit_logical_id(tmp_path):
    """The behaviour that produced the duplicates, pinned so it stays visible."""
    a = _plan(tmp_path, "run-one")
    b = _plan(tmp_path, "run-two")
    assert a.content_hash == b.content_hash
    assert a.logical_id != b.logical_id


def test_explicit_logical_id_overrides_doc_key(tmp_path):
    """The fix: the caller can bind a re-ingest to the document already holding
    this content, so the existing supersede path expires the older copy."""
    first = _plan(tmp_path, "run-one")
    second = _plan(tmp_path, "run-two", logical_id=first.logical_id)
    assert second.logical_id == first.logical_id
    assert second.content_hash == first.content_hash


class _FakeResult:
    def __init__(self, row):
        self._row = row

    def single(self):
        return self._row


class _FakeSession:
    def __init__(self, row):
        self._row = row
        self.params: dict = {}

    def run(self, _cypher, **params):
        self.params = params
        return _FakeResult(self._row)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def _exporter():
    from src.unstructured.exporter.exporter import Neo4jExporter

    return Neo4jExporter.__new__(Neo4jExporter)


def test_logical_id_holding_hash_finds_the_owning_document():
    session = _FakeSession({"logical_id": "doc_report"})
    assert (
        _exporter().logical_id_holding_hash(session, "abc123", "default")
        == "doc_report"
    )
    assert session.params == {"content_hash": "abc123", "tenant_id": "default"}


def test_logical_id_holding_hash_returns_none_for_new_content():
    assert _exporter().logical_id_holding_hash(_FakeSession(None), "abc123", "default") is None


def test_hash_lookup_is_tenant_scoped():
    """One tenant's upload must not be absorbed into another tenant's document."""
    session = _FakeSession({"logical_id": "doc_report"})
    _exporter().logical_id_holding_hash(session, "abc123", "acme")
    assert session.params["tenant_id"] == "acme"
