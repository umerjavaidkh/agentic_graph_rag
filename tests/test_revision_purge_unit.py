"""Superseding a revision must clear it from every store, not just Neo4j.

`BlobStore.delete_prefix` and `VectorStore.delete_by_filter` were both written
for this purge and had zero call sites, so every re-ingest left its whole
previous text and embedding set behind. Measured on the dev instance before
the fix: Neo4j held 2 revisions, the blob store 51,406 objects across 23, and
Qdrant 6,509 points across 18.
"""
from src.document.purge import (
    purge_revision,
    revision_blob_keys,
    revision_blob_prefixes,
)


class _FakeBlobStore:
    def __init__(self, keys=()):
        self.keys = set(keys)
        self.deleted_prefixes: list[str] = []

    def delete_prefix(self, prefix: str) -> int:
        self.deleted_prefixes.append(prefix)
        matched = {k for k in self.keys if k.startswith(prefix)}
        self.keys -= matched
        return len(matched)

    def exists(self, key: str) -> bool:
        return key in self.keys

    def delete(self, key: str) -> None:
        self.keys.discard(key)


class _FakeVectorStore:
    def __init__(self):
        self.filters: list[dict] = []

    def delete_by_filter(self, filters: dict) -> None:
        self.filters.append(filters)


ARGS = dict(tenant_id="default", logical_id="doc_report", revision_id="doc_report:r1")


def test_prefixes_cover_every_root_a_revision_writes_to():
    """Node text, snapshots and the source file each live under a different
    root; missing one leaks that whole category."""
    prefixes = revision_blob_prefixes(**ARGS)
    assert "default/doc_report/doc_report:r1/" in prefixes
    assert "graph_snapshots/default/doc_report/doc_report:r1/" in prefixes
    assert "documents/default/doc_report/doc_report:r1/" in prefixes


def test_prefixes_end_in_a_slash_so_r1_does_not_swallow_r10():
    """Without the trailing slash, purging r1 would delete r10's data too."""
    for prefix in revision_blob_prefixes(**ARGS):
        assert prefix.endswith("/")
    r1 = revision_blob_prefixes(**ARGS)[0]
    assert not "default/doc_report/doc_report:r10/text".startswith(r1)


def test_page_report_is_an_exact_key_not_a_prefix():
    """It is named for the revision rather than a directory under it, so a
    prefix delete on r1 would also match r10.json."""
    assert revision_blob_keys(**ARGS) == ["page_reports/default/doc_report/doc_report:r1.json"]


def test_keep_source_spares_the_original_upload_only():
    """Derived data can be rebuilt by re-parsing; the uploaded file cannot be
    rebuilt from anything."""
    kept = revision_blob_prefixes(**ARGS, keep_source=True)
    assert not any(p.startswith("documents/") for p in kept)
    assert "default/doc_report/doc_report:r1/" in kept
    assert "graph_snapshots/default/doc_report/doc_report:r1/" in kept


def test_purge_deletes_blobs_and_vectors():
    blob = _FakeBlobStore([
        "default/doc_report/doc_report:r1/n1/text",
        "default/doc_report/doc_report:r1/n2/text",
        "graph_snapshots/default/doc_report/doc_report:r1/x1_structural.json",
        "documents/default/doc_report/doc_report:r1/source.pdf",
        "page_reports/default/doc_report/doc_report:r1.json",
    ])
    vec = _FakeVectorStore()
    result = purge_revision(**ARGS, blob_store=blob, vector_store=vec)

    assert blob.keys == set()
    assert result["blobs"] == 5
    assert vec.filters == [{"revision_id": "doc_report:r1"}]
    assert result["errors"] == []


def test_purge_leaves_another_revisions_data_alone():
    blob = _FakeBlobStore([
        "default/doc_report/doc_report:r1/n1/text",
        "default/doc_report/doc_report:r2/n1/text",
        "default/other_doc/other_doc:r1/n1/text",
    ])
    purge_revision(**ARGS, blob_store=blob, vector_store=_FakeVectorStore())
    assert blob.keys == {
        "default/doc_report/doc_report:r2/n1/text",
        "default/other_doc/other_doc:r1/n1/text",
    }


class _ExplodingBlobStore(_FakeBlobStore):
    def delete_prefix(self, prefix: str) -> int:
        raise RuntimeError("object store unreachable")


def test_purge_is_best_effort_and_still_clears_vectors():
    """A leaked blob costs storage; a raised exception here would fail an
    ingest that has already succeeded, which costs the document."""
    vec = _FakeVectorStore()
    result = purge_revision(**ARGS, blob_store=_ExplodingBlobStore(), vector_store=vec)
    assert result["errors"]
    assert vec.filters == [{"revision_id": "doc_report:r1"}]
