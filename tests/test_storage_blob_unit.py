"""
tests/test_storage_blob_unit.py — BlobStore interface round-trip tests.

Run with:
    python -m pytest tests/test_storage_blob_unit.py -v
"""
from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock


from src.storage.blob.local_store import LocalFsBlobStore
from src.storage.blob.factory import get_blob_store


def _store() -> LocalFsBlobStore:
    tmp = tempfile.mkdtemp()
    return LocalFsBlobStore(tmp)


def test_put_then_get_round_trips():
    store = _store()
    store.put("doc1/rev1/node1/text", "hello world")
    assert store.get("doc1/rev1/node1/text") == "hello world"


def test_get_missing_key_returns_none():
    store = _store()
    assert store.get("nope/nope/nope") is None


def test_exists_reflects_put_and_delete():
    store = _store()
    key = "doc1/rev1/node1/text"
    assert not store.exists(key)
    store.put(key, "content")
    assert store.exists(key)
    store.delete(key)
    assert not store.exists(key)


def test_delete_missing_key_is_noop():
    store = _store()
    store.delete("nope/nope/nope")  # should not raise


def test_delete_prefix_removes_all_keys_under_revision():
    store = _store()
    store.put("doc1/rev1/node1/text", "a")
    store.put("doc1/rev1/node2/text", "b")
    store.put("doc1/rev2/node1/text", "c")  # different revision, must survive

    deleted = store.delete_prefix("doc1/rev1")
    assert deleted == 2
    assert store.get("doc1/rev1/node1/text") is None
    assert store.get("doc1/rev1/node2/text") is None
    assert store.get("doc1/rev2/node1/text") == "c"


def test_factory_defaults_to_local_backend(monkeypatch):
    # BLOB_STORE_BACKEND defaults to "local" (settings.py) when unset; force it
    # here so the test is isolated from whatever backend the local/deployed .env
    # actually configures (e.g. BLOB_STORE_BACKEND=minio in a real deployment).
    import src.config.settings as settings_mod
    import src.storage.blob.factory as factory_mod

    monkeypatch.setattr(settings_mod, "BLOB_STORE_BACKEND", "local")
    factory_mod._store_singleton = None
    try:
        store = get_blob_store()
        assert isinstance(store, LocalFsBlobStore)
    finally:
        factory_mod._store_singleton = None


# ── MinioBlobStore, with the `minio` SDK mocked (no real MinIO required) ────


class _FakeS3Error(Exception):
    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


def _install_fake_minio_sdk() -> MagicMock:
    """Stub the `minio` package in sys.modules; returns the fake Minio client class."""
    minio_mod = types.ModuleType("minio")
    minio_error_mod = types.ModuleType("minio.error")
    minio_error_mod.S3Error = _FakeS3Error
    fake_client_cls = MagicMock(name="Minio")
    minio_mod.Minio = fake_client_cls
    minio_mod.error = minio_error_mod
    sys.modules["minio"] = minio_mod
    sys.modules["minio.error"] = minio_error_mod
    return fake_client_cls


def _reload_minio_store():
    import importlib

    if "src.storage.blob.minio_store" in sys.modules:
        importlib.reload(sys.modules["src.storage.blob.minio_store"])
    import src.storage.blob.minio_store as mod

    return mod


def test_minio_store_put_calls_put_object():
    fake_cls = _install_fake_minio_sdk()
    fake_client = MagicMock()
    fake_client.bucket_exists.return_value = True
    fake_cls.return_value = fake_client
    mod = _reload_minio_store()

    store = mod.MinioBlobStore("localhost:9000", "ak", "sk", "bucket")
    store.put("doc1/rev1/node1/text", "hello")

    assert fake_client.put_object.called
    args, kwargs = fake_client.put_object.call_args
    assert args[0] == "bucket"
    assert args[1] == "doc1/rev1/node1/text"


def test_minio_store_creates_bucket_if_missing():
    fake_cls = _install_fake_minio_sdk()
    fake_client = MagicMock()
    fake_client.bucket_exists.return_value = False
    fake_cls.return_value = fake_client
    mod = _reload_minio_store()

    mod.MinioBlobStore("localhost:9000", "ak", "sk", "bucket")

    fake_client.make_bucket.assert_called_once_with("bucket")


def test_minio_store_get_missing_key_returns_none():
    fake_cls = _install_fake_minio_sdk()
    fake_client = MagicMock()
    fake_client.bucket_exists.return_value = True
    fake_client.get_object.side_effect = _FakeS3Error("NoSuchKey")
    fake_cls.return_value = fake_client
    mod = _reload_minio_store()

    store = mod.MinioBlobStore("localhost:9000", "ak", "sk", "bucket")
    assert store.get("missing/key") is None


def test_minio_store_exists_false_on_no_such_key():
    fake_cls = _install_fake_minio_sdk()
    fake_client = MagicMock()
    fake_client.bucket_exists.return_value = True
    fake_client.stat_object.side_effect = _FakeS3Error("NoSuchKey")
    fake_cls.return_value = fake_client
    mod = _reload_minio_store()

    store = mod.MinioBlobStore("localhost:9000", "ak", "sk", "bucket")
    assert store.exists("missing/key") is False


def test_minio_store_delete_prefix_removes_listed_objects():
    fake_cls = _install_fake_minio_sdk()
    fake_client = MagicMock()
    fake_client.bucket_exists.return_value = True
    fake_obj1 = MagicMock(object_name="doc1/rev1/node1/text")
    fake_obj2 = MagicMock(object_name="doc1/rev1/node2/text")
    fake_client.list_objects.return_value = [fake_obj1, fake_obj2]
    fake_cls.return_value = fake_client
    mod = _reload_minio_store()

    store = mod.MinioBlobStore("localhost:9000", "ak", "sk", "bucket")
    deleted = store.delete_prefix("doc1/rev1")

    assert deleted == 2
    assert fake_client.remove_object.call_count == 2
