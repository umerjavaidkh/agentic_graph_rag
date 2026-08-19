"""
tests/test_ingestion_triage_unit.py — bulk-ingestion triage (structural sanity + dedup).

Run with:
    python -m pytest tests/test_ingestion_triage_unit.py -v
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


# Importing src.unstructured.ingestion.triage runs src/ingestion/__init__.py, which imports
# service.py, which needs fastapi/neo4j — neither installed in this env, and
# triage.py itself has no real dependency on them. Minimal stubs, same
# convention as tests/test_ingestion_manager_di_unit.py.
if "fastapi" not in sys.modules:
    _stub_module("fastapi")
sys.modules["fastapi"].UploadFile = MagicMock()

if "neo4j" not in sys.modules:
    _stub_module("neo4j")
if "neo4j.exceptions" not in sys.modules:
    _stub_module("neo4j.exceptions")
sys.modules["neo4j"].GraphDatabase = MagicMock()
sys.modules["neo4j.exceptions"].ClientError = type("ClientError", (Exception,), {"message": "", "code": ""})

for _n in ["src.shared.auth", "src.shared.auth.rbac_setup"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["src.shared.auth.rbac_setup"].GraphRBAC = MagicMock()

for _n in ["src.unstructured.graph", "src.unstructured.graph.constants", "src.shared.neo4j.driver"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["src.unstructured.graph.constants"].DOC_REVISION_LABEL = "DocRevision"
sys.modules["src.unstructured.graph.constants"].DOCUMENT_LOGICAL_LABEL = "DocumentLogical"
sys.modules["src.unstructured.graph.constants"].DOCUMENT_ROOT_CYPHER = "Document|Book"
sys.modules["src.shared.neo4j.driver"].get_neo4j_driver = MagicMock()

from src.unstructured.ingestion.triage import check_duplicate, check_structural_sanity


# ── check_structural_sanity ──────────────────────────────────────────────────


def test_missing_file_is_rejected(tmp_path):
    reason = check_structural_sanity(
        tmp_path / "nope.pdf", supported_extensions={".pdf"}, max_pdf_pages=10
    )
    assert reason == "file does not exist"


def test_unsupported_extension_is_rejected(tmp_path):
    f = tmp_path / "doc.docx"
    f.write_text("hello")
    reason = check_structural_sanity(f, supported_extensions={".pdf"}, max_pdf_pages=10)
    assert "unsupported extension" in reason


def test_zero_byte_file_is_rejected(tmp_path):
    f = tmp_path / "empty.pdf"
    f.write_bytes(b"")
    reason = check_structural_sanity(f, supported_extensions={".pdf"}, max_pdf_pages=10)
    assert reason == "zero-byte file"


def test_valid_small_pdf_is_accepted(tmp_path):
    import fitz

    f = tmp_path / "real.pdf"
    doc = fitz.open()
    doc.new_page()
    doc.save(str(f))
    doc.close()

    reason = check_structural_sanity(f, supported_extensions={".pdf"}, max_pdf_pages=10)
    assert reason is None


def test_corrupt_pdf_is_rejected(tmp_path):
    f = tmp_path / "corrupt.pdf"
    f.write_bytes(b"%PDF-1.4 not actually a real pdf body")
    reason = check_structural_sanity(f, supported_extensions={".pdf"}, max_pdf_pages=10)
    assert reason is not None
    assert "corrupt or unreadable PDF" in reason


def test_oversized_pdf_is_rejected(monkeypatch, tmp_path):
    f = tmp_path / "big.pdf"
    f.write_bytes(b"%PDF-1.4 fake")

    fake_doc = MagicMock()
    fake_doc.page_count = 5000
    fake_doc.__enter__ = MagicMock(return_value=fake_doc)
    fake_doc.__exit__ = MagicMock(return_value=False)

    import fitz

    monkeypatch.setattr(fitz, "open", lambda *_a, **_k: fake_doc)

    reason = check_structural_sanity(f, supported_extensions={".pdf"}, max_pdf_pages=2000)
    assert "exceeds cap" in reason


def test_zero_page_pdf_is_rejected(monkeypatch, tmp_path):
    f = tmp_path / "blank.pdf"
    f.write_bytes(b"%PDF-1.4 fake")

    fake_doc = MagicMock()
    fake_doc.page_count = 0
    fake_doc.__enter__ = MagicMock(return_value=fake_doc)
    fake_doc.__exit__ = MagicMock(return_value=False)

    import fitz

    monkeypatch.setattr(fitz, "open", lambda *_a, **_k: fake_doc)

    reason = check_structural_sanity(f, supported_extensions={".pdf"}, max_pdf_pages=2000)
    assert reason == "PDF has zero pages"


# ── check_duplicate ──────────────────────────────────────────────────────────


class _FakeSession:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeDriver:
    def __init__(self, session):
        self._session = session

    def session(self):
        return self._session


def test_check_duplicate_returns_reason_when_hash_matches(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_text("some content")

    exporter = MagicMock()
    exporter.active_revision_has_hash.return_value = True
    driver = _FakeDriver(_FakeSession())

    reason = check_duplicate(f, logical_id="doc_test", exporter=exporter, driver=driver)

    assert reason == "duplicate of already-ACTIVE revision"


def test_check_duplicate_returns_none_when_not_duplicate(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_text("some content")

    exporter = MagicMock()
    exporter.active_revision_has_hash.return_value = False
    driver = _FakeDriver(_FakeSession())

    reason = check_duplicate(f, logical_id="doc_test", exporter=exporter, driver=driver)

    assert reason is None


def test_check_duplicate_propagates_driver_errors(tmp_path):
    f = tmp_path / "doc.pdf"
    f.write_text("some content")

    exporter = MagicMock()
    exporter.active_revision_has_hash.side_effect = RuntimeError("neo4j is down")
    driver = _FakeDriver(_FakeSession())

    with pytest.raises(RuntimeError):
        check_duplicate(f, logical_id="doc_test", exporter=exporter, driver=driver)
