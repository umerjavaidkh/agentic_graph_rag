"""
tests/test_ingestion_manager_di_unit.py — IngestionManager dependency injection.

Proves _process_unstructured goes through injected parser/model-provider/
blob-store/vector-store/exporter factories instead of hardcoded concrete
classes, so ingestion can be exercised without a real PDF, OpenAI key,
Neo4j, MinIO, or Qdrant.

Run with:
    python -m pytest tests/test_ingestion_manager_di_unit.py -v
"""
from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ── Minimal stubs for heavy/unavailable deps — must come before src.* imports ─
# Mirrors tests/test_scalable_pipeline_unit.py's stubbing style. Unlike that
# file, we deliberately do NOT stub src.ingestion.service, src.document.parser*,
# src.model_providers*, or src.storage* — this test exercises the REAL
# IngestionManager and proves DI works, so those must be real.


def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Drop any stale stubs a previously-collected test file may have left behind
# for modules we need to import for real in this file.
for _mod_name in list(sys.modules):
    if _mod_name.startswith("src.ingestion") or _mod_name.startswith("src.document"):
        del sys.modules[_mod_name]

if "neo4j" not in sys.modules:
    _stub_module("neo4j")
if "neo4j.exceptions" not in sys.modules:
    _stub_module("neo4j.exceptions")
sys.modules["neo4j"].GraphDatabase = MagicMock()
sys.modules["neo4j.exceptions"].ClientError = type("ClientError", (Exception,), {"message": "", "code": ""})

if "fastapi" not in sys.modules:
    _stub_module("fastapi")
sys.modules["fastapi"].UploadFile = MagicMock()

for _n in ["src.auth", "src.auth.rbac_setup"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["src.auth.rbac_setup"].GraphRBAC = MagicMock()

for _n in ["src.graph", "src.graph.constants", "src.graph.driver"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["src.graph.constants"].DOC_REVISION_LABEL = "DocRevision"
sys.modules["src.graph.constants"].DOCUMENT_LOGICAL_LABEL = "DocumentLogical"
sys.modules["src.graph.driver"].get_neo4j_driver = MagicMock()

from src.ingestion.service import IngestionJob, IngestionManager
from src.ingestion.job_store import InMemoryJobStore
from src.model_providers.base import ModelProvider
from src.storage.vector.memory_store import InMemoryVectorStore


# ── Fakes proving each dependency is swappable ───────────────────────────────


class FakeParser:
    """Records every source it was asked to parse; returns an empty graph."""

    def __init__(self):
        self.parsed_sources: list[str] = []

    def parse(self, source):
        self.parsed_sources.append(str(source))
        return [], []


class FakeModelProvider(ModelProvider):
    def chat_completion(self, model, messages, **kwargs):
        raise AssertionError("chat_completion should not be called in this DI test")

    def embeddings(self, model, input, **kwargs):
        raise AssertionError("embeddings should not be called in this DI test")


class FakeExporter:
    """Stand-in for Neo4jExporter — records construction, no real Neo4j I/O."""

    instances: list["FakeExporter"] = []

    def __init__(self, output_dir):
        self.output_dir = output_dir
        FakeExporter.instances.append(self)


@pytest.fixture()
def tmp_input_file():
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(b"%PDF-1.4 fake content")
        path = Path(f.name)
    yield path
    path.unlink(missing_ok=True)


@pytest.fixture(autouse=True)
def _reset_fake_exporter():
    FakeExporter.instances.clear()
    yield
    FakeExporter.instances.clear()


def _make_manager(fake_parser: FakeParser) -> IngestionManager:
    return IngestionManager(
        store=InMemoryJobStore(),
        parser_factory=lambda source: fake_parser,
        model_provider=FakeModelProvider(),
        blob_store=InMemoryBlobStoreForTest(),
        vector_store=InMemoryVectorStore(),
        exporter_factory=FakeExporter,
    )


class InMemoryBlobStoreForTest:
    """Tiny in-memory BlobStore stand-in (avoids touching the real filesystem)."""

    def __init__(self):
        self._data: dict[str, str] = {}

    def put(self, key, content, *, content_type="text/plain"):
        self._data[key] = content
        return key

    def get(self, key):
        return self._data.get(key)

    def delete(self, key):
        self._data.pop(key, None)

    def exists(self, key):
        return key in self._data

    def delete_prefix(self, prefix):
        to_delete = [k for k in self._data if k.startswith(prefix)]
        for k in to_delete:
            del self._data[k]
        return len(to_delete)


def test_process_unstructured_uses_injected_parser(tmp_input_file, monkeypatch):
    import src.ingestion.service as service_mod

    monkeypatch.setattr(service_mod, "OPENAI_API_KEY", "")
    monkeypatch.setattr(service_mod, "AUTO_LOAD_TO_NEO4J", False)

    fake_parser = FakeParser()
    manager = _make_manager(fake_parser)
    job = IngestionJob(id="job1", type="unstructured", input_path=tmp_input_file)

    manager._process_unstructured(job)

    assert fake_parser.parsed_sources == [str(tmp_input_file)]


def test_process_unstructured_uses_injected_exporter_factory(tmp_input_file, monkeypatch):
    import src.ingestion.service as service_mod

    monkeypatch.setattr(service_mod, "OPENAI_API_KEY", "")
    monkeypatch.setattr(service_mod, "AUTO_LOAD_TO_NEO4J", False)

    manager = _make_manager(FakeParser())
    job = IngestionJob(id="job2", type="unstructured", input_path=tmp_input_file)

    manager._process_unstructured(job)

    assert len(FakeExporter.instances) == 1
    assert job.neo4j_load_status == "skipped"


def test_manager_defaults_are_settings_driven_factories():
    # No kwargs beyond store: production call sites (api.py/tasks.py) construct
    # IngestionManager(store=...) unchanged, and everything else resolves via
    # get_model_provider()/get_blob_store()/get_vector_store() defaults.
    manager = IngestionManager(store=InMemoryJobStore())

    assert manager.blob_store is not None
    assert manager.vector_store is not None
    assert manager.model_provider is not None
    assert manager.parser_factory is not None


def test_process_unstructured_rejects_unsupported_extension(monkeypatch):
    import src.ingestion.service as service_mod

    monkeypatch.setattr(service_mod, "OPENAI_API_KEY", "")
    monkeypatch.setattr(service_mod, "AUTO_LOAD_TO_NEO4J", False)

    manager = _make_manager(FakeParser())
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
        f.write(b"not a pdf")
        bad_path = Path(f.name)
    try:
        job = IngestionJob(id="job3", type="unstructured", input_path=bad_path)
        with pytest.raises(ValueError):
            manager._process_unstructured(job)
    finally:
        bad_path.unlink(missing_ok=True)


def test_cleanup_job_inputs_skips_unowned_path(monkeypatch, tmp_input_file):
    import src.ingestion.service as service_mod

    monkeypatch.setattr(service_mod, "CLEANUP_TMP_INGEST", True)
    manager = _make_manager(FakeParser())
    job = IngestionJob(
        id="job-unowned", type="unstructured", input_path=tmp_input_file, owns_input_path=False
    )

    manager._cleanup_job_inputs(job)

    assert tmp_input_file.exists()


def test_cleanup_job_inputs_deletes_owned_path(monkeypatch, tmp_input_file):
    import src.ingestion.service as service_mod

    monkeypatch.setattr(service_mod, "CLEANUP_TMP_INGEST", True)
    manager = _make_manager(FakeParser())
    job = IngestionJob(
        id="job-owned", type="unstructured", input_path=tmp_input_file, owns_input_path=True
    )

    manager._cleanup_job_inputs(job)

    assert not tmp_input_file.exists()


# ── Corpus (bulk directory/manifest) ingestion ───────────────────────────────


def _real_pdf_bytes() -> bytes:
    import fitz

    doc = fitz.open()
    doc.new_page()
    data = doc.tobytes()
    doc.close()
    return data


def _make_corpus_manager(monkeypatch, *, enqueue_calls=None):
    """IngestionManager with Neo4j/OpenAI disabled and enqueue_ingest faked
    out (real enqueue_ingest would try a real Redis connection)."""
    import src.ingestion.service as service_mod

    monkeypatch.setattr(service_mod, "OPENAI_API_KEY", "")
    monkeypatch.setattr(service_mod, "AUTO_LOAD_TO_NEO4J", False)

    calls = enqueue_calls if enqueue_calls is not None else []

    def fake_enqueue(job_id):
        calls.append(job_id)
        return None  # simulate no-Redis: caller falls back to manager.run_job

    monkeypatch.setattr(service_mod, "enqueue_ingest", fake_enqueue)
    return _make_manager(FakeParser())


def test_process_corpus_walks_directory_and_accepts_valid_files(tmp_path, monkeypatch):
    (tmp_path / "good.pdf").write_bytes(_real_pdf_bytes())
    (tmp_path / "empty.pdf").write_bytes(b"")
    # Directory walk pre-filters by extension (protects the CORPUS_MAX_FILES
    # cap from files that could never be ingested) — this never becomes a
    # candidate at all, unlike a manifest entry (see the manifest test below).
    (tmp_path / "notes.txt").write_text("unsupported extension")

    calls = []
    manager = _make_corpus_manager(monkeypatch, enqueue_calls=calls)
    job = manager.submit_corpus(tmp_path)

    manager._process_corpus(job)

    assert len(job.child_job_ids) == 1
    assert len(calls) == 1
    logs_text = "\n".join(job.logs)
    assert "empty.pdf" in logs_text and "zero-byte file" in logs_text
    assert "notes.txt" not in logs_text
    assert "2 found" in logs_text


def test_process_corpus_skips_directories_and_dotfiles(tmp_path, monkeypatch):
    (tmp_path / "good.pdf").write_bytes(_real_pdf_bytes())
    hidden_dir = tmp_path / ".hidden"
    hidden_dir.mkdir()
    (hidden_dir / "sneaky.pdf").write_bytes(_real_pdf_bytes())
    (tmp_path / ".dotfile.pdf").write_bytes(_real_pdf_bytes())

    manager = _make_corpus_manager(monkeypatch)
    job = manager.submit_corpus(tmp_path)

    manager._process_corpus(job)

    assert len(job.child_job_ids) == 1


def test_process_corpus_manifest_file(tmp_path, monkeypatch):
    good = tmp_path / "good.pdf"
    good.write_bytes(_real_pdf_bytes())
    missing = tmp_path / "missing.pdf"  # listed but never created

    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"# a comment\n{good}\n\n{missing}\n")

    manager = _make_corpus_manager(monkeypatch)
    job = manager.submit_corpus(manifest)

    manager._process_corpus(job)

    assert len(job.child_job_ids) == 1
    assert any("does not exist" in line for line in job.logs)


def test_process_corpus_manifest_logs_unsupported_extension(tmp_path, monkeypatch):
    """Unlike a directory walk (which pre-filters silently), a manifest entry
    with an unsupported extension shows up as a logged triage rejection."""
    bad = tmp_path / "notes.txt"
    bad.write_text("not a pdf")
    manifest = tmp_path / "manifest.txt"
    manifest.write_text(f"{bad}\n")

    manager = _make_corpus_manager(monkeypatch)
    job = manager.submit_corpus(manifest)

    manager._process_corpus(job)

    assert job.child_job_ids == []
    assert any("unsupported extension" in line for line in job.logs)


def test_process_corpus_manifest_rejects_relative_path(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.txt"
    manifest.write_text("relative/path.pdf\n")

    manager = _make_corpus_manager(monkeypatch)
    job = manager.submit_corpus(manifest)

    with pytest.raises(ValueError):
        manager._process_corpus(job)


def test_process_corpus_respects_max_files_cap(tmp_path, monkeypatch):
    import src.ingestion.service as service_mod

    (tmp_path / "a.pdf").write_bytes(_real_pdf_bytes())
    (tmp_path / "b.pdf").write_bytes(_real_pdf_bytes())
    monkeypatch.setattr(service_mod, "CORPUS_MAX_FILES", 1)

    manager = _make_corpus_manager(monkeypatch)
    job = manager.submit_corpus(tmp_path)

    with pytest.raises(ValueError):
        manager._process_corpus(job)
    assert job.child_job_ids == []


def test_process_corpus_falls_back_to_synchronous_run_when_no_queue(tmp_path, monkeypatch):
    """enqueue_ingest returning None (no Redis) must trigger manager.run_job(child_id)."""
    import src.ingestion.service as service_mod

    (tmp_path / "good.pdf").write_bytes(_real_pdf_bytes())
    monkeypatch.setattr(service_mod, "OPENAI_API_KEY", "")
    monkeypatch.setattr(service_mod, "AUTO_LOAD_TO_NEO4J", False)
    monkeypatch.setattr(service_mod, "enqueue_ingest", lambda job_id: None)

    manager = _make_manager(FakeParser())
    run_job_calls = []
    original_run_job = manager.run_job

    def spy_run_job(job_id):
        run_job_calls.append(job_id)
        return original_run_job(job_id)

    monkeypatch.setattr(manager, "run_job", spy_run_job)

    job = manager.submit_corpus(tmp_path)
    manager._process_corpus(job)

    assert run_job_calls == job.child_job_ids
    child = manager.get_job(job.child_job_ids[0])
    assert child.status.value == "completed"


def test_submit_corpus_raises_for_missing_source(monkeypatch):
    manager = _make_corpus_manager(monkeypatch)
    with pytest.raises(FileNotFoundError):
        manager.submit_corpus("/definitely/not/a/real/path/xyz")
