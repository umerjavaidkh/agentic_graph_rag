"""
tests/test_scalable_pipeline_unit.py — Unit tests for the scalable ingestion pipeline.

Covers:
  1. JobStore round-trip (InMemoryJobStore always; RedisJobStore via fakeredis if available).
  2. enqueue_ingest wiring (mocked queue).
  3. Axis 2 parallel NER produces entities on all nodes and handles LLM errors gracefully.
  4. Axis 2 LLM-pair cap limits the number of LLM calls.
  5. Exporter _node_to_param_dict builds correct parameter rows for UNWIND.
  6. Exporter batch label-grouping logic.

Run with:
    python -m pytest tests/test_scalable_pipeline_unit.py -v
"""
from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

# ── Minimal stubs — installed in setup_module(), not at module level ────────
# pytest collection imports every test module in the session before any test
# executes, so stubbing sys.modules here at import time would leak into every
# other file's test-execution phase until this file's own tests finished and
# teardown_module ran below — a collection-vs-execution ordering bug that
# corrupted src.shared.model_providers.factory (get_chat_provider etc. resolved to
# this file's MagicMock) for any test file whose tests ran before this file's
# position in the suite. setup_module() runs immediately before this file's
# own first test executes, matching teardown_module's timing.

def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


class _FakeBaseModel:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


# Provide a minimal IngestionJob so job_store and tests can use it without
# pulling in the full fastapi/neo4j import chain via service.py. We load the
# REAL models.py (no heavy deps) to get IngestionStatus.
_root = Path(__file__).resolve().parents[1]

from src.ingestion.models import IngestionStatus  # noqa: E402 – after path setup

@dataclass
class _IngestionJob:
    id: str
    type: str
    name: Optional[str] = None
    doc_key: Optional[str] = None
    status: IngestionStatus = IngestionStatus.queued
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    input_path: Optional[Path] = None
    output_dir: Optional[Path] = None
    cypher_params: Optional[Dict[str, object]] = None
    neo4j_load_status: Optional[str] = None
    neo4j_load_message: Optional[str] = None
    logs: List[str] = field(default_factory=list)
    error: Optional[str] = None
    logical_doc_id: Optional[str] = None
    revision_id: Optional[str] = None
    content_hash: Optional[str] = None
    version_number: Optional[int] = None
    skipped_duplicate: bool = False
    owns_input_path: bool = True
    child_job_ids: List[str] = field(default_factory=list)
    tenant_id: Optional[str] = None


def setup_module(module) -> None:
    """Install this file's fake stand-ins for heavy/unavailable deps. Runs
    right before this file's own tests execute (pytest's setup_module hook),
    not at import/collection time — see the module-level comment above for
    why that distinction matters."""
    # --- neo4j stubs ---
    if "neo4j" not in sys.modules:
        _stub_module("neo4j")
    if "neo4j.exceptions" not in sys.modules:
        _stub_module("neo4j.exceptions")
    sys.modules["neo4j"].GraphDatabase = MagicMock()
    sys.modules["neo4j.exceptions"].ClientError = type("ClientError", (Exception,), {"message": "", "code": ""})

    # --- fastapi stubs ---
    if "fastapi" not in sys.modules:
        _stub_module("fastapi")
    if "fastapi.responses" not in sys.modules:
        _stub_module("fastapi.responses")
    if "fastapi.staticfiles" not in sys.modules:
        _stub_module("fastapi.staticfiles")
    _fa = sys.modules["fastapi"]
    _fa.UploadFile = MagicMock()
    _fa.File = MagicMock()
    _fa.Form = MagicMock()
    _fa.HTTPException = type("HTTPException", (Exception,), {"status_code": 0, "detail": ""})
    _fa.BackgroundTasks = MagicMock()
    _fa.FastAPI = MagicMock()
    sys.modules["fastapi.responses"].HTMLResponse = MagicMock()
    sys.modules["fastapi.responses"].RedirectResponse = MagicMock()
    sys.modules["fastapi.staticfiles"].StaticFiles = MagicMock()

    # --- pydantic stubs ---
    if "pydantic" not in sys.modules:
        _stub_module("pydantic")
    sys.modules["pydantic"].BaseModel = _FakeBaseModel
    sys.modules["pydantic"].Field = lambda *a, **kw: None

    # --- openai stubs ---
    if "openai" not in sys.modules:
        _stub_module("openai")
    sys.modules["openai"].OpenAI = MagicMock()

    # --- langgraph stubs ---
    for _n in ["langgraph", "langgraph.graph"]:
        if _n not in sys.modules:
            _stub_module(_n)

    # --- sklearn / hdbscan stubs ---
    for _n in ["sklearn", "sklearn.cluster"]:
        if _n not in sys.modules:
            _stub_module(_n)
    sys.modules["sklearn.cluster"].KMeans = MagicMock()
    if "hdbscan" not in sys.modules:
        _stub_module("hdbscan")
    sys.modules["hdbscan"].HDBSCAN = MagicMock()

    # --- model_providers stubs ---
    for _n in [
        "src.shared.model_providers",
        "src.shared.model_providers.base",
        "src.shared.model_providers.factory",
        "src.shared.model_providers.openai_provider",
    ]:
        if _n not in sys.modules:
            _stub_module(_n)
    _factory_mock = MagicMock()
    sys.modules["src.shared.model_providers.base"].ModelProvider = object
    sys.modules["src.shared.model_providers.factory"].get_model_provider = _factory_mock
    sys.modules["src.shared.model_providers.factory"].get_chat_provider = _factory_mock
    sys.modules["src.shared.model_providers.factory"].get_embedding_provider = _factory_mock
    sys.modules["src.shared.model_providers"].get_model_provider = _factory_mock

    # --- auth stubs ---
    # Always create fresh fake modules here (never reuse/mutate a real src.shared.auth
    # that an earlier-collected test file may have already imported) — mutating
    # the real module's classes in place would corrupt it for every other test
    # file that runs afterward in the same pytest process.
    for _n in ["src.shared.auth", "src.shared.auth.rbac_setup", "src.shared.auth.roles"]:
        _stub_module(_n)
    sys.modules["src.shared.auth.rbac_setup"].GraphRBAC = MagicMock()
    sys.modules["src.shared.auth.roles"].Role = MagicMock()
    sys.modules["src.shared.auth.roles"].UserContext = MagicMock()
    sys.modules["src.shared.auth.roles"].validate_role = MagicMock()

    # --- document stubs ---
    # Always create fresh fake modules here (never reuse/mutate a real
    # src.document that an earlier-collected test file may have already
    # imported) — mutating the real module's functions in place would corrupt
    # it for every other test file that runs afterward in the same pytest
    # process. Same fix as the src.shared.auth block above.
    for _n in [
        "src.document",
        "src.document.versioning",
        "src.document.light",
        "src.document.light.parser",
        "src.document.parser_base",
        "src.document.parser_registry",
        "src.document.page_vision",
        "src.document.graph_snapshot",
        "src.document.page_report",
        "src.document.page_validation",
    ]:
        _stub_module(_n)
    sys.modules["src.document.parser_base"].DocumentParser = object
    sys.modules["src.document.graph_snapshot"].X1_STAGE = "x1_structural"
    sys.modules["src.document.graph_snapshot"].X2_STAGE = "x2_semantic"
    sys.modules["src.document.graph_snapshot"].write_snapshot = MagicMock()
    sys.modules["src.document.page_report"].write_page_report = MagicMock()
    sys.modules["src.document.page_validation"].check_construction_coverage = MagicMock(
        return_value={"pages": [], "summary": {"page_count": 0, "avg_coverage": 0.0, "pages_failing": 0, "requires_reprocessing": False}}
    )
    sys.modules["src.document.versioning"].resolve_logical_id = MagicMock(return_value="doc_test")
    sys.modules["src.document.versioning"].build_revision_plan = MagicMock()
    sys.modules["src.document.versioning"].apply_revision_to_graph = MagicMock(return_value=([], []))
    sys.modules["src.document.versioning"].file_content_sha256 = MagicMock(return_value="abc123")
    sys.modules["src.document.versioning"].source_file_blob_key = MagicMock(return_value="blob/key")
    # DocumentRevisionPlan as a simple MagicMock class (exporter.py uses it only as a type annotation)
    sys.modules["src.document.versioning"].DocumentRevisionPlan = MagicMock

    sys.modules["src.document.light.parser"].LightPdfParser = MagicMock()
    _fake_parser_instance = MagicMock()
    sys.modules["src.document.parser_registry"].get_parser = MagicMock(return_value=_fake_parser_instance)
    sys.modules["src.document.parser_registry"].supported_extensions = MagicMock(return_value={".pdf"})

    # --- graph.constants / graph.driver stubs ---
    for _n in ["src.graph", "src.graph.constants", "src.shared.neo4j.driver"]:
        if _n not in sys.modules:
            _stub_module(_n)
    sys.modules["src.graph.constants"].DOC_REVISION_LABEL = "DocRevision"
    sys.modules["src.graph.constants"].DOCUMENT_LOGICAL_LABEL = "DocumentLogical"
    sys.modules["src.graph.constants"].DOCUMENT_ROOT_CYPHER = "Document|Book"
    sys.modules["src.shared.neo4j.driver"].get_neo4j_driver = MagicMock()

    # --- bridge/conversation stubs ---
    for _n in ["src.bridge", "src.shared.conversation", "src.routing", "src.router"]:
        if _n not in sys.modules:
            _stub_module(_n)

    # --- src.ingestion.service stub ---
    # Inject the stub service module BEFORE job_store.py is imported.
    _svc_stub = _stub_module("src.ingestion.service")
    _svc_stub.IngestionJob = _IngestionJob
    _svc_stub.IngestionManager = MagicMock()

    # Stub the ingestion package itself so __init__.py doesn't run,
    # but set __path__ so Python can still find sub-modules on disk.
    _ing_pkg = _stub_module("src.ingestion")
    _ing_pkg.IngestionManager = MagicMock()
    _ing_pkg.IngestionJob = _IngestionJob
    _ing_pkg.__path__ = [str(_root / "src" / "ingestion")]
    _ing_pkg.__package__ = "src.ingestion"


_STUBBED_MODULE_NAMES = (
    "neo4j", "neo4j.exceptions",
    "fastapi", "fastapi.responses", "fastapi.staticfiles",
    "pydantic", "openai", "langgraph", "langgraph.graph",
    "sklearn", "sklearn.cluster",
    "src.shared.model_providers", "src.shared.model_providers.base",
    "src.shared.model_providers.factory", "src.shared.model_providers.openai_provider",
    "src.shared.auth", "src.shared.auth.rbac_setup", "src.shared.auth.roles",
    "src.document", "src.document.versioning", "src.document.light",
    "src.document.light.parser", "src.document.parser_base",
    "src.document.parser_registry", "src.document.page_vision",
    "src.document.graph_snapshot",
    "src.graph", "src.graph.constants", "src.shared.neo4j.driver",
    "src.bridge", "src.shared.conversation", "src.routing", "src.router",
    "src.ingestion.service", "src.ingestion",
)


def teardown_module(module) -> None:
    """Remove this file's fake stand-ins once its own tests are done, so a
    test file collected afterward gets a clean sys.modules and can import
    the real neo4j/fastapi/src.document/src.graph/etc. if it needs them —
    otherwise whichever of those modules this file stubbed stays faked
    (or, worse, a bare non-package ModuleType with no __path__) for every
    test that runs later in the same pytest process. Same fix as
    test_ingestion_manager_di_unit.py's own teardown_module, applied here
    after this file's stub list quietly drifted out of sync with what
    src/ingestion/service.py actually imports (missing
    src.document.graph_snapshot, source_file_blob_key, get_chat_provider)
    and, separately, was found leaking a fake src.graph package into
    tests/test_search_text_derivation_unit.py's collection."""
    for _n in _STUBBED_MODULE_NAMES:
        sys.modules.pop(_n, None)


# ── Test 1: InMemoryJobStore round-trip ──────────────────────────────────────

class TestInMemoryJobStore:
    def _make_job(self, job_id: str = "test_job_001") -> _IngestionJob:
        return _IngestionJob(id=job_id, type="unstructured", name="Test Doc")

    def test_save_and_get(self):
        from src.ingestion.job_store import InMemoryJobStore
        store = InMemoryJobStore()
        job = self._make_job()
        store.save(job)
        retrieved = store.get(job.id)
        assert retrieved is not None
        assert retrieved.id == job.id
        assert retrieved.type == job.type
        assert retrieved.name == job.name

    def test_get_missing_returns_none(self):
        from src.ingestion.job_store import InMemoryJobStore
        store = InMemoryJobStore()
        assert store.get("nonexistent") is None

    def test_append_and_get_logs(self):
        from src.ingestion.job_store import InMemoryJobStore
        store = InMemoryJobStore()
        job = self._make_job()
        store.save(job)
        store.append_log(job.id, "line 1")
        store.append_log(job.id, "line 2")
        logs = store.get_logs(job.id)
        assert logs == ["line 1", "line 2"]

    def test_list_ids(self):
        from src.ingestion.job_store import InMemoryJobStore
        store = InMemoryJobStore()
        for i in range(5):
            store.save(self._make_job(f"job_{i}"))
        ids = store.list_ids(limit=10)
        assert len(ids) == 5
        assert "job_0" in ids

    def test_delete(self):
        from src.ingestion.job_store import InMemoryJobStore
        store = InMemoryJobStore()
        job = self._make_job()
        store.save(job)
        store.delete(job.id)
        assert store.get(job.id) is None

    def test_status_change_persisted(self):
        from src.ingestion.job_store import InMemoryJobStore
        store = InMemoryJobStore()
        job = self._make_job()
        store.save(job)
        job.status = IngestionStatus.completed
        store.save(job)
        retrieved = store.get(job.id)
        assert retrieved.status == IngestionStatus.completed


# ── Test 2: RedisJobStore round-trip (via fakeredis) ─────────────────────────

class TestRedisJobStore:
    @pytest.fixture
    def fake_client(self):
        try:
            import fakeredis
            return fakeredis.FakeRedis()
        except ImportError:
            pytest.skip("fakeredis not installed")

    def _make_job(self, job_id: str = "redis_job_001") -> _IngestionJob:
        j = _IngestionJob(
            id=job_id,
            type="unstructured",
            name="Redis Test Doc",
            logical_doc_id="doc_redis_test",
            content_hash="abc123",
            version_number=1,
        )
        return j

    def test_save_and_get(self, fake_client):
        from src.ingestion.job_store import RedisJobStore
        store = RedisJobStore(fake_client)
        job = self._make_job()
        store.save(job)
        retrieved = store.get(job.id)
        assert retrieved is not None
        assert retrieved.id == job.id
        assert retrieved.logical_doc_id == job.logical_doc_id
        assert retrieved.content_hash == job.content_hash
        assert retrieved.version_number == job.version_number

    def test_logs_survive_round_trip(self, fake_client):
        from src.ingestion.job_store import RedisJobStore
        store = RedisJobStore(fake_client)
        job = self._make_job()
        store.save(job)
        store.append_log(job.id, "worker started")
        store.append_log(job.id, "parsing done")
        retrieved = store.get(job.id)
        assert "worker started" in retrieved.logs
        assert "parsing done" in retrieved.logs

    def test_list_ids_deduplication(self, fake_client):
        from src.ingestion.job_store import RedisJobStore
        store = RedisJobStore(fake_client)
        job = self._make_job("dedup_job")
        store.save(job)
        store.save(job)  # save twice — should appear once in list
        ids = store.list_ids(limit=100)
        assert ids.count("dedup_job") == 1


# ── Test 3: job_to_dict / job_from_dict round-trip ───────────────────────────

class TestJobSerialization:
    def test_round_trip_preserves_all_fields(self):
        from src.ingestion.job_store import job_to_dict, job_from_dict

        job = _IngestionJob(
            id="ser_test_001",
            type="unstructured",
            name="Serialisation Test",
            doc_key="my-doc",
            status=IngestionStatus.semantic_enrichment,
            logical_doc_id="doc_my_doc",
            revision_id="rev_001",
            content_hash="deadbeef",
            version_number=3,
            skipped_duplicate=True,
        )
        job.started_at = datetime(2026, 1, 1, 12, 0, 0)
        job.finished_at = datetime(2026, 1, 1, 12, 5, 0)

        d = job_to_dict(job)
        restored = job_from_dict(d)

        assert restored.id == job.id
        assert restored.type == job.type
        assert restored.doc_key == job.doc_key
        assert restored.status == job.status
        assert restored.logical_doc_id == job.logical_doc_id
        assert restored.revision_id == job.revision_id
        assert restored.content_hash == job.content_hash
        assert restored.version_number == job.version_number
        assert restored.skipped_duplicate is True
        assert restored.started_at == job.started_at
        assert restored.finished_at == job.finished_at


# ── Test 4: enqueue_ingest wiring ────────────────────────────────────────────

class TestQueueWiring:
    def test_returns_none_when_no_redis(self):
        """When REDIS_URL is empty, enqueue_ingest should return None."""
        import src.ingestion.queue as queue_mod
        queue_mod._queue = None  # reset singleton
        with patch("src.ingestion.queue.get_ingest_queue", return_value=None):
            result = queue_mod.enqueue_ingest("test_job_id")
        assert result is None

    def test_list_failed_jobs_returns_empty_without_redis(self):
        import src.ingestion.queue as queue_mod
        queue_mod._queue = None
        with patch("src.ingestion.queue.get_ingest_queue", return_value=None):
            result = queue_mod.list_failed_jobs()
        assert result == []

    def test_queue_depth_returns_none_without_redis(self):
        import src.ingestion.queue as queue_mod
        queue_mod._queue = None
        with patch("src.ingestion.queue.get_ingest_queue", return_value=None):
            result = queue_mod.queue_depth()
        assert result is None


# ── Test 5: Axis 2 parallel NER ──────────────────────────────────────────────

class TestAxis2ParallelNER:
    def _make_nodes(self, n: int = 5):
        from src.models import DKGNode, NodeType
        return [
            DKGNode(id=f"node_{i}", type=NodeType.SECTION,
                    title=f"Section {i}", text=f"Content for section {i}. Entity A.", order=i)
            for i in range(n)
        ]

    def test_parallel_ner_sets_entities_on_all_nodes(self):
        from src.semantic.axis2 import Axis2Builder

        mock_client = MagicMock()
        mock_client.chat_completion.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content='["Entity A", "Entity B"]'))]
        )
        builder = Axis2Builder.__new__(Axis2Builder)
        builder.client = mock_client

        nodes = self._make_nodes(6)
        result_nodes = builder._extract_entities(nodes)
        entities_assigned = [n for n in result_nodes if getattr(n, "entities", None)]
        assert len(entities_assigned) == 6

    def test_parallel_ner_handles_llm_errors_gracefully(self):
        """If some LLM calls raise, the rest should still succeed."""
        from src.semantic.axis2 import Axis2Builder

        call_count = {"n": 0}

        def flaky_completion(**kwargs):
            call_count["n"] += 1
            if call_count["n"] % 2 == 0:
                raise RuntimeError("Simulated LLM error")
            m = MagicMock()
            m.choices[0].message.content = '["Safe Entity"]'
            return m

        mock_client = MagicMock()
        mock_client.chat_completion.side_effect = flaky_completion
        builder = Axis2Builder.__new__(Axis2Builder)
        builder.client = mock_client

        nodes = self._make_nodes(4)
        result_nodes = builder._extract_entities(nodes)
        # Should not raise; errored nodes get []
        for node in result_nodes:
            assert isinstance(getattr(node, "entities", []), list)

    def test_llm_pair_cap_limits_candidates(self):
        """_build_llm_edges should not send more than AXIS2_MAX_LLM_PAIRS calls."""
        import src.semantic.axis2 as axis2_mod
        from src.semantic.axis2 import Axis2Builder
        import numpy as np
        from src.models import DKGNode, NodeType

        # Patch the axis2 module's local name (captured at import time)
        original_cap = axis2_mod.AXIS2_MAX_LLM_PAIRS
        axis2_mod.AXIS2_MAX_LLM_PAIRS = 3

        try:
            builder = Axis2Builder.__new__(Axis2Builder)
            builder.client = MagicMock()
            builder.client.chat_completion.return_value = MagicMock(
                choices=[MagicMock(message=MagicMock(
                    content='{"relationship":"NONE","direction":"A_TO_B","confidence":0.0,"reason":""}'
                ))]
            )

            nodes = []
            for i in range(6):
                n = DKGNode(id=f"n{i}", type=NodeType.SECTION,
                            title=f"S{i}", text="x", order=i)
                n.embedding = [1.0] * 8  # all cosine-similar
                nodes.append(n)

            builder._build_llm_edges(nodes)
            call_count = builder.client.chat_completion.call_count
            assert call_count <= 3, f"Expected ≤3 LLM calls, got {call_count}"
        finally:
            axis2_mod.AXIS2_MAX_LLM_PAIRS = original_cap


# ── Test 6: Exporter UNWIND batch parameter builder ─────────────────────────

class TestExporterBatch:
    def _make_node(self, node_id: str = "n1"):
        from src.models import DKGNode, NodeType
        node = DKGNode(id=node_id, type=NodeType.SECTION,
                       title="Test section", text="Hello world", order=0)
        node.logical_doc_id = "doc_test"
        node.revision_id = "rev_001"
        node.lifecycle_status = "ACTIVE"
        node.content_hash = "abc"
        node.version_number = 1
        node.ingested_at = 0
        node.source_filename = "test.pdf"
        return node

    def test_node_to_param_dict_has_required_keys(self):
        from src.exporter.exporter import Neo4jExporter
        node = self._make_node()
        d = Neo4jExporter._node_to_param_dict(node)
        for key in ("id", "title", "search_text", "vector_id", "logical_doc_id",
                    "revision_id", "lifecycle_status", "entities"):
            assert key in d, f"Missing key: {key}"

    def test_node_to_param_dict_excludes_text_and_embedding(self):
        """text/embedding are authoritative in the blob/vector stores only
        (see _dual_write_chunk) -- Neo4j must never receive either as of
        the phase-3 write-side strip (docs/DESIGN_unstructured_graph_v2.md).
        search_text/blob_key_text/vector_id are what Neo4j keeps instead."""
        from src.exporter.exporter import Neo4jExporter
        node = self._make_node()
        d = Neo4jExporter._node_to_param_dict(node)
        assert "text" not in d
        assert "embedding" not in d

    def test_node_to_param_dict_no_node_type_enum(self):
        """The dict must NOT contain NodeType enum objects — only JSON-safe values."""
        from src.exporter.exporter import Neo4jExporter
        from src.models import NodeType
        node = self._make_node()
        d = Neo4jExporter._node_to_param_dict(node)
        for v in d.values():
            assert not isinstance(v, NodeType), f"Found NodeType enum value in dict: {v}"

    def test_batch_grouping_by_label(self):
        """Nodes of different labels must be grouped separately for UNWIND."""
        from src.models import DKGNode, NodeType
        from collections import defaultdict

        sections = [self._make_node(f"s{i}") for i in range(3)]
        pages = []
        for i in range(2):
            n = DKGNode(id=f"p{i}", type=NodeType.PAGE,
                        title=f"Page {i}", text="page text", order=i)
            n.logical_doc_id = "doc_test"
            pages.append(n)

        all_nodes = sections + pages
        skip = {"DocumentLogical", "DocRevision", "Book"}
        nodes_by_label = defaultdict(list)
        for node in all_nodes:
            label = node.type.value if hasattr(node.type, "value") else str(node.type)
            if label not in skip:
                nodes_by_label[label].append(node)

        assert "Section" in nodes_by_label
        assert "Page" in nodes_by_label
        assert len(nodes_by_label["Section"]) == 3
        assert len(nodes_by_label["Page"]) == 2


# ── Test 7: Exporter dual-write to blob/vector stores ───────────────────────


class _FakeBlobStore:
    def __init__(self):
        self.puts: dict[str, str] = {}

    def put(self, key, content, *, content_type="text/plain"):
        self.puts[key] = content
        return key

    def get(self, key):
        return self.puts.get(key)

    def delete(self, key):
        self.puts.pop(key, None)

    def exists(self, key):
        return key in self.puts

    def delete_prefix(self, prefix):
        return 0


class _FakeVectorStore:
    def __init__(self):
        self.batches: list[list[tuple]] = []

    def upsert(self, id, embedding, *, metadata=None):
        self.upsert_batch([(id, embedding, metadata)])

    def upsert_batch(self, items):
        self.batches.append(list(items))

    def query(self, embedding, top_k=10, *, filters=None):
        return []

    def point_id_for(self, node_id):
        return f"point_{node_id}"

    def delete(self, id):
        pass

    def delete_by_filter(self, filters):
        pass


class TestExporterDualWrite:
    def _plan(self):
        from src.document.versioning import DocumentRevisionPlan

        return DocumentRevisionPlan(
            logical_id="doc_test",
            revision_id="doc_test:r1",
            version_number=1,
            content_hash="abc123",
            content_root_id="doc_test:r1::root",
            title="Test",
            source_filename="test.pdf",
            tenant_id="default",
        )

    def _make_node(self, node_id="n1", *, text="Hello world", embedding=None, visual=None):
        from src.models import DKGNode, NodeType

        return DKGNode(
            id=node_id, type=NodeType.SECTION, title="T", text=text, order=0,
            embedding=embedding, visual_content=visual,
        )

    def test_dual_write_chunk_puts_text_and_sets_blob_key(self):
        from src.exporter.exporter import Neo4jExporter

        blob_store, vector_store = _FakeBlobStore(), _FakeVectorStore()
        exporter = Neo4jExporter(output_dir="output/_test_dual_write", blob_store=blob_store, vector_store=vector_store)
        node = self._make_node()
        plan = self._plan()

        exporter._dual_write_chunk([node], plan)

        assert node.blob_key_text == "default/doc_test/doc_test:r1/n1/text"
        assert blob_store.get(node.blob_key_text) == "Hello world"

    def test_dual_write_chunk_batches_embeddings(self):
        from src.exporter.exporter import Neo4jExporter

        blob_store, vector_store = _FakeBlobStore(), _FakeVectorStore()
        exporter = Neo4jExporter(output_dir="output/_test_dual_write", blob_store=blob_store, vector_store=vector_store)
        nodes = [self._make_node(f"n{i}", embedding=[0.1, 0.2]) for i in range(3)]
        plan = self._plan()

        exporter._dual_write_chunk(nodes, plan)

        assert len(vector_store.batches) == 1  # one batched call, not per-node
        assert len(vector_store.batches[0]) == 3

    def test_dual_write_chunk_skips_nodes_without_text_or_embedding(self):
        from src.exporter.exporter import Neo4jExporter

        blob_store, vector_store = _FakeBlobStore(), _FakeVectorStore()
        exporter = Neo4jExporter(output_dir="output/_test_dual_write", blob_store=blob_store, vector_store=vector_store)
        node = self._make_node(text="", embedding=None)
        plan = self._plan()

        exporter._dual_write_chunk([node], plan)

        assert node.blob_key_text is None
        assert blob_store.puts == {}
        assert vector_store.batches == []

    def test_node_to_param_dict_includes_blob_keys(self):
        from src.exporter.exporter import Neo4jExporter

        node = self._make_node()
        node.blob_key_text = "some/key/text"
        d = Neo4jExporter._node_to_param_dict(node)

        assert d["blob_key_text"] == "some/key/text"
        assert d["blob_key_visual"] is None

    def test_exporter_defaults_to_factory_stores_when_not_injected(self, monkeypatch):
        from src.exporter.exporter import Neo4jExporter
        from src.shared.storage.blob.local_store import LocalFsBlobStore
        from src.shared.storage.vector.memory_store import InMemoryVectorStore

        # Force the local/memory defaults so this doesn't depend on whatever
        # backend the local/deployed .env actually configures (e.g. minio/qdrant).
        import src.shared.config.settings as settings_mod
        import src.shared.storage.blob.factory as blob_factory_mod
        import src.shared.storage.vector.factory as vector_factory_mod

        monkeypatch.setattr(settings_mod, "BLOB_STORE_BACKEND", "local")
        monkeypatch.setattr(settings_mod, "VECTOR_STORE_BACKEND", "memory")
        blob_factory_mod._store_singleton = None
        vector_factory_mod._store_singleton = None
        try:
            exporter = Neo4jExporter(output_dir="output/_test_dual_write")

            assert isinstance(exporter.blob_store, LocalFsBlobStore)
            assert isinstance(exporter.vector_store, InMemoryVectorStore)
        finally:
            blob_factory_mod._store_singleton = None
            vector_factory_mod._store_singleton = None


# ── Test 8: Exporter edge confidence/provenance write-path ──────────────────


class TestExporterEdgeConfidence:
    def _edge(self, **kwargs):
        from src.models import DKGEdge, RelType

        defaults = dict(source_id="a", target_id="b", rel_type=RelType.CONTAINS)
        defaults.update(kwargs)
        return DKGEdge(**defaults)

    def test_edge_to_param_dict_has_required_keys(self):
        from src.exporter.exporter import Neo4jExporter

        edge = self._edge()
        d = Neo4jExporter._edge_to_param_dict(edge)

        for key in ("source_id", "target_id", "weight", "axis", "properties", "confidence", "confidence_tier"):
            assert key in d, f"Missing key: {key}"

    def test_edge_to_param_dict_defaults_extracted(self):
        from src.exporter.exporter import Neo4jExporter

        d = Neo4jExporter._edge_to_param_dict(self._edge())

        assert d["confidence"] == 1.0
        assert d["confidence_tier"] == "EXTRACTED"

    def test_edge_to_param_dict_no_enum_leaks_into_dict(self):
        """The dict must contain only JSON-safe values, no RelType/EdgeConfidenceTier objects."""
        from src.exporter.exporter import Neo4jExporter
        from src.models import EdgeConfidenceTier, RelType

        edge = self._edge(
            rel_type=RelType.SEMANTICALLY_SIMILAR,
            confidence=0.83,
            confidence_tier=EdgeConfidenceTier.INFERRED,
        )
        d = Neo4jExporter._edge_to_param_dict(edge)

        assert d["confidence_tier"] == "INFERRED"
        assert isinstance(d["confidence_tier"], str)
        for v in d.values():
            assert not isinstance(v, (RelType, EdgeConfidenceTier)), f"Found enum value: {v}"

    def test_edge_to_param_dict_handles_plain_string_tier(self):
        """confidence_tier may already be a plain str (not the enum) — must not crash."""
        from src.exporter.exporter import Neo4jExporter

        edge = self._edge(confidence_tier="AMBIGUOUS")
        d = Neo4jExporter._edge_to_param_dict(edge)

        assert d["confidence_tier"] == "AMBIGUOUS"

    def test_write_edge_csvs_includes_confidence_columns(self, tmp_path, monkeypatch):
        from src.exporter.exporter import Neo4jExporter
        from src.models import EdgeConfidenceTier, RelType

        # No store/vector_store injected below, so Neo4jExporter falls back to the
        # real factory defaults -- force in-memory so this doesn't require a real
        # Qdrant endpoint when the local/deployed .env configures that backend.
        import src.shared.config.settings as settings_mod
        import src.shared.storage.vector.factory as vector_factory_mod

        monkeypatch.setattr(settings_mod, "VECTOR_STORE_BACKEND", "memory")
        vector_factory_mod._store_singleton = None
        try:
            exporter = Neo4jExporter(output_dir=str(tmp_path))
            edges = [
                self._edge(rel_type=RelType.CONTAINS, axis=1),
                self._edge(
                    rel_type=RelType.SEMANTICALLY_SIMILAR,
                    axis=2,
                    confidence=0.91,
                    confidence_tier=EdgeConfidenceTier.INFERRED,
                ),
            ]

            exporter._write_edge_csvs(edges)

            import csv

            with open(tmp_path / "edges" / "axis1_structural.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            assert rows[0]["confidence"] == "1.0"
            assert rows[0]["confidence_tier"] == "EXTRACTED"

            with open(tmp_path / "edges" / "axis2_semantic.csv", newline="") as f:
                rows = list(csv.DictReader(f))
            assert rows[0]["confidence"] == "0.91"
            assert rows[0]["confidence_tier"] == "INFERRED"
        finally:
            vector_factory_mod._store_singleton = None
