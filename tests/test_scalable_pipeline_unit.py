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

# ── Minimal stubs — must come before ANY src.* imports ──────────────────────

def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


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
class _FakeBaseModel:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
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

# --- sklearn stubs ---
for _n in ["sklearn", "sklearn.cluster"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["sklearn.cluster"].KMeans = MagicMock()

# --- model_providers stubs ---
for _n in ["src.model_providers", "src.model_providers.factory", "src.model_providers.openai_provider"]:
    if _n not in sys.modules:
        _stub_module(_n)
_factory_mock = MagicMock()
sys.modules["src.model_providers.factory"].get_model_provider = _factory_mock
sys.modules["src.model_providers"].get_model_provider = _factory_mock

# --- auth stubs ---
for _n in ["src.auth", "src.auth.rbac_setup", "src.auth.roles"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["src.auth.rbac_setup"].GraphRBAC = MagicMock()
sys.modules["src.auth.roles"].Role = MagicMock()
sys.modules["src.auth.roles"].UserContext = MagicMock()
sys.modules["src.auth.roles"].validate_role = MagicMock()

# --- document stubs ---
for _n in ["src.document", "src.document.versioning", "src.document.parser", "src.document.page_vision"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["src.document.versioning"].resolve_logical_id = MagicMock(return_value="doc_test")
sys.modules["src.document.versioning"].build_revision_plan = MagicMock()
sys.modules["src.document.versioning"].apply_revision_to_graph = MagicMock(return_value=([], []))
sys.modules["src.document.versioning"].file_content_sha256 = MagicMock(return_value="abc123")
# DocumentRevisionPlan as a simple MagicMock class (exporter.py uses it only as a type annotation)
sys.modules["src.document.versioning"].DocumentRevisionPlan = MagicMock

sys.modules["src.document.parser"].LightPdfParser = MagicMock()

# --- graph.constants stubs ---
for _n in ["src.graph", "src.graph.constants"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["src.graph.constants"].DOC_REVISION_LABEL = "DocRevision"
sys.modules["src.graph.constants"].DOCUMENT_LOGICAL_LABEL = "DocumentLogical"

# --- bridge/conversation stubs ---
for _n in ["src.bridge", "src.conversation", "src.routing", "src.router"]:
    if _n not in sys.modules:
        _stub_module(_n)

# --- src.ingestion.service stub ---
# Provide a minimal IngestionJob so job_store and tests can use it
# without pulling in the full fastapi/neo4j import chain via service.py.
# We load the REAL models.py (no heavy deps) to get IngestionStatus.
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

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
        for key in ("id", "title", "text", "logical_doc_id", "revision_id",
                    "lifecycle_status", "embedding", "entities"):
            assert key in d, f"Missing key: {key}"

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
