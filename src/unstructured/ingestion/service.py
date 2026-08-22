from __future__ import annotations

import contextlib
import mimetypes
import os
import re
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Callable, Dict, List, Optional

from fastapi import UploadFile
from neo4j.exceptions import ClientError

from ...shared.config.settings import (
    AUTO_LOAD_TO_NEO4J,
    CHAT_PROVIDER_API_KEY,
    CLEANUP_TMP_INGEST,
    CORPUS_MAX_FILES,
    CORPUS_MAX_PDF_PAGES,
    CYPHER_INGEST_SKIP_GENAI,
    DEFAULT_TENANT_ID,
    DOC_SKIP_DUPLICATE_HASH,
    ENABLE_PAGE_VISION,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OPENAI_API_KEY,
    REDIS_URL,
    STORE_INGESTION_ARTIFACTS,
)
from ...shared.model_providers.errors import ModelRateLimitError
from ..document.graph_snapshot import X1_STAGE, X2_STAGE, write_snapshot
from ..document.page_report import write_page_report
from ..document.page_validation import check_construction_coverage
from ..document.versioning import (
    DocumentRevisionPlan,
    apply_revision_to_graph,
    build_revision_plan,
    file_content_sha256,
    resolve_logical_id,
    source_file_blob_key,
)
from ..document.parser_base import DocumentParser
from ..document.parser_registry import get_parser, supported_extensions
from ...shared.model_providers.base import ModelProvider
from ...shared.model_providers.factory import get_chat_provider
from ..models import NodeType
from ..exporter.exporter import Neo4jExporter
from ..models import DKGEdge, DKGNode
from ...shared.storage.blob.base import BlobStore
from ...shared.storage.blob.factory import get_blob_store
from ...shared.storage.vector.base import VectorStore
from ...shared.storage.vector.factory import get_vector_store
from ...pipeline.ingestion.models import IngestionStatus
from ...pipeline.ingestion.job_store import JobStore, get_job_store
from ...pipeline.ingestion.queue import enqueue_ingest
from .triage import check_duplicate, check_structural_sanity

from ...shared.auth.rbac_setup import GraphRBAC
from ...shared.neo4j.driver import get_neo4j_driver

if TYPE_CHECKING:
    from ..graph.construction_service import GraphConstructionService


@dataclass
class IngestionJob:
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
    tenant_id: Optional[str] = None
    skipped_duplicate: bool = False
    # False for corpus-scanned child jobs, whose input_path points directly
    # at a file in the user's own source directory (not a tmp_ingest/ copy) —
    # _cleanup_job_inputs must never unlink() it.
    owns_input_path: bool = True
    # Populated on a "corpus" job as it fans out per-file "unstructured" jobs.
    child_job_ids: List[str] = field(default_factory=list)


_TENANT_STAMPING_RE = re.compile(r"\b(?:CREATE|MERGE)\s*\(", re.I)
_HAS_TENANT_PROP_RE = re.compile(r"tenant_id\s*:", re.I)


def warn_missing_tenant_stamps(statements: list[str]) -> list[str]:
    """
    Non-blocking heuristic: flag CREATE/MERGE statements that don't mention a
    tenant_id property, for arbitrary uploaded Cypher (see _process_cypher's
    docstring — this route can't be mechanically guaranteed tenant-safe).
    """
    warnings: list[str] = []
    for idx, stmt in enumerate(statements, start=1):
        if _TENANT_STAMPING_RE.search(stmt) and not _HAS_TENANT_PROP_RE.search(stmt):
            preview = " ".join(stmt.split())[:120]
            warnings.append(
                f"Statement {idx} has a CREATE/MERGE with no tenant_id property: {preview}..."
            )
    return warnings


class IngestionManager:
    def __init__(
        self,
        store: Optional[JobStore] = None,
        *,
        parser_factory: Callable[[Path], DocumentParser] = get_parser,
        model_provider: Optional[ModelProvider] = None,
        blob_store: Optional[BlobStore] = None,
        vector_store: Optional[VectorStore] = None,
        exporter_factory: Callable[..., Neo4jExporter] = Neo4jExporter,
        graph_service_factory: Optional[Callable[[], "GraphConstructionService"]] = None,
    ):
        self.store: JobStore = store if store is not None else get_job_store()
        self.parser_factory = parser_factory
        self.model_provider = model_provider or get_chat_provider()
        self.blob_store = blob_store or get_blob_store()
        self.vector_store = vector_store or get_vector_store()
        self.exporter_factory = exporter_factory
        # Defaults resolved lazily at point of use (see _process_unstructured)
        # rather than imported at module top, matching the existing lazy
        # `from ..semantic.axis2 import Axis2Builder` style already in this
        # file -- GraphConstructionService pulls in axis1/axis2's own
        # dependency chains, no need to pay for that on every import of
        # this module.
        self.graph_service_factory = graph_service_factory
        self.temp_dir = Path("tmp_ingest")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_base = Path("output/ingestion")
        if STORE_INGESTION_ARTIFACTS:
            self.output_base.mkdir(parents=True, exist_ok=True)

    # ── Public submission API ──────────────────────────────────────────────

    def submit_unstructured(
        self,
        upload: UploadFile,
        tenant_id: str,
        job_name: Optional[str] = None,
        doc_key: Optional[str] = None,
    ) -> IngestionJob:
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id is required for ingestion.")
        job = self._create_job("unstructured", job_name=job_name)
        job.tenant_id = tenant_id.strip()
        job.doc_key = doc_key
        job.input_path = self._save_upload(upload, job.id)
        if STORE_INGESTION_ARTIFACTS:
            job.output_dir = self.output_base / job.id
            job.output_dir.mkdir(parents=True, exist_ok=True)
        self.store.save(job)
        self._log(job, f"Created unstructured ingestion job: {job.name or job.id}")
        return job

    def submit_cypher(
        self,
        upload: UploadFile,
        tenant_id: str,
        job_name: Optional[str] = None,
        cypher_params: Optional[Dict[str, object]] = None,
    ) -> IngestionJob:
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id is required for ingestion.")
        job = self._create_job("cypher", job_name=job_name)
        job.tenant_id = tenant_id.strip()
        job.cypher_params = cypher_params or None
        job.input_path = self._save_upload(upload, job.id)
        if STORE_INGESTION_ARTIFACTS:
            job.output_dir = self.output_base / job.id
            job.output_dir.mkdir(parents=True, exist_ok=True)
        self.store.save(job)
        self._log(job, f"Created cypher ingestion job: {job.name or job.id}")
        return job

    def submit_corpus(
        self,
        source: str | Path,
        tenant_id: str,
        *,
        job_name: Optional[str] = None,
        doc_key_prefix: Optional[str] = None,
    ) -> IngestionJob:
        """
        Create a job that scans a directory (recursively) or a manifest file
        (newline-delimited absolute paths) and fans out one "unstructured"
        job per accepted file, after a cheap dedup + structural-sanity pass.
        """
        if not tenant_id or not tenant_id.strip():
            raise ValueError("tenant_id is required for ingestion.")
        source_path = Path(source)
        if not source_path.exists():
            raise FileNotFoundError(f"Corpus source not found: {source_path}")

        job = self._create_job("corpus", job_name=job_name)
        job.tenant_id = tenant_id.strip()
        job.doc_key = doc_key_prefix
        job.input_path = source_path
        job.owns_input_path = False  # the user's own directory/manifest, never a copy
        self.store.save(job)
        self._log(job, f"Created corpus ingestion job for source: {source_path}")
        return job

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        return self.store.get(job_id)

    def list_job_ids(self, limit: int = 100) -> List[str]:
        return self.store.list_ids(limit=limit)

    # ── Job execution ──────────────────────────────────────────────────────

    def run_job(self, job_id: str) -> None:
        job = self.store.get(job_id)
        if job is None:
            return

        job.started_at = datetime.utcnow()
        self._set_status(job, IngestionStatus.validating, "Validating ingestion inputs")
        self._ensure_rbac_schema(job)

        try:
            if job.type == "unstructured":
                self._process_unstructured(job)
            elif job.type == "cypher":
                self._process_cypher(job)
            elif job.type == "corpus":
                self._process_corpus(job)
            else:
                raise ValueError(f"Unsupported ingestion type: {job.type}")

            job.finished_at = datetime.utcnow()
            # A document that parsed, embedded and summarised but never
            # reached Neo4j has produced nothing retrievable. Reporting that
            # as success is how 46 documents in a row appeared to ingest
            # while the corpus count did not move.
            if job.neo4j_load_status == "failed":
                self._set_status(
                    job,
                    IngestionStatus.failed,
                    f"Graph load failed: {job.neo4j_load_message}",
                )
            else:
                self._set_status(job, IngestionStatus.completed, "Job completed successfully")
            self._clear_structured_query_caches()
        except ModelRateLimitError as exc:
            # Surfaced on its own terms rather than behind "Job failed:".
            # This is the one failure the user can fix directly, and the
            # message already says how -- burying it under a generic prefix
            # is why an exhausted daily quota looked like a crash.
            job.finished_at = datetime.utcnow()
            job.error = str(exc)
            self._set_status(job, IngestionStatus.failed, f"⚠ {job.error}")
        except Exception as exc:
            job.finished_at = datetime.utcnow()
            job.error = str(exc)
            self._set_status(job, IngestionStatus.failed, f"Job failed: {job.error}")
        finally:
            self._cleanup_job_inputs(job)

    @staticmethod
    def _clear_structured_query_caches() -> None:
        """A completed ingestion job may add node/relationship types the
        structured query path hasn't seen — its schema and entity-summary
        caches are process-lifetime by design (re-querying Neo4j on every
        request would be wasteful), so they need an explicit bust here
        instead of silently staying stale until the next restart. Deferred
        imports: avoids constructing the structured-retrieval singleton (and
        its Neo4j/RBAC/LLM collaborators) as a side effect of importing this
        module, and there's no reverse dependency either way to worry about.
        """
        try:
            from ...structured.retrieval.graph import retriever as _structured_retriever

            _structured_retriever.clear_schema_cache()
        except Exception:
            pass
        try:
            from ...interface.routing import clear_structured_entity_cache

            clear_structured_entity_cache()
        except Exception:
            pass

    # ── Internal helpers ───────────────────────────────────────────────────

    def _ensure_rbac_schema(self, job: IngestionJob) -> None:
        """Seed RBAC in Neo4j only when the schema is not already present."""
        rbac = GraphRBAC(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
        try:
            if rbac.is_initialized():
                self._log(job, "RBAC schema already present; skipping setup")
                return
            self._log(job, "RBAC schema missing; running setup")
            rbac.setup_schema()
        finally:
            rbac.close()

    def _persist_source_file(self, job: IngestionJob, plan: DocumentRevisionPlan) -> None:
        """Save the original uploaded file to blob storage so it survives
        _cleanup_job_inputs' unlink() — powers the document-viewer side panel
        (GET /documents/{id}/file). Best-effort: a failure here must not fail
        the ingestion job, since the graph load already succeeded by the time
        this runs.
        """
        if not job.input_path or not job.input_path.exists():
            return
        try:
            content_type = mimetypes.guess_type(str(job.input_path))[0] or "application/octet-stream"
            key = source_file_blob_key(
                tenant_id=plan.tenant_id,
                logical_id=plan.logical_id,
                revision_id=plan.revision_id,
                source_filename=plan.source_filename,
            )
            self.blob_store.put_bytes(key, job.input_path.read_bytes(), content_type=content_type)
            self._log(job, f"Saved original source file to blob storage: {key}")
        except Exception as exc:
            self._log(job, f"Source file persist skipped: {exc}")

    def _cleanup_job_inputs(self, job: IngestionJob) -> None:
        if not job or not job.owns_input_path:
            return
        if not CLEANUP_TMP_INGEST:
            return
        if not job.input_path:
            return
        try:
            if job.input_path.exists():
                job.input_path.unlink()
                self._log(job, f"Cleaned up temp input: {job.input_path}")
        except Exception as exc:
            self._log(job, f"Temp cleanup failed for {job.input_path}: {exc}")

    @contextlib.contextmanager
    def _doc_lock(self, logical_id: str, tenant_id: str = ""):
        """
        Acquire a per-logical-document Redis lock (if Redis is configured).

        Prevents two workers from racing to install a new revision for the
        same logical document. Documents with *different* logical IDs are
        unaffected and process fully in parallel. The lock key is namespaced
        by tenant_id so two tenants that coincidentally pick the same
        logical_id slug don't needlessly serialize on each other's ingest.
        """
        if not logical_id or not REDIS_URL:
            yield
            return

        try:
            import redis as _redis

            conn = _redis.from_url(REDIS_URL, decode_responses=False)
            lock_key = f"ingest:lock:{tenant_id or DEFAULT_TENANT_ID}:{logical_id}"
            lock = conn.lock(lock_key, timeout=1800, blocking_timeout=1800)
            acquired = lock.acquire(blocking=True)
            try:
                yield
            finally:
                if acquired:
                    with contextlib.suppress(Exception):
                        lock.release()
        except Exception:
            # Redis unavailable: proceed without lock (best-effort).
            yield

    def _logical_id_for_content(
        self, driver, exporter, job: IngestionJob, logical_id: str
    ) -> str:
        """Reuse the logical id this exact content already has, if it has one.

        A logical id is derived from the filename unless the caller supplies a
        doc_key, and supersede only ever fires within one logical id. So the
        same PDF ingested twice under different doc_keys became two documents
        rather than two revisions of one -- three copies of one file, all
        titled from the same filename and so indistinguishable in the document
        picker. Adopting the first logical id makes the re-ingest a revision,
        which the existing supersede path expires and deletes.

        Best-effort: on any failure the caller's own logical id stands, which
        is the pre-existing behaviour.
        """
        if not job.content_hash:
            return logical_id
        try:
            with driver.session() as session:
                owner = exporter.logical_id_holding_hash(
                    session, job.content_hash, job.tenant_id or "default"
                )
        except Exception as exc:
            self._log(job, f"Content-hash lookup failed (keeping {logical_id}): {exc}")
            return logical_id
        if not owner or owner == logical_id:
            return logical_id
        job.logical_doc_id = owner
        self.store.save(job)
        self._log(
            job,
            f"Identical content is already document {owner!r}; ingesting as a new "
            f"revision of it rather than as a second copy under {logical_id!r}.",
        )
        return owner

    def _process_unstructured(self, job: IngestionJob) -> None:
        self._set_status(job, IngestionStatus.parsing, "Parsing document")
        if not job.input_path or not job.input_path.exists():
            raise FileNotFoundError("Uploaded file was not saved correctly.")

        if job.input_path.suffix.lower() not in supported_extensions():
            raise ValueError(
                f"No parser registered for file type {job.input_path.suffix!r}."
            )

        logical_id = resolve_logical_id(
            job.input_path, doc_key=job.doc_key, job_id=job.id
        )
        job.logical_doc_id = logical_id
        self.store.save(job)

        # Fast duplicate check (no lock needed — reading is safe). The logical
        # id is settled first and unconditionally: DOC_SKIP_DUPLICATE_HASH
        # decides whether an identical re-upload is skipped, but even when it
        # is off the re-upload must land as a REVISION of the document that
        # already holds this content, never as a second copy of it.
        if AUTO_LOAD_TO_NEO4J:
            job.content_hash = file_content_sha256(job.input_path)
            exporter_probe = self.exporter_factory(
                output_dir=str(job.output_dir) if job.output_dir else Path(".")
            )
            driver = get_neo4j_driver()
            logical_id = self._logical_id_for_content(
                driver, exporter_probe, job, logical_id
            )
            if DOC_SKIP_DUPLICATE_HASH and check_duplicate(
                job.input_path, logical_id=logical_id, exporter=exporter_probe, driver=driver
            ):
                job.skipped_duplicate = True
                job.neo4j_load_status = "skipped"
                job.neo4j_load_message = (
                    "Identical content already ACTIVE for this logical document; "
                    "ingest skipped (no parse)."
                )
                self._log(job, job.neo4j_load_message)
                return

        if self.graph_service_factory is not None:
            graph_service = self.graph_service_factory()
        else:
            from ..graph.construction_service import GraphConstructionService

            graph_service = GraphConstructionService()

        parser = self.parser_factory(job.input_path)
        ir = parser.parse_ir(str(job.input_path))
        nodes, edges, _chunks = graph_service.build_structure(ir)
        self._log(job, f"Parsed {len(nodes)} nodes and {len(edges)} edges")

        content_root_id = next(
            (
                n.id
                for n in nodes
                if n.type
                in (
                    NodeType.DOCUMENT,
                    NodeType.DOCUMENT.value,
                    NodeType.BOOK,
                    NodeType.BOOK.value,
                )
            ),
            f"doc_{job.id}",
        )
        exporter = self.exporter_factory(output_dir=str(job.output_dir) if job.output_dir else Path("."))
        version_number = 1
        if AUTO_LOAD_TO_NEO4J:
            driver = get_neo4j_driver()
            with driver.session() as session:
                version_number = exporter.next_version_number(session, logical_id)

        plan = build_revision_plan(
            job.input_path,
            tenant_id=job.tenant_id or DEFAULT_TENANT_ID,
            doc_key=job.doc_key,
            job_id=job.id,
            version_number=version_number,
            content_root_id=content_root_id,
            logical_id=logical_id,
        )
        nodes, edges = apply_revision_to_graph(nodes, edges, plan)
        job.logical_doc_id = plan.logical_id
        job.revision_id = plan.revision_id
        job.content_hash = plan.content_hash
        job.version_number = plan.version_number
        self.store.save(job)
        self._log(
            job,
            f"Revision plan: logical={plan.logical_id} rev={plan.revision_id} "
            f"v{plan.version_number} hash={plan.content_hash[:12]}…",
        )

        # X1 snapshot for the graph-inspector UI: the structural graph as
        # it exists right after parsing + lineage stamping, before Axis-2
        # touches anything. Transient in-memory state otherwise -- nothing
        # else persists this shape once semantic enrichment adds its own
        # edges on top, so it must be captured here or not at all. Never
        # allowed to fail the ingestion job (a debugging aid, not a
        # correctness requirement).
        try:
            write_snapshot(
                self.blob_store,
                X1_STAGE,
                tenant_id=plan.tenant_id,
                logical_doc_id=plan.logical_id,
                revision_id=plan.revision_id,
                nodes=nodes,
                edges=edges,
            )
        except Exception as exc:
            self._log(job, f"X1 graph snapshot skipped: {exc}")

        # Structural page-coverage check: catches a total or partial
        # heading-detection collapse (every page's text landing in one
        # catch-all section, or a page ending up with no text at all)
        # immediately, using the exact `ir`/`nodes` this run just produced
        # -- not a re-parse, so it can't disagree with what was actually
        # ingested. Persisted so /ingest/quality/{id}/pages doesn't need to
        # re-parse the source file later. Never allowed to fail the
        # ingestion job, same posture as the X1/X2 snapshots above.
        try:
            coverage_report = check_construction_coverage(ir, nodes)
            write_page_report(
                self.blob_store,
                tenant_id=plan.tenant_id,
                logical_doc_id=plan.logical_id,
                revision_id=plan.revision_id,
                report=coverage_report,
            )
            summary = coverage_report["summary"]
            if summary["requires_reprocessing"]:
                self._log(
                    job,
                    f"WARNING: structural coverage check flagged {summary['pages_failing']}/"
                    f"{summary['page_count']} page(s) (avg coverage "
                    f"{summary['avg_coverage'] * 100:.0f}%) — possible heading-detection collapse "
                    "or OCR/extraction gap. See /ingest/quality/{}/pages.".format(plan.logical_id),
                )
            else:
                self._log(job, f"Structural coverage check passed ({summary['page_count']} page(s))")
        except Exception as exc:
            self._log(job, f"Structural coverage check skipped: {exc}")

        if (
            ENABLE_PAGE_VISION
            and OPENAI_API_KEY
            and job.input_path.suffix.lower() == ".pdf"
        ):
            self._set_status(
                job,
                IngestionStatus.vision_enrichment,
                "Vision enrichment (tables, charts, diagrams on selected pages)",
            )
            try:
                from ..document.page_vision import PageVisionEnricher

                count = PageVisionEnricher(api_key=OPENAI_API_KEY).enrich_document(
                    job.input_path, nodes
                )
                self._log(job, f"Vision enriched {count} page(s)")
            except Exception as exc:
                self._log(job, f"Vision enrichment skipped: {exc}")

        if CHAT_PROVIDER_API_KEY:
            self._set_status(job, IngestionStatus.semantic_enrichment, "Running semantic enrichment (Axis 2)")
            try:
                nodes, semantic_edges = graph_service.build_ideas(nodes, run_llm_pass=True)
                edges += semantic_edges
                self._log(job, f"Added {len(semantic_edges)} semantic edges")
            except Exception as exc:
                self._log(job, f"Semantic enrichment skipped: {exc}")

            # X2 snapshot: structural + semantic edges together, right
            # after Axis-2 finishes and before Neo4j load -- same
            # best-effort, never-fails-the-job reasoning as the X1
            # snapshot above.
            try:
                write_snapshot(
                    self.blob_store,
                    X2_STAGE,
                    tenant_id=plan.tenant_id,
                    logical_doc_id=plan.logical_id,
                    revision_id=plan.revision_id,
                    nodes=nodes,
                    edges=edges,
                )
            except Exception as exc:
                self._log(job, f"X2 graph snapshot skipped: {exc}")

            self._set_status(job, IngestionStatus.chapter_summarization, "Summarizing chapters")
            try:
                from ..semantic.chapter_summary import ChapterSummaryBuilder

                nodes = ChapterSummaryBuilder().build(nodes, edges)
                summarized = sum(1 for n in nodes if getattr(n, "summary", None))
                self._log(job, f"Summarized {summarized} chapter(s)")
            except Exception as exc:
                self._log(job, f"Chapter summarization skipped: {exc}")
        else:
            self._log(job, "No chat provider API key configured; skipping semantic enrichment")

        if STORE_INGESTION_ARTIFACTS and job.output_dir:
            self._set_status(job, IngestionStatus.exporting, "Exporting Neo4j import artifacts")
            exporter.export(nodes, edges)

        # Acquire per-logical-doc lock only around the Neo4j revision install.
        # Workers processing different documents are never blocked.
        if AUTO_LOAD_TO_NEO4J:
            self._set_status(job, IngestionStatus.exporting, "Loading graph into Neo4j")
            with self._doc_lock(plan.logical_id, plan.tenant_id):
                try:
                    load_meta = exporter.load_to_neo4j(
                        nodes,
                        edges,
                        NEO4J_URI,
                        NEO4J_USER,
                        NEO4J_PASSWORD,
                        revision_plan=plan,
                        skip_if_duplicate_hash=DOC_SKIP_DUPLICATE_HASH,
                    )
                    if load_meta.get("skipped_duplicate"):
                        job.skipped_duplicate = True
                        job.neo4j_load_status = "skipped"
                        job.neo4j_load_message = (
                            "Identical content already ACTIVE for this logical document; "
                            "ingest skipped."
                        )
                        self._log(job, job.neo4j_load_message)
                    else:
                        job.neo4j_load_status = "success"
                        job.neo4j_load_message = (
                            f"Graph loaded (revision {plan.revision_id})"
                        )
                        self._log(job, job.neo4j_load_message)
                        self._persist_source_file(job, plan)
                except Exception as exc:
                    # Logged with a traceback, not just str(exc). This failure
                    # cost a 104-document run: every document reported
                    # "dictionary update sequence element #0 has length 1",
                    # which names the symptom and not one frame of where it
                    # happened, and the message was the only record.
                    job.neo4j_load_status = "failed"
                    job.neo4j_load_message = str(exc)
                    logger.exception("Neo4j load failed for job %s", job.id)
                    self._log(job, f"Neo4j load failed: {exc}")
        else:
            job.neo4j_load_status = "skipped"
            job.neo4j_load_message = "AUTO_LOAD_TO_NEO4J disabled"
            self._log(job, "Neo4j load skipped")

    def _walk_corpus_source(self, source: Path) -> List[Path]:
        """Directory: recursive walk filtered by supported_extensions(), sorted.
        File: treated as a newline-delimited manifest of absolute paths."""
        if source.is_dir():
            exts = supported_extensions()
            candidates: List[Path] = []
            for dirpath, dirnames, filenames in os.walk(source):
                dirnames[:] = [d for d in dirnames if not d.startswith(".")]
                for name in filenames:
                    if name.startswith("."):
                        continue
                    path = Path(dirpath) / name
                    if path.suffix.lower() in exts:
                        candidates.append(path)
            return sorted(candidates)

        if source.is_file():
            return self._parse_manifest(source)

        raise FileNotFoundError(f"Corpus source not found: {source}")

    def _parse_manifest(self, manifest_path: Path) -> List[Path]:
        seen: dict[Path, None] = {}
        for lineno, raw_line in enumerate(
            manifest_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line)
            if not path.is_absolute():
                raise ValueError(
                    f"Manifest {manifest_path} line {lineno}: relative path {line!r} "
                    "is not allowed — only absolute paths are supported."
                )
            seen.setdefault(path, None)
        return list(seen.keys())

    def _process_corpus(self, job: IngestionJob) -> None:
        self._set_status(job, IngestionStatus.scanning, "Scanning corpus source")
        if not job.input_path:
            raise FileNotFoundError("Corpus source path was not set on the job.")

        candidates = self._walk_corpus_source(job.input_path)
        self._log(job, f"Found {len(candidates)} candidate file(s)")
        if len(candidates) > CORPUS_MAX_FILES:
            raise ValueError(
                f"Corpus source has {len(candidates)} files, exceeding "
                f"CORPUS_MAX_FILES={CORPUS_MAX_FILES}"
            )

        exts = supported_extensions()
        driver = get_neo4j_driver() if AUTO_LOAD_TO_NEO4J else None
        accepted = 0
        skipped = 0

        for i, path in enumerate(candidates, start=1):
            reason = check_structural_sanity(
                path, supported_extensions=exts, max_pdf_pages=CORPUS_MAX_PDF_PAGES
            )
            if reason is None and driver is not None and DOC_SKIP_DUPLICATE_HASH:
                child_doc_key = f"{job.doc_key}:{path.stem}" if job.doc_key else None
                logical_id = resolve_logical_id(path, doc_key=child_doc_key, job_id=job.id)
                try:
                    exporter_probe = self.exporter_factory(output_dir=Path("."))
                    reason = check_duplicate(
                        path, logical_id=logical_id, exporter=exporter_probe, driver=driver
                    )
                except Exception as exc:
                    # Best-effort: the child job's own duplicate check (in
                    # _process_unstructured) is the authoritative, fail-loud gate.
                    self._log(job, f"Duplicate check failed for {path.name} (accepting): {exc}")
                    reason = None

            if reason:
                skipped += 1
                self._log(job, f"Skipped {path.name}: {reason}")
                continue

            child_doc_key = f"{job.doc_key}:{path.stem}" if job.doc_key else None
            child = self._create_job("unstructured", job_name=path.name)
            child.tenant_id = job.tenant_id
            child.doc_key = child_doc_key
            child.input_path = path
            child.owns_input_path = False
            self.store.save(child)
            job.child_job_ids.append(child.id)
            accepted += 1

            if enqueue_ingest(child.id) is None:
                self.run_job(child.id)

            if i % 100 == 0:
                self._log(job, f"Progress: {i}/{len(candidates)} scanned")
                self.store.save(job)

        self.store.save(job)
        self._log(
            job,
            f"Corpus scan complete: {len(candidates)} found, {accepted} accepted/queued, "
            f"{skipped} skipped",
        )

    def _process_cypher(self, job: IngestionJob) -> None:
        """
        Execute a user-provided Cypher file against Neo4j.

        Intended for loading arbitrary schemas/datasets.

        NOT covered by the automatic tenant-stamping guarantee that
        DKGNode/DKGEdge ingestion gets: arbitrary uploaded CREATE/MERGE
        statements can't be mechanically forced to carry tenant_id. This
        route already requires an admin session (resolve_admin_session) —
        treat it as admin-trust-only in a genuinely multi-tenant deployment,
        and rely on the warning below, not a hard guarantee.
        """
        self._set_status(job, IngestionStatus.parsing, "Executing Cypher script")
        if not job.input_path or not job.input_path.exists():
            raise FileNotFoundError("Uploaded Cypher file was not saved correctly.")

        cypher_text = job.input_path.read_text(encoding="utf-8")
        statements, params = self._parse_cypher_script(cypher_text)
        if job.cypher_params:
            params = {**params, **job.cypher_params}
        if not statements:
            raise ValueError("Cypher file contained no statements.")
        params.setdefault("tenant_id", job.tenant_id or DEFAULT_TENANT_ID)

        for warning in warn_missing_tenant_stamps(statements):
            self._log(job, f"WARNING: {warning}")

        driver = get_neo4j_driver()
        with driver.session() as session:
            for idx, stmt in enumerate(statements, start=1):
                preview = " ".join(stmt.split())[:120]
                self._log(job, f"Running statement {idx}/{len(statements)}: {preview}...")
                try:
                    session.run(stmt, **params).consume()
                except ClientError as exc:
                    code = getattr(exc, "code", "") or ""
                    if code in {
                        "Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists",
                        "Neo.ClientError.Schema.IndexAlreadyExists",
                        "Neo.ClientError.Schema.ConstraintAlreadyExists",
                    }:
                        self._log(job, f"Skipping non-fatal schema error ({code}): {exc.message}")
                        continue
                    raise

        job.neo4j_load_status = "success"
        job.neo4j_load_message = f"Executed {len(statements)} Cypher statements successfully"
        self._log(job, job.neo4j_load_message)

    def _parse_cypher_script(self, raw_text: str) -> tuple[list[str], dict]:
        """
        Parse a Cypher script that may include Neo4j Browser directives like:
          :param key => 'value';
        """
        params: dict = {}
        cypher_lines: list[str] = []

        def _parse_param_value(value: str):
            v = value.strip().rstrip(";")
            if (v.startswith("'") and v.endswith("'")) or (v.startswith('"') and v.endswith('"')):
                return v[1:-1]
            low = v.lower()
            if low == "true":
                return True
            if low == "false":
                return False
            if low == "null":
                return None
            try:
                if "." in v:
                    return float(v)
                return int(v)
            except Exception:
                return v

        for line in raw_text.splitlines():
            stripped = line.strip()
            if not stripped:
                cypher_lines.append(line)
                continue
            if stripped.startswith(":"):
                if stripped.lower().startswith(":param"):
                    rest = stripped[len(":param"):].strip()
                    if "=>" in rest:
                        key, value = rest.split("=>", 1)
                        key = key.strip().lstrip("$")
                        if key:
                            params[key] = _parse_param_value(value)
                continue
            cypher_lines.append(line)

        cypher_text = "\n".join(cypher_lines)
        statements = [s.strip() for s in cypher_text.split(";") if s.strip()]
        if CYPHER_INGEST_SKIP_GENAI:
            filtered: list[str] = []
            for stmt in statements:
                s = stmt.lower()
                if "genai.vector.encode" in s or "ai.text.embed" in s:
                    continue
                if "db.create.setnodevectorproperty" in s:
                    continue
                filtered.append(stmt)
            statements = filtered
        return statements, params

    @staticmethod
    def _safe_upload_name(filename: Optional[str]) -> str:
        """The bare file name from whatever the client called this upload.

        An upload's filename is client-supplied text, not a path, and this
        one gets interpolated into a real one. Two things follow.

        It can carry directories. Picking a folder in the browser sends each
        file as `<folder>/<name>` (webkitRelativePath), so the save path named
        a subdirectory that was never created and every upload in a folder
        failed on FileNotFoundError.

        And it can carry `..`. A name like `../../etc/cron.d/job` would have
        written outside the ingest directory entirely. Taking the last
        component of both separators closes that off -- Windows clients send
        backslashes, which PurePosixPath does not treat as separators.
        """
        raw = (filename or "").replace("\\", "/")
        name = PurePosixPath(raw).name
        # `.`, `..` and the empty string all resolve to no name at all.
        if name in ("", ".", ".."):
            return "upload"
        return name

    def _save_upload(self, upload: UploadFile, job_id: str) -> Path:
        target = self.temp_dir / f"{job_id}_{self._safe_upload_name(upload.filename)}"
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as out_file:
            shutil.copyfileobj(upload.file, out_file)
        return target

    def _create_job(
        self,
        ingestion_type: str,
        job_name: Optional[str] = None,
    ) -> IngestionJob:
        job_id = uuid.uuid4().hex
        return IngestionJob(
            id=job_id,
            type=ingestion_type,
            name=job_name,
        )

    def _set_status(self, job: IngestionJob, status: IngestionStatus, message: Optional[str] = None) -> None:
        job.status = status
        if message:
            self._log(job, message)
        # Persist status change immediately so any observer (API, other worker) can read it.
        self.store.save(job)

    def _log(self, job: IngestionJob, message: str) -> None:
        timestamp = datetime.utcnow().isoformat() + "Z"
        entry = f"{timestamp} - {message}"
        job.logs.append(entry)
        if len(job.logs) > 200:
            job.logs = job.logs[-200:]
        self.store.append_log(job.id, entry)
