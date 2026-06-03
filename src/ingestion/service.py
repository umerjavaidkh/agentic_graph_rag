from __future__ import annotations

import contextlib
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import UploadFile
from neo4j.exceptions import ClientError

from ..config.settings import (
    AUTO_LOAD_TO_NEO4J,
    CLEANUP_TMP_INGEST,
    CYPHER_INGEST_SKIP_GENAI,
    DOC_SKIP_DUPLICATE_HASH,
    ENABLE_PAGE_VISION,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OPENAI_API_KEY,
    REDIS_URL,
    STORE_INGESTION_ARTIFACTS,
)
from ..document.versioning import (
    apply_revision_to_graph,
    build_revision_plan,
    file_content_sha256,
    resolve_logical_id,
)
from ..document.parser import LightPdfParser
from ..models import NodeType
from ..exporter.exporter import Neo4jExporter
from ..models import DKGEdge, DKGNode
from .models import IngestionStatus
from .job_store import JobStore, get_job_store

from ..auth.rbac_setup import GraphRBAC
from ..graph.driver import get_neo4j_driver


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
    skipped_duplicate: bool = False


class IngestionManager:
    def __init__(self, store: Optional[JobStore] = None):
        self.store: JobStore = store if store is not None else get_job_store()
        self.temp_dir = Path("tmp_ingest")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_base = Path("output/ingestion")
        if STORE_INGESTION_ARTIFACTS:
            self.output_base.mkdir(parents=True, exist_ok=True)

    # ── Public submission API ──────────────────────────────────────────────

    def submit_unstructured(
        self,
        upload: UploadFile,
        job_name: Optional[str] = None,
        doc_key: Optional[str] = None,
    ) -> IngestionJob:
        job = self._create_job("unstructured", job_name=job_name)
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
        job_name: Optional[str] = None,
        cypher_params: Optional[Dict[str, object]] = None,
    ) -> IngestionJob:
        job = self._create_job("cypher", job_name=job_name)
        job.cypher_params = cypher_params or None
        job.input_path = self._save_upload(upload, job.id)
        if STORE_INGESTION_ARTIFACTS:
            job.output_dir = self.output_base / job.id
            job.output_dir.mkdir(parents=True, exist_ok=True)
        self.store.save(job)
        self._log(job, f"Created cypher ingestion job: {job.name or job.id}")
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
            else:
                raise ValueError(f"Unsupported ingestion type: {job.type}")

            job.finished_at = datetime.utcnow()
            self._set_status(job, IngestionStatus.completed, "Job completed successfully")
        except Exception as exc:
            job.finished_at = datetime.utcnow()
            job.error = str(exc)
            self._set_status(job, IngestionStatus.failed, f"Job failed: {job.error}")
        finally:
            self._cleanup_job_inputs(job)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _ensure_rbac_schema(self, job: IngestionJob) -> None:
        """Seed RBAC in Neo4j only when the schema is not already present."""
        rbac = GraphRBAC(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)
        try:
            if rbac.is_initialized():
                self._log(job, "RBAC schema already present; skipping setup")
                return
            self._log(job, "RBAC schema missing; running setup")
            rbac.setup_schema("src/auth/rbac_schema.cypher")
        finally:
            rbac.close()

    def _cleanup_job_inputs(self, job: IngestionJob) -> None:
        if not CLEANUP_TMP_INGEST:
            return
        if not job or not job.input_path:
            return
        try:
            if job.input_path.exists():
                job.input_path.unlink()
                self._log(job, f"Cleaned up temp input: {job.input_path}")
        except Exception as exc:
            self._log(job, f"Temp cleanup failed for {job.input_path}: {exc}")

    @contextlib.contextmanager
    def _doc_lock(self, logical_id: str):
        """
        Acquire a per-logical-document Redis lock (if Redis is configured).

        Prevents two workers from racing to install a new revision for the
        same logical document. Documents with *different* logical IDs are
        unaffected and process fully in parallel.
        """
        if not logical_id or not REDIS_URL:
            yield
            return

        try:
            import redis as _redis

            conn = _redis.from_url(REDIS_URL, decode_responses=False)
            lock_key = f"ingest:lock:{logical_id}"
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

    def _process_unstructured(self, job: IngestionJob) -> None:
        self._set_status(job, IngestionStatus.parsing, "Parsing document")
        if not job.input_path or not job.input_path.exists():
            raise FileNotFoundError("Uploaded file was not saved correctly.")

        if job.input_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF ingestion is supported by the lightweight parser.")

        logical_id = resolve_logical_id(
            job.input_path, doc_key=job.doc_key, job_id=job.id
        )
        job.logical_doc_id = logical_id
        self.store.save(job)

        # Fast duplicate check (no lock needed — reading is safe).
        if AUTO_LOAD_TO_NEO4J and DOC_SKIP_DUPLICATE_HASH:
            content_hash = file_content_sha256(job.input_path)
            job.content_hash = content_hash
            exporter_probe = Neo4jExporter(
                output_dir=str(job.output_dir) if job.output_dir else Path(".")
            )
            driver = get_neo4j_driver()
            with driver.session() as session:
                if exporter_probe.active_revision_has_hash(
                    session, logical_id, content_hash
                ):
                    job.skipped_duplicate = True
                    job.neo4j_load_status = "skipped"
                    job.neo4j_load_message = (
                        "Identical content already ACTIVE for this logical document; "
                        "ingest skipped (no parse)."
                    )
                    self._log(job, job.neo4j_load_message)
                    return

        parser = LightPdfParser()
        nodes, edges = parser.parse(str(job.input_path))
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
        exporter = Neo4jExporter(output_dir=str(job.output_dir) if job.output_dir else Path("."))
        version_number = 1
        if AUTO_LOAD_TO_NEO4J:
            driver = get_neo4j_driver()
            with driver.session() as session:
                version_number = exporter.next_version_number(session, logical_id)

        plan = build_revision_plan(
            job.input_path,
            doc_key=job.doc_key,
            job_id=job.id,
            version_number=version_number,
            content_root_id=content_root_id,
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

        if OPENAI_API_KEY:
            self._set_status(job, IngestionStatus.semantic_enrichment, "Running semantic enrichment (Axis 2)")
            try:
                from ..semantic.axis2 import Axis2Builder

                builder = Axis2Builder(api_key=OPENAI_API_KEY)
                nodes, semantic_edges = builder.build(nodes, run_llm_pass=True)
                edges += semantic_edges
                self._log(job, f"Added {len(semantic_edges)} semantic edges")
            except Exception as exc:
                self._log(job, f"Semantic enrichment skipped: {exc}")
        else:
            self._log(job, "OPENAI_API_KEY not configured; skipping semantic enrichment")

        if STORE_INGESTION_ARTIFACTS and job.output_dir:
            self._set_status(job, IngestionStatus.exporting, "Exporting Neo4j import artifacts")
            exporter.export(nodes, edges)

        # Acquire per-logical-doc lock only around the Neo4j revision install.
        # Workers processing different documents are never blocked.
        if AUTO_LOAD_TO_NEO4J:
            self._set_status(job, IngestionStatus.exporting, "Loading graph into Neo4j")
            with self._doc_lock(plan.logical_id):
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
                except Exception as exc:
                    job.neo4j_load_status = "failed"
                    job.neo4j_load_message = str(exc)
                    self._log(job, f"Neo4j load failed: {exc}")
        else:
            job.neo4j_load_status = "skipped"
            job.neo4j_load_message = "AUTO_LOAD_TO_NEO4J disabled"
            self._log(job, "Neo4j load skipped")

    def _process_cypher(self, job: IngestionJob) -> None:
        """
        Execute a user-provided Cypher file against Neo4j.

        Intended for loading arbitrary schemas/datasets (e.g. Northwind).
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

    def _save_upload(self, upload: UploadFile, job_id: str) -> Path:
        target = self.temp_dir / f"{job_id}_{upload.filename}"
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
