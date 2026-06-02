from __future__ import annotations

import csv
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import UploadFile
from neo4j import GraphDatabase
from neo4j.exceptions import ClientError

from ..config.settings import (
    AUTO_LOAD_TO_NEO4J,
    CLEANUP_BOOK_ASSETS_ON_INGEST,
    CLEANUP_TMP_INGEST,
    CYPHER_INGEST_SKIP_GENAI,
    ENABLE_PAGE_VISION,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    OPENAI_API_KEY,
    STORE_INGESTION_ARTIFACTS,
)
from ..assets.cleanup import cleanup_document_assets
from ..document.parser import LightPdfParser
from ..models import NodeType
from ..exporter.exporter import Neo4jExporter
from ..models import DKGEdge, DKGNode
from .models import IngestionStatus

from ..auth.rbac_setup import GraphRBAC

@dataclass
class IngestionJob:
    id: str
    type: str
    name: Optional[str] = None
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


class IngestionManager:
    def __init__(self):
        self.jobs: Dict[str, IngestionJob] = {}
        self.temp_dir = Path("tmp_ingest")
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.output_base = Path("output/ingestion")
        if STORE_INGESTION_ARTIFACTS:
            self.output_base.mkdir(parents=True, exist_ok=True)

    def submit_unstructured(
        self,
        upload: UploadFile,
        job_name: Optional[str] = None,
    ) -> IngestionJob:
        job = self._create_job("unstructured", job_name=job_name)
        job.input_path = self._save_upload(upload, job.id)
        if STORE_INGESTION_ARTIFACTS:
            job.output_dir = self.output_base / job.id
            job.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs[job.id] = job
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
        self.jobs[job.id] = job
        self._log(job, f"Created cypher ingestion job: {job.name or job.id}")
        return job

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        return self.jobs.get(job_id)

    def has_active_job(self) -> bool:
        """True while a job is queued or in progress (not completed/failed)."""
        terminal = {IngestionStatus.completed, IngestionStatus.failed}
        return any(j.status not in terminal for j in self.jobs.values())

    def run_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
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
            # Cleanup failure shouldn't fail the job.
            self._log(job, f"Temp cleanup failed for {job.input_path}: {exc}")

    def _process_unstructured(self, job: IngestionJob) -> None:
        self._set_status(job, IngestionStatus.parsing, "Parsing document")
        if not job.input_path or not job.input_path.exists():
            raise FileNotFoundError("Uploaded file was not saved correctly.")

        if job.input_path.suffix.lower() != ".pdf":
            raise ValueError("Only PDF ingestion is supported by the lightweight parser.")

        parser = LightPdfParser()
        nodes, edges = parser.parse(str(job.input_path))
        self._log(job, f"Parsed {len(nodes)} nodes and {len(edges)} edges")

        document_id = next(
            (n.id for n in nodes if n.type in (NodeType.DOCUMENT, NodeType.DOCUMENT.value, NodeType.BOOK, NodeType.BOOK.value)),
            f"doc_{job.id}",
        )
        if CLEANUP_BOOK_ASSETS_ON_INGEST:
            try:
                removed = cleanup_document_assets(document_id)
                if removed:
                    self._log(
                        job,
                        f"Removed {removed} stale asset file(s) for {document_id}",
                    )
            except Exception as exc:
                self._log(job, f"Asset cleanup skipped for {document_id}: {exc}")

        if job.input_path.suffix.lower() == ".pdf":
            from ..assets.page_images import save_document_page_images
            from ..assets.region_images import save_region_images

            try:
                region_count = save_region_images(
                    job.input_path, document_id, nodes
                )
                self._log(job, f"Stored {region_count} region crop(s) (TABLE/FIGURE)")
            except Exception as exc:
                self._log(job, f"Region image storage skipped: {exc}")
            try:
                img_count = save_document_page_images(
                    job.input_path, document_id, nodes
                )
                self._log(job, f"Stored {img_count} full-page image(s) (JPEG fallback)")
            except Exception as exc:
                self._log(job, f"Page image storage skipped: {exc}")

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

        # Always attempt Axis 2 enrichment if OpenAI key is available
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

        exporter = Neo4jExporter(output_dir=str(job.output_dir) if job.output_dir else Path("."))
        if STORE_INGESTION_ARTIFACTS and job.output_dir:
            self._set_status(job, IngestionStatus.exporting, "Exporting Neo4j import artifacts")
            exporter.export(nodes, edges)

        if AUTO_LOAD_TO_NEO4J:
            self._set_status(job, IngestionStatus.exporting, "Loading graph into Neo4j")
            try:
                exporter.load_to_neo4j(nodes, edges, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
                job.neo4j_load_status = "success"
                job.neo4j_load_message = "Graph loaded into Neo4j successfully"
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
            # Prefer explicit job params over any :param directives in the file.
            params = {**params, **job.cypher_params}
        if not statements:
            raise ValueError("Cypher file contained no statements.")

        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        try:
            with driver.session() as session:
                for idx, stmt in enumerate(statements, start=1):
                    preview = " ".join(stmt.split())[:120]
                    self._log(job, f"Running statement {idx}/{len(statements)}: {preview}...")
                    try:
                        session.run(stmt, **params).consume()
                    except ClientError as exc:
                        # Make common schema operations idempotent when scripts are re-run.
                        # Examples: CREATE INDEX / CONSTRAINT where equivalent already exists.
                        code = getattr(exc, "code", "") or ""
                        if code in {
                            "Neo.ClientError.Schema.EquivalentSchemaRuleAlreadyExists",
                            "Neo.ClientError.Schema.IndexAlreadyExists",
                            "Neo.ClientError.Schema.ConstraintAlreadyExists",
                        }:
                            self._log(job, f"Skipping non-fatal schema error ({code}): {exc.message}")
                            continue
                        raise
        finally:
            driver.close()

        job.neo4j_load_status = "success"
        job.neo4j_load_message = f"Executed {len(statements)} Cypher statements successfully"
        self._log(job, job.neo4j_load_message)

    def _parse_cypher_script(self, raw_text: str) -> tuple[list[str], dict]:
        """
        Parse a Cypher script that may include Neo4j Browser directives like:
          :param key => 'value';

        Browser directives are not valid Cypher over the driver. We:
        - Extract :param directives into a parameters dict
        - Strip other ':' directives (e.g. :use)
        - Split remaining Cypher by semicolon into statements
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

            # Neo4j Browser directives are prefixed with ':'
            if stripped.startswith(":"):
                # Support ':param name => value'
                if stripped.lower().startswith(":param"):
                    # Example: :param openAIKey => 'sk-...';
                    rest = stripped[len(":param"):].strip()
                    if "=>" in rest:
                        key, value = rest.split("=>", 1)
                        key = key.strip().lstrip("$")
                        if key:
                            params[key] = _parse_param_value(value)
                # Always skip directive lines (they are not Cypher)
                continue

            cypher_lines.append(line)

        cypher_text = "\n".join(cypher_lines)
        statements = [s.strip() for s in cypher_text.split(";") if s.strip()]
        if CYPHER_INGEST_SKIP_GENAI:
            filtered: list[str] = []
            for stmt in statements:
                s = stmt.lower()
                # Skip GenAI embedding calls and the most common follow-up write step.
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

    def _log(self, job: IngestionJob, message: str) -> None:
        timestamp = datetime.utcnow().isoformat() + "Z"
        entry = f"{timestamp} - {message}"
        job.logs.append(entry)
        if len(job.logs) > 100:
            job.logs = job.logs[-100:]
