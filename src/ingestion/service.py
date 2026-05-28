from __future__ import annotations

import csv
import shutil
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import UploadFile

from ..config.settings import AUTO_LOAD_TO_NEO4J, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER, OPENAI_API_KEY
from ..document.parser import DoclingParser
from ..exporter.exporter import Neo4jExporter
from ..models import DKGEdge, DKGNode
from ..semantic.axis2 import Axis2Builder
from .models import IngestionStatus, StructuredCSVMapping


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
    mapping: Optional[StructuredCSVMapping] = None
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
        self.output_base.mkdir(parents=True, exist_ok=True)

    def submit_unstructured(
        self,
        upload: UploadFile,
        job_name: Optional[str] = None,
    ) -> IngestionJob:
        job = self._create_job("unstructured", job_name=job_name)
        job.input_path = self._save_upload(upload, job.id)
        job.output_dir = self.output_base / job.id
        job.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs[job.id] = job
        self._log(job, f"Created unstructured ingestion job: {job.name or job.id}")
        return job

    def submit_structured(
        self,
        upload: UploadFile,
        mapping: StructuredCSVMapping,
        job_name: Optional[str] = None,
    ) -> IngestionJob:
        job = self._create_job("structured", job_name=job_name)
        job.mapping = mapping
        job.input_path = self._save_upload(upload, job.id)
        job.output_dir = self.output_base / job.id
        job.output_dir.mkdir(parents=True, exist_ok=True)
        self.jobs[job.id] = job
        self._log(job, f"Created structured ingestion job: {job.name or job.id}")
        return job

    def get_job(self, job_id: str) -> Optional[IngestionJob]:
        return self.jobs.get(job_id)

    def run_job(self, job_id: str) -> None:
        job = self.jobs.get(job_id)
        if job is None:
            return

        job.started_at = datetime.utcnow()
        self._set_status(job, IngestionStatus.validating, "Validating ingestion inputs")

        try:
            if job.type == "unstructured":
                self._process_unstructured(job)
            elif job.type == "structured":
                self._process_structured(job)
            else:
                raise ValueError(f"Unsupported ingestion type: {job.type}")

            job.finished_at = datetime.utcnow()
            self._set_status(job, IngestionStatus.completed, "Job completed successfully")
        except Exception as exc:
            job.finished_at = datetime.utcnow()
            job.error = str(exc)
            self._set_status(job, IngestionStatus.failed, f"Job failed: {job.error}")

    def _process_unstructured(self, job: IngestionJob) -> None:
        self._set_status(job, IngestionStatus.parsing, "Parsing document")
        if not job.input_path or not job.input_path.exists():
            raise FileNotFoundError("Uploaded file was not saved correctly.")

        parser = DoclingParser()
        nodes, edges = parser.parse(str(job.input_path))
        self._log(job, f"Parsed {len(nodes)} nodes and {len(edges)} edges")

        # Always attempt Axis 2 enrichment if OpenAI key is available
        if OPENAI_API_KEY:
            self._set_status(job, IngestionStatus.semantic_enrichment, "Running semantic enrichment (Axis 2)")
            builder = Axis2Builder(api_key=OPENAI_API_KEY)
            nodes, semantic_edges = builder.build(nodes, run_llm_pass=True)
            edges += semantic_edges
            self._log(job, f"Added {len(semantic_edges)} semantic edges")
        else:
            self._log(job, "OPENAI_API_KEY not configured; skipping semantic enrichment")

        self._set_status(job, IngestionStatus.exporting, "Exporting Neo4j import artifacts")
        exporter = Neo4jExporter(output_dir=str(job.output_dir))
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

    def _process_structured(self, job: IngestionJob) -> None:
        self._set_status(job, IngestionStatus.parsing, "Loading structured CSV")
        if not job.input_path or not job.input_path.exists() or job.mapping is None:
            raise FileNotFoundError("Structured CSV or mapping is missing.")

        with job.input_path.open("r", encoding="utf-8", newline="") as file_handle:
            reader = csv.DictReader(file_handle)
            header = reader.fieldnames or []
            job.mapping.validate_column_names(header)
            nodes_by_id: Dict[str, DKGNode] = {}
            edges: List[DKGEdge] = []

            for row_index, row in enumerate(reader, start=1):
                row_id = row.get(job.mapping.id_field, "").strip()
                if not row_id:
                    self._log(job, f"Skipping row {row_index} because {job.mapping.id_field} is empty")
                    continue

                title = row.get(job.mapping.properties[0], row_id) if job.mapping.properties else row_id
                text_parts = []
                for prop in job.mapping.properties:
                    value = row.get(prop, "")
                    if value:
                        text_parts.append(f"{prop}: {value}")

                node = DKGNode(
                    id=row_id,
                    type=job.mapping.node_label,
                    title=title,
                    text="\n".join(text_parts),
                    order=row_index,
                )
                nodes_by_id[row_id] = node

                for relation in job.mapping.relations:
                    target_value = row.get(relation.target_id_field or relation.field, "").strip()
                    if not target_value:
                        continue

                    source_id = row_id if relation.direction == "out" else target_value
                    target_id = target_value if relation.direction == "out" else row_id
                    edges.append(
                        DKGEdge(
                            source_id=source_id,
                            target_id=target_id,
                            rel_type=relation.rel_type,
                            axis=1,
                            properties={"source_field": relation.field},
                        )
                    )

                    if target_id not in nodes_by_id:
                        placeholder = DKGNode(
                            id=target_id,
                            type=relation.target_label,
                            title=target_id,
                            text="",
                            order=0,
                        )
                        nodes_by_id[target_id] = placeholder

        nodes = list(nodes_by_id.values())
        self._log(job, f"Created {len(nodes)} nodes and {len(edges)} relationships from CSV")

        self._set_status(job, IngestionStatus.exporting, "Exporting structured nodes and relationships")
        exporter = Neo4jExporter(output_dir=str(job.output_dir))
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
