"""
exporter.py — DKGNode/DKGEdge → Neo4j import artifacts.

Produces:
  output/
    setup.cypher          ← constraints + indexes (run once)
    nodes/
      books.csv
      chapters.csv
      sections.csv
      pages.csv
      concepts.csv
    edges/
      axis1_structural.csv
      axis2_semantic.csv
    import.cypher         ← LOAD CSV statements for all files
    full_import.cypher    ← single-file alternative (no CSV needed)
"""
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ...shared.neo4j.driver import get_neo4j_driver

from ...shared.config.settings import NEO4J_WRITE_BATCH
from ..document.versioning import DocumentRevisionPlan
from ..graph.constants import DOC_REVISION_LABEL, DOCUMENT_LOGICAL_LABEL
from ..models import DKGNode, DKGEdge, EdgeConfidenceTier, NodeType, RelType
from ...shared.storage.blob.base import BlobStore
from ...shared.storage.blob.factory import get_blob_store
from ...shared.storage.vector.base import VectorStore
from ...shared.storage.vector.factory import get_vector_store


OUTPUT_DIR = Path("output")


class Neo4jExporter:

    def __init__(
        self,
        output_dir: str | Path = OUTPUT_DIR,
        blob_store: BlobStore | None = None,
        vector_store: VectorStore | None = None,
    ):
        self.out = Path(output_dir)
        (self.out / "nodes").mkdir(parents=True, exist_ok=True)
        (self.out / "edges").mkdir(parents=True, exist_ok=True)
        # Dual-write: Neo4j properties stay authoritative (existing reads on
        # already-ingested data keep working unmodified); text/embeddings are
        # additionally written here so the blob/vector-store interfaces are
        # exercised on every new ingest, with no backfill of prior revisions.
        self.blob_store = blob_store or get_blob_store()
        self.vector_store = vector_store or get_vector_store()

    def _label_to_str(self, label: str | NodeType) -> str:
        if isinstance(label, NodeType):
            return label.value
        return str(label)

    def _rel_type_to_str(self, rel_type: str | RelType) -> str:
        if isinstance(rel_type, RelType):
            return rel_type.value
        return str(rel_type)

    def _safe_name(self, value: str) -> str:
        return "".join(ch if ch.isalnum() else "_" for ch in value).lower()

    def export(self, nodes: list[DKGNode], edges: list[DKGEdge]) -> None:
        self._write_setup_cypher()
        self._write_node_csvs(nodes)
        self._write_edge_csvs(edges)
        self._write_load_csv_cypher(nodes, edges)
        self._write_full_cypher(nodes, edges)
        print(f"\n✅ Export complete → {self.out.resolve()}")
        print(f"   Nodes : {len(nodes)}")
        print(f"   Edges : {len(edges)}")

    def load_to_neo4j(
        self,
        nodes: list[DKGNode],
        edges: list[DKGEdge],
        uri: str,
        user: str,
        password: str,
        *,
        revision_plan: DocumentRevisionPlan,
        skip_if_duplicate_hash: bool = True,
    ) -> dict:
        """
        Load graph into Neo4j as a versioned revision (expire prior ACTIVE
        revision, purge its content subgraph, load snapshot). Returns
        metadata dict: skipped_duplicate, revision_id, logical_doc_id,
        version_number.

        `revision_plan` is required -- the only caller (IngestionManager._
        process_unstructured) always builds one first, and the earlier
        no-revision merge path (_merge_node/_merge_edge) had zero other
        callers; kept as a required kwarg rather than removed outright so
        the call site stays self-documenting about what's being loaded.
        """
        driver = get_neo4j_driver(uri, user, password)
        meta: dict = {
            "skipped_duplicate": False,
            "revision_id": revision_plan.revision_id,
            "logical_doc_id": revision_plan.logical_id,
            "version_number": revision_plan.version_number,
        }
        with driver.session() as session:
            self._ensure_constraints(session, nodes)
            self._ensure_versioning_constraints(session)
            self._ensure_indexes(session)

            if skip_if_duplicate_hash and self.active_revision_has_hash(
                session, revision_plan.logical_id, revision_plan.content_hash
            ):
                meta["skipped_duplicate"] = True
                return meta
            self._install_revision_snapshot(session, revision_plan, nodes, edges)
        print("✅ Loaded graph into Neo4j")
        return meta

    def _ensure_versioning_constraints(self, session) -> None:
        session.run(
            f"CREATE CONSTRAINT document_logical_id IF NOT EXISTS "
            f"FOR (n:{DOCUMENT_LOGICAL_LABEL}) REQUIRE n.logical_id IS UNIQUE"
        )
        session.run(
            f"CREATE CONSTRAINT doc_revision_id IF NOT EXISTS "
            f"FOR (n:{DOC_REVISION_LABEL}) REQUIRE n.id IS UNIQUE"
        )
        # Backfill before the indexes: lifecycle_active() now emits a
        # direct equality so the composite indexes below are seekable, and a
        # node written before lifecycle_status existed would otherwise become
        # invisible rather than merely slow. This is its own statement: when
        # it shared a session.run() with the CREATE INDEX below, the index
        # string landed in the `parameters` slot and the index was silently
        # never created -- which is exactly the index the scoped-read path
        # depends on.
        session.run(
            "MATCH (n) WHERE n.logical_doc_id IS NOT NULL "
            "AND n.lifecycle_status IS NULL "
            "SET n.lifecycle_status = 'ACTIVE'"
        )
        session.run(
            "CREATE INDEX doc_revision_logical IF NOT EXISTS "
            f"FOR (n:{DOC_REVISION_LABEL}) ON (n.logical_doc_id)"
        )
        # Every text-bearing label needs this composite, not just Section and
        # Page. Chapter and Region had none, so a document-scoped read that
        # named them scanned the whole database -- which also holds the
        # structured business graph -- at 1,234,143 db hits against 27 for
        # the same lookup once indexed. Document is here for the same reason:
        # resolving a document by logical_doc_id is on every scoped path.
        for _label in ("Chapter", "Region", "Document"):
            session.run(
                f"CREATE INDEX {_label.lower()}_logical_lifecycle IF NOT EXISTS "
                f"FOR (n:{_label}) ON (n.logical_doc_id, n.lifecycle_status)"
            )
        session.run(
            "CREATE INDEX content_logical_lifecycle IF NOT EXISTS "
            "FOR (n:Section) ON (n.logical_doc_id, n.lifecycle_status)"
        )
        session.run(
            "CREATE INDEX page_logical_lifecycle IF NOT EXISTS "
            "FOR (n:Page) ON (n.logical_doc_id, n.lifecycle_status)"
        )
        session.run(
            "CREATE INDEX section_revision IF NOT EXISTS "
            "FOR (n:Section) ON (n.revision_id)"
        )

    def active_revision_has_hash(
        self, session, logical_id: str, content_hash: str
    ) -> bool:
        row = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL} {{logical_id: $logical_id}})
                  -[:ACTIVE_REVISION]->(rev:{DOC_REVISION_LABEL})
            WHERE rev.content_hash = $content_hash AND rev.status = 'ACTIVE'
            RETURN rev.id AS id LIMIT 1
            """,
            logical_id=logical_id,
            content_hash=content_hash,
        ).single()
        return bool(row and row.get("id"))

    def logical_id_holding_hash(
        self, session, content_hash: str, tenant_id: str
    ) -> Optional[str]:
        """Which logical document already owns this exact content, if any.

        Supersede only ever fires within one logical id, so the same file
        ingested twice under different doc_keys becomes two documents rather
        than two revisions of one. That is how three copies of one PDF ended
        up in the graph, all titled from the same filename and so
        indistinguishable in a document picker. Binding content to the
        logical id it first got makes the second ingest a revision, and the
        existing supersede path then expires and deletes the older one.

        Oldest wins, so the answer is stable no matter how many copies exist.
        """
        row = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL})-[:ACTIVE_REVISION]->(rev:{DOC_REVISION_LABEL})
            WHERE rev.content_hash = $content_hash
              AND rev.status = 'ACTIVE'
              AND coalesce(rev.tenant_id, 'default') = $tenant_id
            RETURN dl.logical_id AS logical_id
            ORDER BY rev.ingested_at ASC
            LIMIT 1
            """,
            content_hash=content_hash,
            tenant_id=tenant_id,
        ).single()
        return row.get("logical_id") if row else None

    def next_version_number(self, session, logical_id: str) -> int:
        row = session.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL} {{logical_id: $logical_id}})
                  -[:HAS_REVISION]->(rev:{DOC_REVISION_LABEL})
            RETURN max(rev.version_number) AS mx
            """,
            logical_id=logical_id,
        ).single()
        mx = row.get("mx") if row else None
        return int(mx or 0) + 1

    def _install_revision_snapshot(
        self,
        session,
        plan: DocumentRevisionPlan,
        nodes: list[DKGNode],
        edges: list[DKGEdge],
    ) -> None:
        superseded = session.execute_write(self._install_revision_tx, plan, nodes, edges)
        if superseded:
            # After the transaction commits, never inside it: an object-store
            # round trip would otherwise hold a Neo4j write lock open for the
            # length of a network call. Best-effort -- a leaked blob is
            # cheaper than failing an ingest that has already succeeded.
            from ..document.purge import purge_revision

            purge_revision(
                tenant_id=plan.tenant_id,
                logical_id=plan.logical_id,
                revision_id=superseded,
                blob_store=self.blob_store,
                vector_store=self.vector_store,
            )

    def _install_revision_tx(
        self, tx, plan: DocumentRevisionPlan, nodes, edges
    ) -> Optional[str]:
        """Returns the id of the revision this one superseded, if any, so the
        caller can purge its blobs and vectors once the write has committed."""
        tx.run(
            f"""
            MERGE (dl:{DOCUMENT_LOGICAL_LABEL} {{logical_id: $logical_id}})
            ON CREATE SET dl.title = $title, dl.created_at = timestamp(), dl.tenant_id = $tenant_id
            ON MATCH SET dl.title = coalesce(dl.title, $title),
                         dl.updated_at = timestamp(),
                         dl.tenant_id = coalesce(dl.tenant_id, $tenant_id)
            """,
            logical_id=plan.logical_id,
            title=plan.title,
            tenant_id=plan.tenant_id,
        )
        row = tx.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL} {{logical_id: $logical_id}})
            OPTIONAL MATCH (dl)-[:ACTIVE_REVISION]->(prev:{DOC_REVISION_LABEL})
            RETURN prev.id AS prev_id, prev.version_number AS prev_ver
            """,
            logical_id=plan.logical_id,
        ).single()
        prev_id = row.get("prev_id") if row else None
        if prev_id:
            tx.run(
                f"""
                MATCH (dl:{DOCUMENT_LOGICAL_LABEL} {{logical_id: $logical_id}})
                      -[ar:ACTIVE_REVISION]->(prev:{DOC_REVISION_LABEL} {{id: $prev_id}})
                DELETE ar
                SET prev.status = 'EXPIRED',
                    prev.expired_at = timestamp(),
                    prev.lifecycle_status = 'EXPIRED'
                WITH prev
                MATCH (n)
                WHERE n.revision_id = $prev_id
                  AND NOT n:{DOC_REVISION_LABEL}
                  AND NOT n:{DOCUMENT_LOGICAL_LABEL}
                DETACH DELETE n
                """,
                logical_id=plan.logical_id,
                prev_id=prev_id,
            )

        tx.run(
            f"""
            MATCH (dl:{DOCUMENT_LOGICAL_LABEL} {{logical_id: $logical_id}})
            CREATE (rev:{DOC_REVISION_LABEL} {{
                id: $revision_id,
                logical_id: $logical_id,
                logical_doc_id: $logical_id,
                revision_id: $revision_id,
                version_number: $version_number,
                status: 'ACTIVE',
                lifecycle_status: 'ACTIVE',
                content_hash: $content_hash,
                title: $title,
                text: $source_filename,
                source_filename: $source_filename,
                tenant_id: $tenant_id,
                ingested_at: timestamp(),
                uploaded_at: timestamp()
            }})
            MERGE (dl)-[:HAS_REVISION]->(rev)
            CREATE (dl)-[:ACTIVE_REVISION]->(rev)
            """,
            logical_id=plan.logical_id,
            revision_id=plan.revision_id,
            version_number=plan.version_number,
            content_hash=plan.content_hash,
            title=plan.title,
            source_filename=plan.source_filename,
            tenant_id=plan.tenant_id,
        )

        # ── Batched node writes grouped by label (UNWIND) ─────────────────
        skip_labels = {DOCUMENT_LOGICAL_LABEL, DOC_REVISION_LABEL, "Book"}
        nodes_by_label: Dict[str, List[DKGNode]] = defaultdict(list)
        for node in nodes:
            label = node.type.value if isinstance(node.type, NodeType) else str(node.type)
            if label not in skip_labels:
                nodes_by_label[label].append(node)

        for label, label_nodes in nodes_by_label.items():
            for chunk_start in range(0, len(label_nodes), NEO4J_WRITE_BATCH):
                chunk = label_nodes[chunk_start : chunk_start + NEO4J_WRITE_BATCH]
                self._dual_write_chunk(chunk, plan)
                rows = [Neo4jExporter._node_to_param_dict(n) for n in chunk]
                tx.run(
                    f"UNWIND $rows AS row "
                    f"CREATE (n:{label}) "
                    "SET n = row",
                    rows=rows,
                )

        # ── Batched edge writes grouped by rel_type (UNWIND) ──────────────
        skip_rels = {
            RelType.HAS_REVISION.value,
            RelType.ACTIVE_REVISION.value,
            RelType.ROOT.value,
        }
        # Endpoint labels, so each MATCH below can use the per-label `id`
        # index (Page_id, Section_id, ...). An unlabelled `MATCH (a {id: ...})`
        # has no index to use and degrades to AllNodesScan: every lookup walks
        # the entire database, and the database also holds the structured
        # business graph. On a 547k-node instance a 16-page document's 477
        # semantic edges came to roughly half a billion node scans, and the
        # single write sat at 100% Neo4j CPU for minutes. Empty-database
        # ingestion never showed it -- there was nothing to scan.
        label_by_id: Dict[str, str] = {
            node.id: (node.type.value if isinstance(node.type, NodeType) else str(node.type))
            for node in nodes
        }

        # Grouped by endpoint labels as well as rel type, since the labels are
        # baked into the query text rather than passed as parameters -- a label
        # cannot be parameterised in Cypher.
        edges_by_shape: Dict[tuple, List[DKGEdge]] = defaultdict(list)
        for edge in edges:
            rel = edge.rel_type.value if isinstance(edge.rel_type, RelType) else str(edge.rel_type)
            if rel not in skip_rels:
                edges_by_shape[
                    (rel, label_by_id.get(edge.source_id), label_by_id.get(edge.target_id))
                ].append(edge)

        for (rel_type, source_label, target_label), rel_edges in edges_by_shape.items():
            # An edge can point at a node this revision did not write (nothing
            # in the current plan does, but the write must not depend on that).
            # Unknown endpoints keep the unlabelled form: slow, but correct.
            a = f"a:{source_label}" if source_label else "a"
            b = f"b:{target_label}" if target_label else "b"
            for chunk_start in range(0, len(rel_edges), NEO4J_WRITE_BATCH):
                chunk = rel_edges[chunk_start : chunk_start + NEO4J_WRITE_BATCH]
                rows = [Neo4jExporter._edge_to_param_dict(e) for e in chunk]
                tx.run(
                    f"UNWIND $rows AS row "
                    f"MATCH ({a} {{id: row.source_id}}) "
                    f"MATCH ({b} {{id: row.target_id}}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    "SET r.weight = row.weight, r.axis = row.axis, r.properties = row.properties, "
                    "r.confidence = row.confidence, r.confidence_tier = row.confidence_tier, "
                    "r.tenant_id = row.tenant_id",
                    rows=rows,
                )

        tx.run(
            f"""
            MATCH (rev:{DOC_REVISION_LABEL} {{id: $revision_id}})
            MATCH (root {{id: $root_id}})
            MERGE (rev)-[:ROOT]->(root)
            """,
            revision_id=plan.revision_id,
            root_id=plan.content_root_id,
        )
        return prev_id

    def _dual_write_chunk(self, chunk: list[DKGNode], plan: DocumentRevisionPlan) -> None:
        """
        Write text/visual_content/embedding to the blob/vector stores in
        addition to the Neo4j properties set by the caller. Batched per
        Neo4j UNWIND chunk (not per-node) to match write throughput.
        """
        vector_items: list[tuple[str, list[float], dict]] = []
        for node in chunk:
            if node.text:
                node.blob_key_text = f"{plan.tenant_id}/{plan.logical_id}/{plan.revision_id}/{node.id}/text"
                self.blob_store.put(node.blob_key_text, node.text)
            if node.visual_content:
                node.blob_key_visual = (
                    f"{plan.tenant_id}/{plan.logical_id}/{plan.revision_id}/{node.id}/visual_content"
                )
                self.blob_store.put(node.blob_key_visual, node.visual_content)
            if node.embedding:
                node.vector_id = self.vector_store.point_id_for(node.id)
                vector_items.append(
                    (
                        node.id,
                        node.embedding,
                        {
                            "logical_doc_id": plan.logical_id,
                            "revision_id": plan.revision_id,
                            "tenant_id": plan.tenant_id,
                        },
                    )
                )
        if vector_items:
            self.vector_store.upsert_batch(vector_items)

    @staticmethod
    def _node_to_param_dict(node: DKGNode) -> dict:
        """Serialise a DKGNode to a plain dict for use in UNWIND parameters."""
        return {
            "id": node.id,
            "title": node.title,
            # `text` (full body) is deliberately NOT written to Neo4j --
            # phase-3 write-side strip (docs/DESIGN_unstructured_graph_v2.md).
            # It's still dual-written to the blob store via blob_key_text
            # (Neo4jExporter._dual_write_chunk, above) for hydration; Neo4j
            # keeps only search_text (chunk-bounded, for lexical matching)
            # and the pointer.
            "search_text": node.search_text,
            "vector_id": node.vector_id,
            "order": node.order,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "depth": node.depth,
            "entities": node.entities,
            # JSON string, not a native map: Neo4j properties cannot hold a
            # nested map. Persisted (rather than left in memory as it was
            # originally) so Axis-2 edges can be REBUILT from the stored
            # graph without re-running NER -- entity type drives the DATE
            # exclusion, the same-type enumeration cap and homonym
            # separation, so rebuilding without it would silently produce
            # worse edges than the ingestion that created them, not merely
            # equivalent ones. NER is the expensive, quota-limited step, so
            # this is what makes iterating on edge quality cheap.
            "entity_types": json.dumps(node.entity_types or {}),
            # A table continued across pages is one logical unit; without
            # these a retrieved part cannot find its siblings.
            "section_path": node.section_path,
            "unit_id": node.unit_id,
            "unit_part": node.unit_part,
            "cluster_id": node.cluster_id,
            "summary": node.summary,
            "visual_content": node.visual_content,
            "pdf_page": node.pdf_page,
            "document_page": node.document_page,
            "page_tags": node.page_tags or [],
            "region_kind": node.region_kind,
            "region_tags": node.region_tags or [],
            "logical_doc_id": node.logical_doc_id,
            "revision_id": node.revision_id,
            "lifecycle_status": node.lifecycle_status,
            "content_hash": node.content_hash,
            "version_number": node.version_number,
            "ingested_at": node.ingested_at,
            "source_filename": node.source_filename,
            "blob_key_text": node.blob_key_text,
            "blob_key_visual": node.blob_key_visual,
            "tenant_id": node.tenant_id,
        }

    @staticmethod
    def _edge_to_param_dict(edge: DKGEdge) -> dict:
        """Serialise a DKGEdge to a plain dict for use in UNWIND parameters."""
        tier = edge.confidence_tier
        return {
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "weight": edge.weight,
            "axis": edge.axis,
            "properties": json.dumps(edge.properties),
            "confidence": edge.confidence,
            "confidence_tier": tier.value if isinstance(tier, EdgeConfidenceTier) else str(tier),
            "tenant_id": edge.tenant_id,
        }

    def _ensure_indexes(self, session) -> None:
        """Idempotently create full-text indexes on every ingestion."""
        statements = [
            "CREATE FULLTEXT INDEX node_text_index IF NOT EXISTS "
            "FOR (n:Book|Chapter|Section|Page|Region|Concept) "
            "ON EACH [n.title, n.search_text, n.visual_content]",
            "CREATE FULLTEXT INDEX page_visual_index IF NOT EXISTS "
            "FOR (n:Page) ON EACH [n.visual_content, n.title, n.search_text, n.document_page]",
            "CREATE FULLTEXT INDEX region_tag_index IF NOT EXISTS "
            "FOR (n:Region) ON EACH [n.title, n.search_text, n.region_tags, n.region_kind]",
            "CREATE FULLTEXT INDEX page_number_index IF NOT EXISTS "
            "FOR (n:Page) ON EACH [n.document_page, n.page_tags, n.title]",
            "CREATE INDEX section_order IF NOT EXISTS FOR (n:Section) ON (n.order)",
            "CREATE INDEX page_order    IF NOT EXISTS FOR (n:Page)    ON (n.order)",
            "CREATE INDEX page_start    IF NOT EXISTS FOR (n:Page)    ON (n.page_start)",
            "CREATE INDEX page_pdf_page IF NOT EXISTS FOR (n:Page) ON (n.pdf_page)",
            "CREATE INDEX section_logical_rev IF NOT EXISTS "
            "FOR (n:Section) ON (n.logical_doc_id, n.revision_id)",
            "CREATE INDEX section_tenant_lifecycle IF NOT EXISTS "
            "FOR (n:Section) ON (n.tenant_id, n.lifecycle_status)",
            "CREATE INDEX page_tenant_lifecycle IF NOT EXISTS "
            "FOR (n:Page) ON (n.tenant_id, n.lifecycle_status)",
        ]
        # No Neo4j-native vector index (section_embedding) here anymore --
        # embeddings never reach Neo4j at all as of the phase-3 write-side
        # strip (docs/DESIGN_unstructured_graph_v2.md), regardless of
        # VECTOR_STORE_BACKEND. A non-Qdrant dev fallback is a VectorStore
        # backend concern (see src/storage/vector/memory_store.py), not a
        # reason to keep a Neo4j index that would only ever index nulls.
        for stmt in statements:
            try:
                session.run(stmt).consume()
            except Exception as e:
                code = getattr(e, 'code', '') or ''
                if any(x in code for x in [
                    'EquivalentSchemaRuleAlreadyExists',
                    'IndexAlreadyExists',
                    'ConstraintAlreadyExists',
                ]):
                    continue
                print(f"⚠️  Index skipped: {e}")

    def _ensure_constraints(self, session, nodes: list[DKGNode]) -> None:
        labels = {self._label_to_str(n.type) for n in nodes}
        for label in sorted(labels):
            safe_label = label.replace(' ', '_')
            session.run(
                f"CREATE CONSTRAINT {safe_label}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE"
            )

    # ─────────────────────────────────────────
    # 1. SETUP CYPHER  (constraints + indexes)
    # ─────────────────────────────────────────
    def _write_setup_cypher(self) -> None:
        cypher = """\
// ─────────────────────────────────────────────────────────────
// Document Knowledge Graph — Neo4j Setup
// Run this ONCE before importing data
// ─────────────────────────────────────────────────────────────

// Unique constraints (also create indexes automatically)
CREATE CONSTRAINT document_id IF NOT EXISTS FOR (n:Document) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT book_id    IF NOT EXISTS FOR (n:Book)    REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT chapter_id IF NOT EXISTS FOR (n:Chapter) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT section_id IF NOT EXISTS FOR (n:Section) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT page_id    IF NOT EXISTS FOR (n:Page)    REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT concept_id IF NOT EXISTS FOR (n:Concept) REQUIRE n.id IS UNIQUE;

// Full-text search index (for semantic query agent)
CREATE FULLTEXT INDEX node_text_index IF NOT EXISTS
FOR (n:Document|Book|Chapter|Section|Page|Concept)
ON EACH [n.title, n.text];

// Ordering indexes
CREATE INDEX chapter_order IF NOT EXISTS FOR (n:Chapter) ON (n.order);
CREATE INDEX section_order IF NOT EXISTS FOR (n:Section) ON (n.order);
CREATE INDEX page_order    IF NOT EXISTS FOR (n:Page)    ON (n.order);
CREATE INDEX page_start    IF NOT EXISTS FOR (n:Page)    ON (n.page_start);

// Vector index for semantic search (requires embeddings on nodes)
CREATE VECTOR INDEX section_embedding IF NOT EXISTS
FOR (n:Section) ON (n.embedding)
OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}};
"""
        (self.out / "setup.cypher").write_text(cypher)

    # ─────────────────────────────────────────
    # 2. NODE CSVs
    # ─────────────────────────────────────────
    def _write_node_csvs(self, nodes: list[DKGNode]) -> None:
        buckets: dict[str, list[DKGNode]] = {}
        for n in nodes:
            label = self._label_to_str(n.type)
            buckets.setdefault(label, []).append(n)

        fieldnames = ["id", "type", "title", "text", "order",
                      "page_start", "page_end", "depth",
                      "entities", "entity_types", "unit_id", "unit_part", "section_path", "cluster_id"]

        for label, type_nodes in buckets.items():
            fname = f"{self._safe_name(label)}s.csv"
            with open(self.out / "nodes" / fname, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for n in type_nodes:
                    writer.writerow({
                        "id":         n.id,
                        "type":       self._label_to_str(n.type),
                        "title":      n.title,
                        "text":       n.text.replace("\n", "\\n"),
                        "order":      n.order,
                        "page_start": n.page_start,
                        "page_end":   n.page_end,
                        "depth":      n.depth,
                        "entities":   json.dumps(n.entities),
                        "entity_types": json.dumps(n.entity_types or {}),
                        "unit_id":    n.unit_id or "",
                        "unit_part":  n.unit_part,
                        "section_path": n.section_path or "",
                        "cluster_id": n.cluster_id if n.cluster_id is not None else "",
                    })

    # ─────────────────────────────────────────
    # 3. EDGE CSVs  (split by axis)
    # ─────────────────────────────────────────
    def _write_edge_csvs(self, edges: list[DKGEdge]) -> None:
        axis1 = [e for e in edges if e.axis == 1]
        axis2 = [e for e in edges if e.axis == 2]

        for fname, edge_list in [
            ("axis1_structural.csv", axis1),
            ("axis2_semantic.csv",   axis2),
        ]:
            with open(self.out / "edges" / fname, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=["source_id", "target_id", "rel_type",
                                   "weight", "axis", "properties",
                                   "confidence", "confidence_tier"]
                )
                writer.writeheader()
                for e in edge_list:
                    tier = e.confidence_tier
                    writer.writerow({
                        "source_id":  e.source_id,
                        "target_id":  e.target_id,
                        "rel_type":   e.rel_type.value,
                        "weight":     e.weight,
                        "axis":       e.axis,
                        "properties": json.dumps(e.properties),
                        "confidence": e.confidence,
                        "confidence_tier": tier.value if isinstance(tier, EdgeConfidenceTier) else str(tier),
                    })

    # ─────────────────────────────────────────
    # 4. LOAD CSV CYPHER
    # ─────────────────────────────────────────
    def _write_load_csv_cypher(
        self, nodes: list[DKGNode], edges: list[DKGEdge]
    ) -> None:
        node_types = {self._label_to_str(n.type) for n in nodes}
        lines = ["// ── LOAD CSV Import ──────────────────────────────────\n"]

        # Node imports per type
        for label in sorted(node_types):
            fname = f"{self._safe_name(label)}s.csv"
            lines.append(f"// {label} nodes")
            lines.append(f"""\
LOAD CSV WITH HEADERS FROM 'file:///nodes/{fname}' AS row
MERGE (n:{label} {{id: row.id}})
SET   n.title      = row.title,
      n.text       = row.text,
      n.order      = toInteger(row.order),
      n.page_start = toInteger(row.page_start),
      n.page_end   = toInteger(row.page_end),
      n.depth      = toInteger(row.depth),
      n.entities   = row.entities,
      n.entity_types = row.entity_types,
      n.unit_id      = row.unit_id,
      n.unit_part    = toInteger(row.unit_part),
      n.section_path = row.section_path,
      n.cluster_id = CASE row.cluster_id WHEN '' THEN null ELSE toInteger(row.cluster_id) END;
""")

        # Axis 1 edges
        lines.append("// Axis 1 — Structural relationships")
        lines.append("""\
LOAD CSV WITH HEADERS FROM 'file:///edges/axis1_structural.csv' AS row
MATCH (a {id: row.source_id}), (b {id: row.target_id})
CALL apoc.merge.relationship(a, row.rel_type, {}, {
  weight: toFloat(row.weight),
  confidence: toFloat(row.confidence),
  confidence_tier: row.confidence_tier
}, b)
YIELD rel RETURN count(rel);
""")

        # Axis 2 edges
        lines.append("// Axis 2 — Semantic relationships")
        lines.append("""\
LOAD CSV WITH HEADERS FROM 'file:///edges/axis2_semantic.csv' AS row
MATCH (a {id: row.source_id}), (b {id: row.target_id})
CALL apoc.merge.relationship(a, row.rel_type, {}, {
  weight:     toFloat(row.weight),
  properties: row.properties,
  confidence: toFloat(row.confidence),
  confidence_tier: row.confidence_tier
}, b)
YIELD rel RETURN count(rel);
""")

        (self.out / "import.cypher").write_text("\n".join(lines))

    # ─────────────────────────────────────────
    # 5. FULL CYPHER (no CSV, single file)
    # ─────────────────────────────────────────
    def _write_full_cypher(
        self, nodes: list[DKGNode], edges: list[DKGEdge]
    ) -> None:
        """
        Single .cypher file with MERGE statements.
        Easier for small documents — just run in Neo4j Browser.
        """
        lines = [
            "// ─────────────────────────────────────────────",
            "// Document Knowledge Graph — Full Import",
            "// Paste into Neo4j Browser or run with cypher-shell",
            "// ─────────────────────────────────────────────\n",
            "// ── NODES ────────────────────────────────────",
        ]

        for n in nodes:
            entities_str  = json.dumps(n.entities).replace("'", "\\'")
            etypes_str    = json.dumps(n.entity_types or {}).replace("'", "\\'")
            text_escaped  = n.text.replace("'", "\\'").replace("\n", "\\n")
            title_escaped = n.title.replace("'", "\\'")
            cluster       = f", n.cluster_id={n.cluster_id}" if n.cluster_id is not None else ""
            label = self._label_to_str(n.type)
            lines.append(
                f"MERGE (n:{label} {{id: '{n.id}'}})"
                f" SET n.title='{title_escaped}', n.text='{text_escaped}',"
                f" n.order={n.order}, n.page_start={n.page_start},"
                f" n.page_end={n.page_end}, n.depth={n.depth},"
                f" n.entities='{entities_str}',"
                f" n.entity_types='{etypes_str}'{cluster};"
            )

        lines += ["\n// ── AXIS 1 — STRUCTURAL EDGES ───────────────"]
        axis1_edges = [e for e in edges if e.axis == 1]
        for e in axis1_edges:
            rel_type = self._rel_type_to_str(e.rel_type)
            tier = e.confidence_tier
            tier_str = tier.value if isinstance(tier, EdgeConfidenceTier) else str(tier)
            lines.append(
                f"MATCH (a {{id: '{e.source_id}'}}), (b {{id: '{e.target_id}'}})"
                f" MERGE (a)-[:{rel_type} {{weight: {e.weight},"
                f" confidence: {e.confidence}, confidence_tier: '{tier_str}'}}]->(b);"
            )

        lines += ["\n// ── AXIS 2 — SEMANTIC EDGES ─────────────────"]
        axis2_edges = [e for e in edges if e.axis == 2]
        for e in axis2_edges:
            rel_type = self._rel_type_to_str(e.rel_type)
            props_str = json.dumps(e.properties).replace("'", "\\'")
            tier = e.confidence_tier
            tier_str = tier.value if isinstance(tier, EdgeConfidenceTier) else str(tier)
            lines.append(
                f"MATCH (a {{id: '{e.source_id}'}}), (b {{id: '{e.target_id}'}})"
                f" MERGE (a)-[:{rel_type} {{weight: {e.weight},"
                f" props: '{props_str}', confidence: {e.confidence},"
                f" confidence_tier: '{tier_str}'}}]->(b);"
            )

        (self.out / "full_import.cypher").write_text("\n".join(lines))