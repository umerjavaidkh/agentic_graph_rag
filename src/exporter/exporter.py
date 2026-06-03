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
from typing import Dict, List

from ..graph.driver import get_neo4j_driver

from ..config.settings import NEO4J_WRITE_BATCH
from ..document.versioning import DocumentRevisionPlan
from ..graph.constants import DOC_REVISION_LABEL, DOCUMENT_LOGICAL_LABEL
from ..models import DKGNode, DKGEdge, NodeType, RelType


OUTPUT_DIR = Path("output")


class Neo4jExporter:

    def __init__(self, output_dir: str | Path = OUTPUT_DIR):
        self.out = Path(output_dir)
        (self.out / "nodes").mkdir(parents=True, exist_ok=True)
        (self.out / "edges").mkdir(parents=True, exist_ok=True)

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
        revision_plan: DocumentRevisionPlan | None = None,
        skip_if_duplicate_hash: bool = True,
    ) -> dict:
        """
        Load graph into Neo4j. When revision_plan is set, runs versioned ingest
        (expire prior ACTIVE revision, purge its content subgraph, load snapshot).
        Returns metadata dict: skipped_duplicate, revision_id, logical_doc_id, version_number.
        """
        driver = get_neo4j_driver(uri, user, password)
        meta: dict = {
            "skipped_duplicate": False,
            "revision_id": None,
            "logical_doc_id": None,
            "version_number": None,
        }
        with driver.session() as session:
            self._ensure_constraints(session, nodes)
            self._ensure_versioning_constraints(session)
            self._ensure_indexes(session)

            if revision_plan is not None:
                meta["logical_doc_id"] = revision_plan.logical_id
                meta["revision_id"] = revision_plan.revision_id
                meta["version_number"] = revision_plan.version_number
                if skip_if_duplicate_hash and self.active_revision_has_hash(
                    session, revision_plan.logical_id, revision_plan.content_hash
                ):
                    meta["skipped_duplicate"] = True
                    return meta
                self._install_revision_snapshot(
                    session, revision_plan, nodes, edges
                )
            else:
                for node in nodes:
                    self._merge_node(session, node)
                for edge in edges:
                    self._merge_edge(session, edge)
        if not meta.get("skipped_duplicate"):
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
        session.run(
            "CREATE INDEX doc_revision_logical IF NOT EXISTS "
            f"FOR (n:{DOC_REVISION_LABEL}) ON (n.logical_doc_id)"
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
        session.execute_write(self._install_revision_tx, plan, nodes, edges)

    @staticmethod
    def _install_revision_tx(tx, plan: DocumentRevisionPlan, nodes, edges) -> None:
        tx.run(
            f"""
            MERGE (dl:{DOCUMENT_LOGICAL_LABEL} {{logical_id: $logical_id}})
            ON CREATE SET dl.title = $title, dl.created_at = timestamp()
            ON MATCH SET dl.title = coalesce(dl.title, $title),
                         dl.updated_at = timestamp()
            """,
            logical_id=plan.logical_id,
            title=plan.title,
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
        edges_by_rel: Dict[str, List[DKGEdge]] = defaultdict(list)
        for edge in edges:
            rel = edge.rel_type.value if isinstance(edge.rel_type, RelType) else str(edge.rel_type)
            if rel not in skip_rels:
                edges_by_rel[rel].append(edge)

        for rel_type, rel_edges in edges_by_rel.items():
            for chunk_start in range(0, len(rel_edges), NEO4J_WRITE_BATCH):
                chunk = rel_edges[chunk_start : chunk_start + NEO4J_WRITE_BATCH]
                rows = [
                    {
                        "source_id": e.source_id,
                        "target_id": e.target_id,
                        "weight": e.weight,
                        "properties": json.dumps(e.properties),
                    }
                    for e in chunk
                ]
                tx.run(
                    f"UNWIND $rows AS row "
                    "MATCH (a {id: row.source_id}), (b {id: row.target_id}) "
                    f"MERGE (a)-[r:{rel_type}]->(b) "
                    "SET r.weight = row.weight, r.properties = row.properties",
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

    @staticmethod
    def _node_to_param_dict(node: DKGNode) -> dict:
        """Serialise a DKGNode to a plain dict for use in UNWIND parameters."""
        return {
            "id": node.id,
            "title": node.title,
            "text": node.text,
            "order": node.order,
            "page_start": node.page_start,
            "page_end": node.page_end,
            "depth": node.depth,
            "entities": node.entities,
            "cluster_id": node.cluster_id,
            "embedding": node.embedding,
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
        }

    @staticmethod
    def _create_node_tx(tx, node: DKGNode) -> None:
        label = node.type.value if isinstance(node.type, NodeType) else str(node.type)
        tx.run(
            f"CREATE (n:{label} {{id: $id}}) "
            "SET n.title = $title, n.text = $text, n.order = $order,"
            " n.page_start = $page_start, n.page_end = $page_end,"
            " n.depth = $depth, n.entities = $entities, n.cluster_id = $cluster_id,"
            " n.embedding = $embedding, n.visual_content = $visual_content,"
            " n.pdf_page = $pdf_page, n.document_page = $document_page,"
            " n.page_tags = $page_tags,"
            " n.region_kind = $region_kind, n.region_tags = $region_tags,"
            " n.logical_doc_id = $logical_doc_id, n.revision_id = $revision_id,"
            " n.lifecycle_status = $lifecycle_status, n.content_hash = $content_hash,"
            " n.version_number = $version_number, n.ingested_at = $ingested_at,"
            " n.source_filename = $source_filename",
            id=node.id,
            title=node.title,
            text=node.text,
            order=node.order,
            page_start=node.page_start,
            page_end=node.page_end,
            depth=node.depth,
            entities=node.entities,
            cluster_id=node.cluster_id,
            embedding=node.embedding,
            visual_content=node.visual_content,
            pdf_page=node.pdf_page,
            document_page=node.document_page,
            page_tags=node.page_tags or [],
            region_kind=node.region_kind,
            region_tags=node.region_tags or [],
            logical_doc_id=node.logical_doc_id,
            revision_id=node.revision_id,
            lifecycle_status=node.lifecycle_status,
            content_hash=node.content_hash,
            version_number=node.version_number,
            ingested_at=node.ingested_at,
            source_filename=node.source_filename,
        )

    @staticmethod
    def _merge_edge_tx(tx, edge: DKGEdge) -> None:
        rel_type = edge.rel_type.value if isinstance(edge.rel_type, RelType) else str(edge.rel_type)
        tx.run(
            "MATCH (a {id: $source_id}), (b {id: $target_id}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "SET r.weight = $weight, r.properties = $properties",
            source_id=edge.source_id,
            target_id=edge.target_id,
            weight=edge.weight,
            properties=json.dumps(edge.properties),
        )

    def _ensure_indexes(self, session) -> None:
        """Idempotently create full-text + vector indexes on every ingestion."""
        statements = [
            "CREATE FULLTEXT INDEX node_text_index IF NOT EXISTS "
            "FOR (n:Book|Chapter|Section|Page|Region|Concept) "
            "ON EACH [n.title, n.text, n.visual_content]",
            "CREATE FULLTEXT INDEX page_visual_index IF NOT EXISTS "
            "FOR (n:Page) ON EACH [n.visual_content, n.title, n.text, n.document_page]",
            "CREATE FULLTEXT INDEX region_tag_index IF NOT EXISTS "
            "FOR (n:Region) ON EACH [n.title, n.text, n.region_tags, n.region_kind]",
            "CREATE FULLTEXT INDEX page_number_index IF NOT EXISTS "
            "FOR (n:Page) ON EACH [n.document_page, n.page_tags, n.title]",
            "CREATE INDEX section_order IF NOT EXISTS FOR (n:Section) ON (n.order)",
            "CREATE INDEX page_order    IF NOT EXISTS FOR (n:Page)    ON (n.order)",
            "CREATE INDEX page_start    IF NOT EXISTS FOR (n:Page)    ON (n.page_start)",
            "CREATE INDEX page_pdf_page IF NOT EXISTS FOR (n:Page) ON (n.pdf_page)",
            """CREATE VECTOR INDEX section_embedding IF NOT EXISTS
            FOR (n:Section) ON (n.embedding)
            OPTIONS {indexConfig: {`vector.dimensions`: 1536, `vector.similarity_function`: 'cosine'}}""",
            "CREATE INDEX section_logical_rev IF NOT EXISTS "
            "FOR (n:Section) ON (n.logical_doc_id, n.revision_id)",
        ]
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

    def _merge_node(self, session, node: DKGNode) -> None:
        label = self._label_to_str(node.type)
        session.run(
            f"MERGE (n:{label} {{id: $id}})"
            " SET n.title = $title, n.text = $text, n.order = $order,"
            " n.page_start = $page_start, n.page_end = $page_end,"
            " n.depth = $depth, n.entities = $entities, n.cluster_id = $cluster_id,"
            " n.embedding = $embedding, n.visual_content = $visual_content,"
            " n.pdf_page = $pdf_page, n.document_page = $document_page,"
            " n.page_tags = $page_tags,"
            " n.region_kind = $region_kind, n.region_tags = $region_tags,"
            " n.logical_doc_id = $logical_doc_id, n.revision_id = $revision_id,"
            " n.lifecycle_status = $lifecycle_status, n.content_hash = $content_hash",
            id=node.id,
            title=node.title,
            text=node.text,
            order=node.order,
            page_start=node.page_start,
            page_end=node.page_end,
            depth=node.depth,
            entities=node.entities,
            cluster_id=node.cluster_id,
            embedding=node.embedding,
            visual_content=node.visual_content,
            pdf_page=node.pdf_page,
            document_page=node.document_page,
            page_tags=node.page_tags or [],
            region_kind=node.region_kind,
            region_tags=node.region_tags or [],
            logical_doc_id=node.logical_doc_id,
            revision_id=node.revision_id,
            lifecycle_status=node.lifecycle_status,
            content_hash=node.content_hash,
        )

    def _merge_edge(self, session, edge: DKGEdge) -> None:
        rel_type = self._rel_type_to_str(edge.rel_type)
        session.run(
            f"MATCH (a {{id: $source_id}}), (b {{id: $target_id}}) "
            f"MERGE (a)-[r:{rel_type}]->(b) "
            "SET r.weight = $weight, r.properties = $properties",
            source_id=edge.source_id,
            target_id=edge.target_id,
            weight=edge.weight,
            properties=json.dumps(edge.properties),
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
                      "entities", "cluster_id"]

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
                                   "weight", "axis", "properties"]
                )
                writer.writeheader()
                for e in edge_list:
                    writer.writerow({
                        "source_id":  e.source_id,
                        "target_id":  e.target_id,
                        "rel_type":   e.rel_type.value,
                        "weight":     e.weight,
                        "axis":       e.axis,
                        "properties": json.dumps(e.properties),
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
      n.cluster_id = CASE row.cluster_id WHEN '' THEN null ELSE toInteger(row.cluster_id) END;
""")

        # Axis 1 edges
        lines.append("// Axis 1 — Structural relationships")
        lines.append("""\
LOAD CSV WITH HEADERS FROM 'file:///edges/axis1_structural.csv' AS row
MATCH (a {id: row.source_id}), (b {id: row.target_id})
CALL apoc.merge.relationship(a, row.rel_type, {}, {weight: toFloat(row.weight)}, b)
YIELD rel RETURN count(rel);
""")

        # Axis 2 edges
        lines.append("// Axis 2 — Semantic relationships")
        lines.append("""\
LOAD CSV WITH HEADERS FROM 'file:///edges/axis2_semantic.csv' AS row
MATCH (a {id: row.source_id}), (b {id: row.target_id})
CALL apoc.merge.relationship(a, row.rel_type, {}, {
  weight:     toFloat(row.weight),
  properties: row.properties
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
            text_escaped  = n.text.replace("'", "\\'").replace("\n", "\\n")
            title_escaped = n.title.replace("'", "\\'")
            cluster       = f", n.cluster_id={n.cluster_id}" if n.cluster_id is not None else ""
            label = self._label_to_str(n.type)
            # Embedding — stored as native float list, not string
            embedding_str = json.dumps(n.embedding) if n.embedding else "null"
            lines.append(
                f"MERGE (n:{label} {{id: '{n.id}'}})"
                f" SET n.title='{title_escaped}', n.text='{text_escaped}',"
                f" n.order={n.order}, n.page_start={n.page_start},"
                f" n.page_end={n.page_end}, n.depth={n.depth},"
                f" n.entities='{entities_str}'{cluster},"
                f" n.embedding={embedding_str};"
            )

        lines += ["\n// ── AXIS 1 — STRUCTURAL EDGES ───────────────"]
        axis1_edges = [e for e in edges if e.axis == 1]
        for e in axis1_edges:
            rel_type = self._rel_type_to_str(e.rel_type)
            lines.append(
                f"MATCH (a {{id: '{e.source_id}'}}), (b {{id: '{e.target_id}'}})"
                f" MERGE (a)-[:{rel_type} {{weight: {e.weight}}}]->(b);"
            )

        lines += ["\n// ── AXIS 2 — SEMANTIC EDGES ─────────────────"]
        axis2_edges = [e for e in edges if e.axis == 2]
        for e in axis2_edges:
            rel_type = self._rel_type_to_str(e.rel_type)
            props_str = json.dumps(e.properties).replace("'", "\\'")
            lines.append(
                f"MATCH (a {{id: '{e.source_id}'}}), (b {{id: '{e.target_id}'}})"
                f" MERGE (a)-[:{rel_type} {{weight: {e.weight},"
                f" props: '{props_str}'}}]->(b);"
            )

        (self.out / "full_import.cypher").write_text("\n".join(lines))