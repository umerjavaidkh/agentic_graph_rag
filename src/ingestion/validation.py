"""ingestion/validation.py — cheap, LLM-free post-ingestion quality report.

Pure Cypher aggregation against the already-ingested graph. Makes no
OpenAI/embedding calls, so it costs nothing to run — including across an
entire 1000-document corpus — unlike re-ingesting or running the eval suite.
"""
from __future__ import annotations

from typing import Any, Optional

from ..graph.constants import DOC_REVISION_LABEL

_MIN_TEXT_CHARS = 20  # below this, a node's text is treated as effectively empty
_CONTENT_LABELS = ("Chapter", "Section", "Page")
_NER_LABELS = ("Section", "Page")  # matches axis2.py's CONCEPT_NODE_TYPES
_EMBED_LABELS = ("Chapter", "Section")  # matches axis2.py's SEMANTIC_NODE_TYPES
_SEMANTIC_REL_TYPES = (
    "SEMANTICALLY_SIMILAR",
    "SHARES_ENTITY",
    "SAME_CATEGORY",
    "CONTRADICTS",
    "ELABORATES",
    "PREREQUISITE_OF",
)


def resolve_active_revision(session, logical_doc_id: str) -> Optional[dict]:
    row = session.run(
        f"""
        MATCH (rev:{DOC_REVISION_LABEL} {{logical_doc_id: $logical_doc_id, lifecycle_status: 'ACTIVE'}})
        RETURN rev.id AS revision_id, rev.version_number AS version_number,
               rev.source_filename AS source_filename, rev.title AS title
        """,
        logical_doc_id=logical_doc_id,
    ).single()
    return dict(row) if row else None


def list_ingested_documents(session, tenant_id: Optional[str] = None, limit: int = 200) -> list[dict]:
    """Cheap listing for a UI picker: every logical document with an ACTIVE revision."""
    where = "rev.lifecycle_status = 'ACTIVE'"
    params: dict[str, Any] = {"limit": limit}
    if tenant_id:
        where += " AND rev.tenant_id = $tenant_id"
        params["tenant_id"] = tenant_id
    rows = session.run(
        f"""
        MATCH (rev:{DOC_REVISION_LABEL})
        WHERE {where}
        RETURN rev.logical_doc_id AS logical_doc_id, rev.id AS revision_id,
               rev.source_filename AS source_filename, rev.title AS title,
               rev.version_number AS version_number, rev.ingested_at AS ingested_at
        ORDER BY rev.ingested_at DESC
        LIMIT $limit
        """,
        **params,
    )
    return [dict(r) for r in rows]


def build_ingestion_quality_report(
    driver,
    logical_doc_id: str,
    revision_id: Optional[str] = None,
) -> dict[str, Any]:
    """Cheap, deterministic, LLM-free quality report for one ingested document."""
    with driver.session() as session:
        if revision_id is None:
            active = resolve_active_revision(session, logical_doc_id)
            if active is None:
                return {"logical_doc_id": logical_doc_id, "found": False}
            revision_id = active["revision_id"]

        scope = {"logical_doc_id": logical_doc_id, "revision_id": revision_id}

        node_counts = {
            r["label"]: r["n"]
            for r in session.run(
                f"""
                MATCH (n) WHERE n.logical_doc_id = $logical_doc_id AND n.revision_id = $revision_id
                  AND NOT n:{DOC_REVISION_LABEL}
                RETURN labels(n)[0] AS label, count(n) AS n
                """,
                **scope,
            )
        }

        edge_counts = {
            r["rel_type"]: r["n"]
            for r in session.run(
                """
                MATCH (n)-[r]->()
                WHERE n.logical_doc_id = $logical_doc_id AND n.revision_id = $revision_id
                RETURN type(r) AS rel_type, count(r) AS n
                """,
                **scope,
            )
        }

        text_row = session.run(
            """
            MATCH (n) WHERE n.logical_doc_id = $logical_doc_id AND n.revision_id = $revision_id
              AND any(l IN labels(n) WHERE l IN $labels)
            RETURN count(n) AS total,
                   sum(CASE WHEN size(coalesce(n.text, '')) < $min_chars THEN 1 ELSE 0 END) AS empty
            """,
            labels=list(_CONTENT_LABELS),
            min_chars=_MIN_TEXT_CHARS,
            **scope,
        ).single()

        ner_row = session.run(
            """
            MATCH (n) WHERE n.logical_doc_id = $logical_doc_id AND n.revision_id = $revision_id
              AND any(l IN labels(n) WHERE l IN $labels)
            RETURN count(n) AS total,
                   sum(CASE WHEN size(coalesce(n.entities, [])) > 0 THEN 1 ELSE 0 END) AS with_entities
            """,
            labels=list(_NER_LABELS),
            **scope,
        ).single()

        embed_row = session.run(
            """
            MATCH (n) WHERE n.logical_doc_id = $logical_doc_id AND n.revision_id = $revision_id
              AND any(l IN labels(n) WHERE l IN $labels)
            RETURN count(n) AS total,
                   sum(CASE WHEN n.embedding IS NOT NULL THEN 1 ELSE 0 END) AS with_embedding
            """,
            labels=list(_EMBED_LABELS),
            **scope,
        ).single()

        page_numbers = [
            r["page_no"]
            for r in session.run(
                """
                MATCH (n:Page) WHERE n.logical_doc_id = $logical_doc_id AND n.revision_id = $revision_id
                RETURN n.order AS page_no ORDER BY n.order
                """,
                **scope,
            )
        ]

        orphan_row = session.run(
            """
            MATCH (n) WHERE n.logical_doc_id = $logical_doc_id AND n.revision_id = $revision_id
              AND any(l IN labels(n) WHERE l IN $labels)
              AND NOT ( ()-[:CONTAINS]->(n) )
            RETURN count(n) AS n
            """,
            labels=list(_CONTENT_LABELS),
            **scope,
        ).single()

    text_coverage = _coverage(text_row["total"], (text_row["total"] or 0) - (text_row["empty"] or 0))
    text_coverage["empty_or_near_empty"] = text_row["empty"] or 0
    ner_coverage = _coverage(ner_row["total"], ner_row["with_entities"])
    embedding_coverage = _coverage(embed_row["total"], embed_row["with_embedding"])
    page_continuity = _page_continuity(page_numbers)
    orphan_nodes = orphan_row["n"] or 0
    semantic_edge_count = sum(edge_counts.get(rel, 0) for rel in _SEMANTIC_REL_TYPES)

    flags = _flag_anomalies(
        node_counts=node_counts,
        semantic_edge_count=semantic_edge_count,
        text_coverage=text_coverage,
        ner_coverage=ner_coverage,
        embedding_coverage=embedding_coverage,
        page_continuity=page_continuity,
        orphan_nodes=orphan_nodes,
    )

    return {
        "logical_doc_id": logical_doc_id,
        "revision_id": revision_id,
        "found": True,
        "node_counts": node_counts,
        "edge_counts": edge_counts,
        "semantic_edge_count": semantic_edge_count,
        "text_coverage": text_coverage,
        "ner_coverage": ner_coverage,
        "embedding_coverage": embedding_coverage,
        "page_continuity": page_continuity,
        "orphan_nodes": orphan_nodes,
        "flags": flags,
    }


def _coverage(total: Optional[int], good_count: Optional[int]) -> dict[str, Any]:
    total = total or 0
    good = good_count or 0
    pct = round(100.0 * good / total, 1) if total else 0.0
    return {"total": total, "good": good, "pct": pct}


def _page_continuity(page_numbers: list[int]) -> dict[str, Any]:
    if not page_numbers:
        return {"count": 0, "min": None, "max": None, "gaps": []}
    lo, hi = min(page_numbers), max(page_numbers)
    present = set(page_numbers)
    gaps = [p for p in range(lo, hi + 1) if p not in present]
    return {"count": len(page_numbers), "min": lo, "max": hi, "gaps": gaps[:50]}


def _flag_anomalies(
    *,
    node_counts: dict[str, int],
    semantic_edge_count: int,
    text_coverage: dict[str, Any],
    ner_coverage: dict[str, Any],
    embedding_coverage: dict[str, Any],
    page_continuity: dict[str, Any],
    orphan_nodes: int,
) -> list[str]:
    flags: list[str] = []
    if not node_counts:
        flags.append("No nodes found for this logical_doc_id/revision — ingestion may have failed entirely.")
        return flags

    if text_coverage["total"] and text_coverage["pct"] < 90:
        flags.append(
            f"{text_coverage['empty_or_near_empty']}/{text_coverage['total']} pages/sections have "
            f"< {_MIN_TEXT_CHARS} chars of text — possible OCR gap or extraction failure."
        )
    if ner_coverage["total"] and ner_coverage["good"] == 0:
        flags.append("0% NER coverage — entity extraction may have failed silently (check OPENAI_API_KEY / logs).")
    if embedding_coverage["total"] and embedding_coverage["good"] == 0:
        flags.append("0% embedding coverage — semantic enrichment (Axis 2) may not have run.")
    if node_counts.get("Chapter", 0) + node_counts.get("Section", 0) > 1 and semantic_edge_count == 0:
        flags.append("No semantic edges (SIMILAR/SHARES_ENTITY/SAME_CATEGORY/...) at all — enrichment likely didn't run.")
    if page_continuity["gaps"]:
        flags.append(
            f"{len(page_continuity['gaps'])} missing page number(s) in sequence "
            f"(first few: {page_continuity['gaps'][:10]})."
        )
    if orphan_nodes:
        flags.append(f"{orphan_nodes} node(s) have no parent CONTAINS edge — broken hierarchy.")

    return flags
