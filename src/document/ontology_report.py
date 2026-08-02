"""ontology_report.py — wires ontology_validation.py's pure scoring logic
up against a live document (Neo4j + blob store + LLM judge).

Split out of scripts/validate_ontology_accuracy.py so both the CLI script
and the Graph Inspector's "Quality" panel (GET /documents/{id}/ontology-score
in src/api.py) can call the same run_for_doc() rather than duplicating the
Neo4j queries. ontology_validation.py itself stays pure/Neo4j-free by
design (see its own docstring) -- this is the impure wiring layer.
"""
from __future__ import annotations

import random
from typing import Optional

from ..config.settings import CHAT_MODEL
from ..model_providers.factory import get_chat_provider
from ..storage.blob.factory import get_blob_store
from ..storage.hydrator import get_hydrator
from .ontology_validation import (
    ONTOLOGY_ACCURACY_TARGET,
    extract_toc_ground_truth,
    score_axis1_against_toc,
    score_axis1_structural_invariants,
    score_axis2_idea_linking,
)
from .versioning import source_file_blob_key

_SEMANTIC_REL_TYPES = ("SHARES_ENTITY", "SAME_CATEGORY", "SEMANTICALLY_SIMILAR")


def _fetch_structural_nodes(session, logical_doc_id: str, revision_id: str) -> list[dict]:
    # Includes the Document root alongside Chapter/Section -- a Section
    # whose only parent is the Document itself (no intervening Chapter,
    # common for SEC filings that never carry a Chapter tier) is a
    # perfectly legitimate structure, not an orphan. Excluding Document
    # from this set made every such Section's real CONTAINS parent
    # unresolvable to score_axis1_structural_invariants (parent_id pointed
    # at a node not present in `constructed`), which misread it as "no
    # parent" -- a false positive in the scorer, not a defect in the graph.
    rows = session.run(
        """
        MATCH (n) WHERE n.logical_doc_id = $logical_doc_id AND n.revision_id = $revision_id
          AND (n:Chapter OR n:Section OR n:Document)
        OPTIONAL MATCH (parent)-[:CONTAINS]->(n)
        RETURN n.id AS id, n.title AS title, n.depth AS depth,
               n.page_start AS page_start, n.page_end AS page_end,
               parent.id AS parent_id
        """,
        logical_doc_id=logical_doc_id,
        revision_id=revision_id,
    )
    return [dict(r) for r in rows]


def _fetch_pdf_bytes(logical_doc_id: str, revision_meta: dict, tenant_id: str) -> Optional[bytes]:
    key = source_file_blob_key(
        tenant_id=revision_meta.get("tenant_id") or tenant_id,
        logical_id=logical_doc_id,
        revision_id=revision_meta["revision_id"],
        source_filename=revision_meta.get("source_filename") or "",
    )
    try:
        return get_blob_store().get_bytes(key)
    except Exception:
        return None


def _sample_edges(session, logical_doc_id: str, revision_id: str, n: int) -> list[dict]:
    """Hydrates source/target text from blob_key_text rather than reading
    `.text` off Neo4j directly (docs/DESIGN_unstructured_graph_v2.md phase
    3) -- full-text hydration, not search_text, so this scoring pass reads
    the exact same text pre- and post-migration. score_axis2_idea_linking
    truncates to [:800] regardless, so identical source text in means
    identical judge input."""
    rows = session.run(
        """
        MATCH (a)-[r]->(b)
        WHERE a.logical_doc_id = $logical_doc_id AND a.revision_id = $revision_id
          AND type(r) IN $rel_types
        RETURN a.blob_key_text AS source_blob_key, b.blob_key_text AS target_blob_key,
               type(r) AS rel_type, coalesce(r.properties, '') AS shared
        ORDER BY rand()
        LIMIT $n
        """,
        logical_doc_id=logical_doc_id,
        revision_id=revision_id,
        rel_types=list(_SEMANTIC_REL_TYPES),
        n=n,
    )
    hydrator = get_hydrator()
    return [
        {
            "source_text": hydrator.hydrate(r["source_blob_key"]),
            "target_text": hydrator.hydrate(r["target_blob_key"]),
            "rel_type": r["rel_type"],
            "shared": r["shared"],
        }
        for r in rows
    ]


def _sample_entities(session, logical_doc_id: str, revision_id: str, n: int) -> list[dict]:
    rows = session.run(
        """
        MATCH (node) WHERE node.logical_doc_id = $logical_doc_id AND node.revision_id = $revision_id
          AND size(coalesce(node.entities, [])) > 0
        RETURN node.blob_key_text AS source_blob_key, node.entities AS entities
        ORDER BY rand()
        LIMIT $n
        """,
        logical_doc_id=logical_doc_id,
        revision_id=revision_id,
        n=n,
    )
    hydrator = get_hydrator()
    out: list[dict] = []
    for r in rows:
        text = hydrator.hydrate(r["source_blob_key"])
        for ent in r["entities"] or []:
            out.append({"source_text": text, "entity": ent})
    random.shuffle(out)
    return out[:n]


def run_for_doc(
    driver, logical_doc_id: str, sample_size: int,
    *, skip_axis1: bool = False, skip_axis2: bool = False,
) -> dict:
    from ..ingestion.validation import resolve_active_revision

    with driver.session() as session:
        active = resolve_active_revision(session, logical_doc_id)
        if active is None:
            return {"logical_doc_id": logical_doc_id, "found": False}
        revision_id = active["revision_id"]

        axis1 = None
        if not skip_axis1:
            constructed = _fetch_structural_nodes(session, logical_doc_id, revision_id)
            pdf_bytes = _fetch_pdf_bytes(logical_doc_id, {**active, "revision_id": revision_id}, "default")
            toc = extract_toc_ground_truth(pdf_bytes) if pdf_bytes else None
            if toc:
                # TOC ground truth only describes Chapter/Section tiers --
                # the Document root (included above so
                # structural_invariants can resolve Chapter-less Sections'
                # real parent) would never match an outline entry and must
                # not be counted against precision.
                chapter_or_section = [n for n in constructed if (n.get("depth") or 0) >= 1]
                axis1 = score_axis1_against_toc(chapter_or_section, toc, logical_doc_id)
            else:
                axis1 = score_axis1_structural_invariants(constructed, logical_doc_id)

        axis2 = None
        if not skip_axis2:
            edges_sample = _sample_edges(session, logical_doc_id, revision_id, sample_size)
            entities_sample = _sample_entities(session, logical_doc_id, revision_id, sample_size)
            provider = get_chat_provider()
            axis2 = score_axis2_idea_linking(
                edges_sample,
                entities_sample,
                provider=provider,
                model=CHAT_MODEL,
                logical_doc_id=logical_doc_id,
            )

    passes_gate = (axis1 is None or axis1.score >= ONTOLOGY_ACCURACY_TARGET) and (
        axis2 is None or axis2.score >= ONTOLOGY_ACCURACY_TARGET
    )
    return {
        "logical_doc_id": logical_doc_id,
        "found": True,
        "axis1": axis1.as_dict() if axis1 else None,
        "axis2": axis2.as_dict() if axis2 else None,
        "passes_gate": passes_gate,
    }
