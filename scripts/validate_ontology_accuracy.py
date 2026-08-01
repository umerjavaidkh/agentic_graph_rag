#!/usr/bin/env python3
"""
Measured ontology-accuracy report for one or all ingested documents —
Axis-1 (structural) scored against the PDF's own embedded outline when
available, Axis-2 (idea-linking) via sampled LLM-judge precision.

This is the acceptance gate for docs/DESIGN_unstructured_graph_v2.md's
>=90% ontology-accuracy target: run it against today's ingestion as a
baseline, then re-run against the v2 GraphConstructionService output to
confirm it clears the bar before cutover.

Costs: Axis-1 is free (pure Cypher + local PDF parsing). Axis-2 makes one
small LLM call per sampled edge/entity — bounded by --sample-size, default
25+25 per document. Don't raise --sample-size casually across a whole
corpus run; see feedback_eval_suite_cost in repo memory.

Examples:
  python scripts/validate_ontology_accuracy.py --doc jnj-10k-2026-02-11
  python scripts/validate_ontology_accuracy.py --all --sample-size 15
  python scripts/validate_ontology_accuracy.py --all --json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config.settings import CHAT_MODEL  # noqa: E402
from src.document.ontology_validation import (  # noqa: E402
    ONTOLOGY_ACCURACY_TARGET,
    score_axis1_against_toc,
    score_axis1_structural_invariants,
    score_axis2_idea_linking,
    extract_toc_ground_truth,
)
from src.document.versioning import source_file_blob_key  # noqa: E402
from src.graph.driver import get_neo4j_driver  # noqa: E402
from src.ingestion.validation import list_ingested_documents, resolve_active_revision  # noqa: E402
from src.model_providers.factory import get_chat_provider  # noqa: E402
from src.storage.blob.factory import get_blob_store  # noqa: E402

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


def _fetch_pdf_bytes(logical_doc_id: str, revision_meta: dict, tenant_id: str) -> bytes | None:
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
    rows = session.run(
        """
        MATCH (a)-[r]->(b)
        WHERE a.logical_doc_id = $logical_doc_id AND a.revision_id = $revision_id
          AND type(r) IN $rel_types
        RETURN a.text AS source_text, b.text AS target_text, type(r) AS rel_type,
               coalesce(r.properties, '') AS shared
        ORDER BY rand()
        LIMIT $n
        """,
        logical_doc_id=logical_doc_id,
        revision_id=revision_id,
        rel_types=list(_SEMANTIC_REL_TYPES),
        n=n,
    )
    return [dict(r) for r in rows]


def _sample_entities(session, logical_doc_id: str, revision_id: str, n: int) -> list[dict]:
    rows = session.run(
        """
        MATCH (node) WHERE node.logical_doc_id = $logical_doc_id AND node.revision_id = $revision_id
          AND size(coalesce(node.entities, [])) > 0
        RETURN node.text AS source_text, node.entities AS entities
        ORDER BY rand()
        LIMIT $n
        """,
        logical_doc_id=logical_doc_id,
        revision_id=revision_id,
        n=n,
    )
    out: list[dict] = []
    for r in rows:
        text = r["source_text"]
        for ent in r["entities"] or []:
            out.append({"source_text": text, "entity": ent})
    random.shuffle(out)
    return out[:n]


def run_for_doc(
    driver, blob_store_ready: bool, logical_doc_id: str, sample_size: int,
    *, skip_axis1: bool = False, skip_axis2: bool = False,
) -> dict:
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


def _print_report(report: dict) -> None:
    if not report.get("found"):
        print(f"  NOT FOUND: {report['logical_doc_id']}")
        return
    a1, a2 = report["axis1"], report["axis2"]
    gate = "PASS" if report["passes_gate"] else "FAIL"
    print(f"  {report['logical_doc_id']}  [{gate}] (target >= {ONTOLOGY_ACCURACY_TARGET:.0%})")
    if a1 is not None:
        print(f"    axis1 ({a1['method']}): score={a1['score']:.2%} "
              f"precision={a1['precision']} recall={a1['recall']} "
              f"matched={a1['matched']}/{a1['total_ground_truth']}")
        if a1["mismatches"]:
            print(f"      mismatches: {a1['mismatches'][:3]}")
    if a2 is not None:
        print(f"    axis2: score={a2['score']:.2%} "
              f"edge_precision={a2['edge_precision']} entity_grounding={a2['entity_grounding_precision']} "
              f"(sampled {a2['sampled_edges']} edges, {a2['sampled_entities']} entities)")
        if a2["invalid_examples"]:
            print(f"      invalid examples: {a2['invalid_examples'][:3]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--doc", help="logical_doc_id to score")
    parser.add_argument("--all", action="store_true", help="score every ingested document")
    parser.add_argument("--tenant", default=None, help="restrict --all to one tenant")
    parser.add_argument("--sample-size", type=int, default=25, help="Axis-2 sample size (edges + entities)")
    parser.add_argument("--axis1-only", action="store_true", help="skip Axis-2 (no LLM calls, free/fast)")
    parser.add_argument("--axis2-only", action="store_true", help="skip Axis-1")
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON instead of text")
    args = parser.parse_args()

    if not args.doc and not args.all:
        parser.error("pass --doc <logical_doc_id> or --all")
    if args.axis1_only and args.axis2_only:
        parser.error("--axis1-only and --axis2-only are mutually exclusive")

    driver = get_neo4j_driver()
    doc_ids: list[str]
    if args.all:
        with driver.session() as session:
            doc_ids = [d["logical_doc_id"] for d in list_ingested_documents(session, tenant_id=args.tenant)]
    else:
        doc_ids = [args.doc]

    reports = [
        run_for_doc(
            driver, True, doc_id, args.sample_size,
            skip_axis1=args.axis2_only, skip_axis2=args.axis1_only,
        )
        for doc_id in doc_ids
    ]

    if args.json:
        print(json.dumps(reports, indent=2))
        return

    for report in reports:
        _print_report(report)
    scored = [r for r in reports if r.get("found")]
    if scored:
        summary_parts = []
        if not args.axis2_only:
            avg1 = sum(r["axis1"]["score"] for r in scored) / len(scored)
            summary_parts.append(f"axis1={avg1:.2%}")
        if not args.axis1_only:
            avg2 = sum(r["axis2"]["score"] for r in scored) / len(scored)
            summary_parts.append(f"axis2={avg2:.2%}")
        n_pass = sum(1 for r in scored if r["passes_gate"])
        print(f"\n  corpus average: {' '.join(summary_parts)}  "
              f"({n_pass}/{len(scored)} documents pass the >= {ONTOLOGY_ACCURACY_TARGET:.0%} gate)")


if __name__ == "__main__":
    main()
