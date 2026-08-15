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
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.document.ontology_report import run_for_doc  # noqa: E402
from src.document.ontology_validation import ONTOLOGY_ACCURACY_TARGET  # noqa: E402
from src.graph.driver import get_neo4j_driver  # noqa: E402
from src.ingestion.validation import list_ingested_documents  # noqa: E402


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
        # score is the WORST dimension, so always show which one that was --
        # otherwise a single number gives no clue where to look.
        dims = a1.get("dimensions") or {}
        if dims:
            print("      dimensions: " + "  ".join(
                f"{name}={value:.1%}" for name, value in sorted(dims.items(), key=lambda kv: kv[1])
            ))
        if a1["mismatches"]:
            print(f"      mismatches: {a1['mismatches'][:3]}")
    if a2 is not None:
        print(f"    axis2: score={a2['score']:.2%} "
              f"edge_precision={a2['edge_precision']} entity_grounding={a2['entity_grounding_precision']} "
              f"(sampled {a2['sampled_edges']} edges, {a2['sampled_entities']} entities)")
        by_type = a2.get("edge_precision_by_type") or {}
        if by_type:
            print("      edge precision by type: " + "  ".join(
                f"{rel}={value:.1%}" for rel, value in sorted(by_type.items(), key=lambda kv: kv[1])
            ))
        # An interval, not just a point: a sampled run is not a measurement,
        # and two runs whose intervals overlap have not shown a difference.
        for label, key in (("edge", "edge_precision_ci"), ("entity", "entity_grounding_ci")):
            ci = a2.get(key)
            if ci:
                print(f"      {label} precision 95% CI: [{ci[0]:.2f}, {ci[1]:.2f}]"
                      f"  (width {ci[1] - ci[0]:.2f})")
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
            driver, doc_id, args.sample_size,
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
