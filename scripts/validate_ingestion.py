#!/usr/bin/env python3
"""
Cheap, LLM-free ingestion-quality report for one or all ingested documents.

No OpenAI calls — pure Cypher aggregation against the already-ingested graph,
using the same Neo4j connection settings as the running app.

Examples:
  python scripts/validate_ingestion.py --doc cost-test-aapl-10k-2024
  python scripts/validate_ingestion.py --all
  python scripts/validate_ingestion.py --all --tenant default --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.neo4j.driver import get_neo4j_driver  # noqa: E402
from src.ingestion.validation import (  # noqa: E402
    build_ingestion_quality_report,
    list_ingested_documents,
)


def _print_report(report: dict) -> None:
    if not report.get("found"):
        print(f"  NOT FOUND: {report['logical_doc_id']}")
        return

    print(f"  logical_doc_id: {report['logical_doc_id']}  (revision {report['revision_id']})")
    print("  nodes:  " + (", ".join(f"{k}={v}" for k, v in sorted(report["node_counts"].items())) or "(none)"))
    print("  edges:  " + (", ".join(f"{k}={v}" for k, v in sorted(report["edge_counts"].items())) or "(none)"))
    tc = report["text_coverage"]
    print(f"  text coverage:      {tc['pct']}%  ({tc['good']}/{tc['total']} with content)")
    nc = report["ner_coverage"]
    print(f"  NER coverage:       {nc['pct']}%  ({nc['good']}/{nc['total']} with entities)")
    ec = report["embedding_coverage"]
    print(f"  embedding coverage: {ec['pct']}%  ({ec['good']}/{ec['total']} embedded)")
    pc = report["page_continuity"]
    print(f"  pages: {pc['count']} (range {pc['min']}-{pc['max']}, {len(pc['gaps'])} gap(s))")
    print(f"  orphan nodes: {report['orphan_nodes']}")
    if report["flags"]:
        print("  FLAGS:")
        for f in report["flags"]:
            print(f"    - {f}")
    else:
        print("  FLAGS: none")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--doc", help="logical_doc_id to validate")
    parser.add_argument("--all", action="store_true", help="validate every ingested document")
    parser.add_argument("--tenant", default=None, help="filter --all by tenant_id")
    parser.add_argument("--json", action="store_true", help="print raw JSON instead of a human report")
    args = parser.parse_args()

    if not args.doc and not args.all:
        parser.error("pass --doc <logical_doc_id> or --all")

    driver = get_neo4j_driver()
    reports = []

    if args.all:
        with driver.session() as session:
            docs = list_ingested_documents(session, tenant_id=args.tenant)
        if not docs:
            print("No ingested documents found.")
            return 0
        for d in docs:
            report = build_ingestion_quality_report(driver, d["logical_doc_id"], d["revision_id"])
            reports.append(report)
            if not args.json:
                print(f"\n{d['source_filename'] or d['logical_doc_id']}")
                _print_report(report)
    else:
        report = build_ingestion_quality_report(driver, args.doc)
        reports.append(report)
        if not args.json:
            _print_report(report)

    if args.json:
        print(json.dumps(reports, indent=2, default=str))

    flagged = sum(1 for r in reports if r.get("flags"))
    if not args.json:
        print(f"\n{len(reports)} document(s) checked, {flagged} with flags.")
    return 1 if flagged else 0


if __name__ == "__main__":
    sys.exit(main())
