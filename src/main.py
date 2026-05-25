"""
main.py — Orchestrates the full pipeline.

Usage:
    # Axis 1 only (no API key needed)
    python main.py book.pdf

    # Axis 1 + Axis 2 cheap (embeddings + NER + clustering)
    OPENAI_API_KEY=sk-... python main.py book.pdf

    # Full (includes LLM CONTRADICTS/ELABORATES/PREREQUISITE_OF pass)
    OPENAI_API_KEY=sk-... python main.py book.pdf --llm-pass

    # Custom output dir
    python main.py book.pdf --output ./my_graph
"""
import argparse
import os
import sys
from pathlib import Path

from .config.settings import OPENAI_API_KEY
from .document.parser import DoclingParser
from .semantic.axis2 import Axis2Builder
from .exporter.exporter import Neo4jExporter


def run(source: str, output_dir: str, llm_pass: bool) -> None:
    print(f"\n📄 Parsing: {source}")
    print("━" * 50)

    # ── Phase 1: Docling → Node tree + Axis 1 edges ──────────
    print("🌳 Phase 1: Building document tree (Axis 1)...")
    parser = DoclingParser()
    nodes, edges = parser.parse(source)

    axis1_count = sum(1 for e in edges if e.axis == 1)
    print(f"   ✓ {len(nodes)} nodes | {axis1_count} structural edges")

    # ── Phase 2: Axis 2 semantic edges ───────────────────────
    api_key = OPENAI_API_KEY
    if api_key:
        print(f"\n🔗 Phase 2: Semantic relationship discovery (Axis 2)...")
        if llm_pass:
            print("   ⚡ LLM pass enabled (CONTRADICTS/ELABORATES/PREREQUISITE_OF)")
        builder = Axis2Builder(api_key=api_key)
        nodes, semantic_edges = builder.build(nodes, run_llm_pass=llm_pass)
        edges += semantic_edges

        axis2_count = sum(1 for e in edges if e.axis == 2)
        print(f"   ✓ {axis2_count} semantic edges added")
    else:
        print("\n⚠️  No OPENAI_API_KEY — skipping Axis 2 (structural only)")
        print("   Set OPENAI_API_KEY to enable embeddings + NER + clustering")

    # ── Phase 3: Export to Neo4j ──────────────────────────────
    print(f"\n📦 Phase 3: Exporting to Neo4j ({output_dir})...")
    exporter = Neo4jExporter(output_dir=output_dir)
    exporter.export(nodes, edges)

    # ── Summary ───────────────────────────────────────────────
    from models import RelType
    print("\n📊 Relationship breakdown:")
    rel_counts: dict[str, int] = {}
    for e in edges:
        rel_counts[e.rel_type.value] = rel_counts.get(e.rel_type.value, 0) + 1

    axis1_rels = {RelType.CONTAINS, RelType.PART_OF, RelType.PRECEDES, RelType.FOLLOWS}
    print("\n  Axis 1 — Structural:")
    for rel, count in sorted(rel_counts.items()):
        from models import RelType as RT
        if RT(rel) in axis1_rels:
            print(f"    {rel:<25} {count}")
    print("\n  Axis 2 — Semantic:")
    for rel, count in sorted(rel_counts.items()):
        from models import RelType as RT
        if RT(rel) not in axis1_rels:
            print(f"    {rel:<25} {count}")

    print(f"\n✅ Done! Import to Neo4j:")
    print(f"   1. Run output/setup.cypher  (once)")
    print(f"   2. Run output/full_import.cypher  OR  use LOAD CSV via import.cypher")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Document → Neo4j Knowledge Graph")
    ap.add_argument("source",     help="Path to PDF, DOCX, PPTX, HTML, etc.")
    ap.add_argument("--output",   default="output", help="Output directory")
    ap.add_argument("--llm-pass", action="store_true",
                    help="Run expensive LLM pass for CONTRADICTS/ELABORATES/PREREQUISITE_OF")
    args = ap.parse_args()

    if not Path(args.source).exists():
        print(f"❌ File not found: {args.source}")
        sys.exit(1)

    run(args.source, args.output, args.llm_pass)
