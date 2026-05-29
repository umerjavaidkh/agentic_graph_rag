#!/usr/bin/env python3
"""Check Page.visual_content after ingest with ENABLE_PAGE_VISION=true."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from neo4j import GraphDatabase
from src.config.settings import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


def main() -> None:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    with driver.session() as s:
        stats = s.run(
            """
            MATCH (p:Page)
            RETURN count(p) AS total,
                   count(p.visual_content) AS with_visual
            """
        ).single()
        print(f"Pages total: {stats['total']}, with visual_content: {stats['with_visual']}")

        diff = s.run(
            """
            MATCH (p:Page)
            WHERE p.document_page IS NOT NULL AND p.pdf_page IS NOT NULL
              AND p.document_page <> toString(p.pdf_page)
            RETURN count(p) AS n
            """
        ).single()["n"]
        print(f"Pages where printed label differs from PDF index: {diff}")

        rows = s.run(
            """
            MATCH (p:Page)
            WHERE p.visual_content IS NOT NULL
            RETURN p.id AS id, p.pdf_page AS pdf, p.document_page AS doc,
                   size(p.visual_content) AS chars,
                   substring(p.visual_content, 0, 200) AS preview
            ORDER BY p.pdf_page
            LIMIT 10
            """
        ).data()
        if not rows:
            print("\nNo visual_content yet. Re-ingest a PDF with ENABLE_PAGE_VISION=true.")
            return
        print("\nSample pages with vision text:")
        for r in rows:
            print(f"  {r['id']} (pdf={r['pdf']}, doc={r['doc']}, {r['chars']} chars)")
            print(f"    {r['preview'][:180]}...")

        a6 = s.run(
            """
            MATCH (p:Page)
            WHERE toLower(p.visual_content) CONTAINS 'table a6'
               OR toLower(p.text) CONTAINS 'table a6'
            RETURN p.id AS id, p.order AS page,
                   p.visual_content IS NOT NULL AS has_visual
            LIMIT 5
            """
        ).data()
        print("\nTable A6 matches:")
        for r in a6:
            print(f"  {r}")

    driver.close()


if __name__ == "__main__":
    main()
