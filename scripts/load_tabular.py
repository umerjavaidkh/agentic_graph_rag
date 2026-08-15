#!/usr/bin/env python3
"""
Load CSV / Excel / SQLite into the Neo4j structured graph.

Prints the schema it inferred and stops, unless told to load. An inferred
relationship that is wrong is indistinguishable from a correct one once it is
in the graph, so the plan is shown first by default rather than after.

Usage:
    python scripts/load_tabular.py --source ./csv_dir          # dry run
    python scripts/load_tabular.py --source data.sqlite --load
    python scripts/load_tabular.py --source book.xlsx --load --clear
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.graph.driver import get_neo4j_driver  # noqa: E402
from src.ingestion.tabular import infer_schema, load_schema, source_tag  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="CSV directory, .xlsx workbook, or .sqlite file")
    ap.add_argument("--load", action="store_true", help="actually write to Neo4j (default: dry run)")
    ap.add_argument("--clear", action="store_true", help="delete the labels this schema owns first")
    args = ap.parse_args()

    source = Path(args.source).expanduser()
    if not source.exists():
        sys.exit(f"not found: {source}")

    schema = infer_schema(source)
    print(f"Inferred from {source.name}:\n")
    print(schema.describe())

    links = sum(len(t.foreign_keys) for t in schema.tables)
    print(f"\n  {len(schema.tables)} tables, {links} relationship(s) inferred")

    if not args.load:
        print("\nDry run — nothing written. Re-run with --load once the plan above looks right.")
        return

    driver = get_neo4j_driver()
    with driver.session() as s:
        if args.clear:
            # Only nodes THIS source loaded. Labels are inferred from table
            # names and collide across unrelated datasets, so clearing by
            # label alone deletes other people's data silently.
            tag = source_tag(source)
            removed = 0
            for t in schema.tables:
                while True:
                    n = s.run(
                        f"MATCH (n:{t.label}) WHERE n._source = $tag "
                        f"WITH n LIMIT 10000 DETACH DELETE n RETURN count(n) AS n",
                        tag=tag,
                    ).single()["n"]
                    if not n:
                        break
                    removed += n
            print(f"  cleared {removed:,} node(s) previously loaded from {source.name}")
            for t in schema.tables:
                other = s.run(
                    f"MATCH (n:{t.label}) WHERE coalesce(n._source, '') <> $tag RETURN count(n) AS n",
                    tag=tag,
                ).single()["n"]
                if other:
                    print(f"  note: {other:,} existing :{t.label} node(s) from another source were left alone")

        print("\nLoading:")
        counts = load_schema(s, source, schema)
        for name, n in counts.items():
            print(f"  {name:<32} {n:>9,}")

        print("\nGraph now holds:")
        labels = [t.label for t in schema.tables]
        for r in s.run(
            "MATCH (n) UNWIND labels(n) AS l WITH l WHERE l IN $labels "
            "RETURN l AS label, count(*) AS n ORDER BY n DESC", labels=labels
        ):
            print(f"  {r['label']:<20} {r['n']:>9,}")
        rels = s.run(
            "MATCH (a)-[r]->(b) WHERE any(l IN labels(a) WHERE l IN $labels) "
            "RETURN type(r) AS t, count(r) AS n ORDER BY n DESC LIMIT 10", labels=labels
        )
        for r in rels:
            print(f"  [{r['t']}]{'':<8} {r['n']:>9,}")


if __name__ == "__main__":
    main()
