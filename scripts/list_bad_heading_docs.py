"""list_bad_heading_docs.py — a worklist of documents to re-parse.

Companion to audit_headings.py, which explains the failure modes. This
one answers only "which files do I open", so it resolves each document
back to a real PDF path on disk -- DocRevision.source_filename holds a
blob hash, which identifies a document but does not help anyone find it.

    docker compose exec -T app python - < scripts/list_bad_heading_docs.py \\
        > eval/bad_heading_documents.md
"""
from __future__ import annotations

import glob
import os
import re
from collections import defaultdict

from src.shared.neo4j.driver import get_neo4j_driver

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
_MATH = re.compile(r"[=∑∫≤≥±ℓμσ→∈]|\\\\[a-z]+\{")
_MAX_HEADING_CHARS = 120


def reasons(title: str) -> list[str]:
    t = (title or "").strip()
    out: list[str] = []
    if t.startswith("|"):
        out.append("grid-row")
    if len(t) > _MAX_HEADING_CHARS:
        out.append("body-sentence")
    if t.rstrip().endswith("-"):
        out.append("line-break")
    if t.rstrip().endswith(","):
        out.append("mid-sentence")
    digits = sum(c.isdigit() for c in t)
    if t and digits > len(t) * 0.3:
        out.append("mostly-digits")
    if _MATH.search(t) and len(_WORD.findall(t)) < 4:
        out.append("equation")
    if t and not _WORD.search(t):
        out.append("no-words")
    return out


def _disk_index() -> dict[str, str]:
    """Basename without extension -> path, over the ingested corpora."""
    index: dict[str, str] = {}
    for path in glob.glob("sample_data_to_test/**/*.pdf", recursive=True):
        index.setdefault(os.path.splitext(os.path.basename(path))[0], path)
    return index


def main() -> None:
    per_doc: dict[str, list[str]] = defaultdict(list)
    with get_neo4j_driver().session() as s:
        for r in s.run(
            """
            MATCH (n:Chapter|Section)
            WHERE n.lifecycle_status = 'ACTIVE'
              AND trim(coalesce(n.title, '')) <> ''
            RETURN n.logical_doc_id AS doc, trim(n.title) AS t
            """
        ):
            if r["doc"]:
                per_doc[r["doc"]].append(r["t"])

    disk = _disk_index()
    rows = []
    for doc, titles in per_doc.items():
        bad = [t for t in titles if reasons(t)]
        if not bad:
            continue
        modes = sorted({m for t in bad for m in reasons(t)})
        stem = doc[4:] if doc.startswith("doc_") else doc
        rows.append({
            "doc": doc,
            "path": disk.get(stem, ""),
            "total": len(titles),
            "bad": len(bad),
            "pct": len(bad) / max(len(titles), 1) * 100,
            "modes": ",".join(modes),
        })
    rows.sort(key=lambda r: (-r["pct"], -r["bad"]))
    found = sum(1 for r in rows if r["path"])

    print("# Documents to re-parse\n")
    print("Every ingested document with at least one unusable heading, worst "
          "first. Failure modes are explained with examples in "
          "`heading_quality_report.md`; this file is just the worklist.\n")
    print(f"- documents needing work: **{len(rows)}**")
    print(f"- of those, PDF located on disk: **{found}**")
    print(f"- at or above 30% unusable: **{sum(1 for r in rows if r['pct'] >= 30)}**\n")
    print("| # | Document | PDF | Headings | Unusable | % | Failure modes |")
    print("|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        print(f"| {i} | {r['doc']} | {r['path'] or '_not found_'} | {r['total']} "
              f"| {r['bad']} | {r['pct']:.0f}% | {r['modes']} |")

    missing = [r["doc"] for r in rows if not r["path"]]
    if missing:
        print(f"\n## PDF not found on disk ({len(missing)})\n")
        print("Ingested from a path no longer present, so re-parsing these "
              "needs the original file located first.\n")
        for d in missing:
            print(f"- {d}")


if __name__ == "__main__":
    main()
