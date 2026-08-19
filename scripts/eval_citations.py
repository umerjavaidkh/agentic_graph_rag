#!/usr/bin/env python3
"""
Score document answers on CITATION correctness, separately from answer text.

An answer that is right but cites the wrong page is worse for a public
deployment than one that is slightly wrong and cites correctly, because only
the second lets a reader check it. So citation is scored as a first-class
metric rather than folded into a single pass/fail.

Page and section are scored apart on purpose. Page citations survive bad
heading detection; section citations do not. The gap between the two says
whether a heading-detection fix would change what a reader sees, or only
what the graph calls things -- which is the difference between a bug worth
fixing and a cosmetic one.

Ground truth comes from the graph (Axis-1 page ranges), so no hand-labelling
and no LLM judge.

Usage:
    python scripts/eval_citations.py --suite eval/doc_citation_suite.json [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.auth.roles import Role, UserContext  # noqa: E402
from src.interface.router import ask  # noqa: E402


def cited_pages(sources: list[dict]) -> set[int]:
    """Pages the answer actually pointed the reader at."""
    pages: set[int] = set()
    for s in sources or []:
        raw = s.get("raw") if isinstance(s.get("raw"), dict) else s
        start, end = raw.get("page_start"), raw.get("page_end")
        if isinstance(start, int):
            pages.update(range(start, (end if isinstance(end, int) else start) + 1))
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", required=True)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    cases = json.loads(Path(args.suite).read_text())["cases"]
    if args.limit:
        cases = cases[: args.limit]
    ctx = UserContext(user_id="admin_001", role=Role.ADMIN, department="IT", tenant_id="default")

    hits = {"page": 0, "section": 0, "precision": 0.0}
    for i, c in enumerate(cases, 1):
        result = ask(c["question"], user_context=ctx, retrieval_mode="unstructured") or {}
        answer = (result.get("answer") or "")
        got = cited_pages(result.get("sources") or [])
        want = set(c["expected_pages"])
        # Overlap alone flatters this. Citing six pages of a 52-page document
        # hits a four-page section largely by coverage, so a first pass scored
        # 88% while pages 18, 21, 28 and 32 were being returned for nearly
        # every question regardless of subject. Precision is what a reader
        # experiences: of the pages they were pointed at, how many actually
        # bear on the answer.
        # Cited nodes are Sections, not Pages -- only Section and Chapter are
        # embedded, so semantic retrieval can never return a Page. Comparing a
        # Section citation against Page ground truth scored like-for-unlike and
        # reported 12%. Each cited node is measured against its OWN page span.
        precision = len(got & want) / len(got) if got else 0.0
        page_ok = precision >= 0.5
        section_ok = c["expected_section"].lower() in answer.lower()
        hits["page"] += page_ok
        hits["precision"] += precision
        hits["section"] += section_ok
        print(
            f"[{i}/{len(cases)}] {c['id']:<32} page={'ok ' if page_ok else 'MISS'} "
            f"p={precision:.2f} section={'ok ' if section_ok else 'MISS'} "
            f"cited={sorted(got)[:6]} want={sorted(want)[:4]}",
            file=sys.stderr, flush=True,
        )

    n = len(cases)
    print(f"\n  page citation:    {hits['page']}/{n}  ({100*hits['page']//n if n else 0}%)")
    print(f"  section citation: {hits['section']}/{n}  ({100*hits['section']//n if n else 0}%)")
    print(f"  mean page precision: {hits['precision']/n:.2f}  "
          f"(share of cited pages that bear on the answer)")


if __name__ == "__main__":
    main()
