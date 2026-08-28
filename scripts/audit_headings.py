"""audit_headings.py — which documents have headings a person would not recognise.

Asked to list the chapter headings of a recent arXiv paper, retrieval
answered "this document does not cover chapter headings", and it was
right to: the headings stored for that document were

    'XXli,t, (3) LTM ='
    '1000 2000 3000 4000 5000 6000 7000 8000 Toke'

chart axis tick labels and math fragments, detected as headings at
ingest. No retrieval change can fix that, and the eval that grades
against them is grading against corrupt ground truth.

This ranks every ingested document by how much of its heading structure
is unusable, so the parser work has somewhere to start and a way to tell
whether it worked. Writes markdown to stdout.

    docker compose exec -T app python - < scripts/audit_headings.py \\
        > eval/heading_quality_report.md
"""
from __future__ import annotations

import re
from collections import defaultdict

from src.shared.neo4j.driver import get_neo4j_driver

_WORD = re.compile(r"[A-Za-z][A-Za-z'-]{2,}")
# A heading that is one long sentence is a body line that was promoted.
_MAX_HEADING_CHARS = 120
# Equation debris: an "=" or a math symbol never appears in a real heading,
# but is what a promoted formula line looks like ("XXli,t, (3) LTM =").
_MATH = re.compile(r"[=\u2211\u222b\u2264\u2265\u00b1\u2113\u03bc\u03c3\u2192\u2208]|\\\\[a-z]+\{")


def reasons(title: str) -> list[str]:
    """Why this string is not a heading. Empty means it looks like one.

    Each test names a failure mode seen in the corpus rather than a
    general notion of tidiness, so a count here points at a parser
    behaviour to go and fix.

    Deliberately NOT tested: heading length in words. A first pass counted
    anything under two words unusable, which flagged "Introduction",
    "Abstract" and "Contents" -- 90% of all hits -- and would have sent the
    parser work after 345 documents that are largely fine. Single-word
    headings are the norm, not a defect.
    """
    t = (title or "").strip()
    out: list[str] = []
    if t.startswith("|"):
        out.append("table-grid row")
    if len(t) > _MAX_HEADING_CHARS:
        out.append("body sentence")
    if t.rstrip().endswith("-"):
        out.append("line-break fragment")
    if t.rstrip().endswith(","):
        out.append("mid-sentence fragment")
    digits = sum(c.isdigit() for c in t)
    if t and digits > len(t) * 0.3:
        out.append("mostly digits")
    # Math symbols alone are not enough: "4.1 Global Optimality of pg =
    # pdata" is a real heading that happens to contain "=". Equation debris
    # is short on actual words, so both conditions are required.
    if _MATH.search(t) and len(_WORD.findall(t)) < 4:
        out.append("equation fragment")
    if t and not _WORD.search(t):
        out.append("no words at all")
    return out


def main() -> None:
    per_doc: dict[str, list[str]] = defaultdict(list)
    source: dict[str, str] = {}
    with get_neo4j_driver().session() as s:
        rows = s.run(
            """
            MATCH (n:Chapter|Section)
            WHERE n.lifecycle_status = 'ACTIVE'
              AND trim(coalesce(n.title, '')) <> ''
            RETURN n.logical_doc_id AS doc, trim(n.title) AS t
            """
        )
        for r in rows:
            if r["doc"]:
                per_doc[r["doc"]].append(r["t"])
        for r in s.run(
            "MATCH (r:DocRevision) WHERE r.lifecycle_status = 'ACTIVE' "
            "RETURN r.logical_doc_id AS doc, coalesce(r.source_filename, '') AS f"
        ):
            if r["doc"]:
                source[r["doc"]] = r["f"]

    stats = []
    reason_totals: dict[str, int] = defaultdict(int)
    reason_examples: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for doc, titles in per_doc.items():
        bad = [t for t in titles if reasons(t)]
        for t in bad:
            for why in reasons(t):
                reason_totals[why] += 1
                if len(reason_examples[why]) < 8:
                    reason_examples[why].append((doc, t))
        stats.append({
            "doc": doc, "total": len(titles), "bad": len(bad),
            "pct": len(bad) / max(len(titles), 1) * 100,
            "sample": bad[:3],
        })

    total_h = sum(s["total"] for s in stats)
    total_b = sum(s["bad"] for s in stats)
    stats.sort(key=lambda x: (-x["pct"], -x["bad"]))
    worst = [s for s in stats if s["pct"] >= 30 and s["total"] >= 3]

    print("# Documents with unusable headings\n")
    print("Every ACTIVE Chapter and Section title in the corpus, checked "
          "against five failure modes seen in the data. A title is counted "
          "unusable if it matches any of them.\n")
    print(f"- documents scanned: **{len(stats)}**")
    print(f"- headings scanned: **{total_h:,}**")
    print(f"- unusable: **{total_b:,} ({total_b/max(total_h,1)*100:.0f}%)**")
    print(f"- documents at or above 30% unusable: **{len(worst)}**\n")

    print("## Why they fail\n")
    print("| Failure mode | Headings |")
    print("|---|---|")
    for why, n in sorted(reason_totals.items(), key=lambda kv: -kv[1]):
        print(f"| {why} | {n:,} |")
    print("\n_A heading can fail more than one test, so these sum to more "
          "than the total._\n")

    print("## Examples, by failure mode\n")
    print("What the parser produced, with the document it came from. These "
          "are the concrete cases to work against.\n")
    for why, _ in sorted(reason_totals.items(), key=lambda kv: -kv[1]):
        print(f"### {why}\n")
        print("| Document | Stored heading |")
        print("|---|---|")
        for doc, t in reason_examples[why]:
            shown = t[:90].replace("|", "\\|").replace("`", "'")
            print(f"| {doc} | `{shown}` |")
        print()

    print("""## What the failures look like

Two patterns account for most of the debris, and both are recoverable in
the parser rather than downstream.

**Rotated text read as a horizontal run.** Chart axis labels are drawn
vertically, and come out as character runs:

    'y ca 82 ru cc A v 80 e D I L N 78 M'   <- "Accuracy" / "MNLI Dev"
    ')s µ( yr 105 e u q e p it n'           <- "query time (µs)"
    '0.59 E - n e k oT0.4'

Each is one text run per glyph column. A rotation check on the span, or
dropping spans whose glyphs share an x-coordinate, removes this whole
class -- it is the largest single source of "no words at all" and
"mostly digits".

**Bibliography entries promoted to headings.** Reference lists are
short lines in a distinct style, which reads like a heading run:

    'Adepoju, S., David, S.: An Intelligent API Framework for Real-time'
    'Education, 2023, 54-60, https://doi.org/10.1145/3587102.3588794.'

Anything after a "References"/"Bibliography" heading is a citation, not a
section, and can be excluded by position alone.

The remaining modes -- line-break fragments, mid-sentence fragments and
body sentences -- are all one thing: a body line promoted because it was
short or styled like a heading. A heading does not end in a hyphen or a
comma, which is what those two tests key on.

""")
    print("## Worst documents\n")
    print("| Document | Source file | Headings | Unusable | % | Example |")
    print("|---|---|---|---|---|---|")
    for st in worst[:150]:
        ex = (st["sample"][0] if st["sample"] else "")[:55]
        ex = ex.replace("|", "\\|").replace("`", "'")
        src = (source.get(st["doc"], "") or "")[:34]
        print(f"| {st['doc']} | {src} | {st['total']} | {st['bad']} "
              f"| {st['pct']:.0f}% | `{ex}` |")

    clean = [s for s in stats if s["pct"] == 0 and s["total"] >= 3]
    print(f"\n## Clean\n\n**{len(clean)}** documents have no unusable heading "
          f"(of {len([s for s in stats if s['total'] >= 3])} with three or more).\n")


if __name__ == "__main__":
    main()
