"""dump_eval_questions.py — write out the questions a shape run actually asks.

eval_shapes.py builds its questions from each document's own text and then
records only the scores, so a run could be reproduced but not read. This
regenerates the exact question set for a chosen slice -- same builder, no
API calls, no model spend -- so the questions can be reviewed, asked by
hand in the UI, or checked against the answers they produced.

    docker compose exec -T app python - < scripts/dump_eval_questions.py \\
        > eval/QUESTIONS_ASKED.md
"""
from __future__ import annotations

import sys

sys.path.insert(0, "scripts")

from eval_shapes import build  # noqa: E402
from src.shared.neo4j.driver import get_neo4j_driver  # noqa: E402

# The two slices measured so far, named the way the results refer to them.
SLICES = [
    ("Last 5 of 1,001 (most recently ingested)", "SKIP0", None),
    ("Second-last 5 of 1,001 (mixed parse quality)", "SKIP", None),
    ("5 cleanly-parsed documents", "IDS", [
        "doc_arxiv_2608_16178", "doc_arxiv_2608_17810", "doc_arxiv_2608_18586",
        "doc_arxiv_2608_15888", "doc_irs_p501_dependents",
    ]),
]

MODE_NOTE = {
    "EXACT": "expected span must appear in the answer",
    "COUNT": "expected count must appear",
    "SET": "at least 80% of the expected items must appear",
    "REFUSE": "must decline",
    "RETRIEVAL": "not machine-checkable; scored on right-document and precision only",
}


def main() -> None:
    print("# Questions asked, by document\n")
    print("Regenerated with the same builder the eval uses, so these are the "
          "exact strings that were sent. Questions come from each document's "
          "own text, which is why the cloze wording looks mechanical -- the "
          "expected answer has to be verifiable without a model.\n")
    print("`mode` says how the answer was graded:\n")
    for m, note in MODE_NOTE.items():
        print(f"- **{m}** — {note}")
    print()

    with get_neo4j_driver().session() as s:
        for label, kind, ids in SLICES:
            if kind == "IDS":
                docs = [dict(r) for r in s.run(
                    "MATCH (r:DocRevision) WHERE r.logical_doc_id IN $ids "
                    "AND r.lifecycle_status = 'ACTIVE' "
                    "RETURN DISTINCT r.logical_doc_id AS doc, "
                    "coalesce(r.title, r.logical_doc_id) AS title", ids=ids)]
            else:
                docs = [dict(r) for r in s.run(
                    "MATCH (r:DocRevision) WHERE r.ingested_at IS NOT NULL "
                    "RETURN r.logical_doc_id AS doc, "
                    "coalesce(r.title, r.logical_doc_id) AS title "
                    "ORDER BY r.ingested_at DESC SKIP $k LIMIT 5",
                    k=0 if kind == "SKIP0" else 5)]

            print(f"\n---\n\n# {label}\n")
            for d in docs:
                Q = build(s, d["doc"], d["title"])
                print(f"## {d['doc']}\n")
                print(f"_{len(Q)} questions_\n")
                print("| # | Shape | Mode | Question |")
                print("|---|---|---|---|")
                for i, q in enumerate(Q, 1):
                    text = q["q"].replace("|", "\\|").replace("\n", " ")
                    if len(text) > 300:
                        text = text[:300] + " …"
                    print(f"| {i} | {q['shape']} | {q['mode']} | {text} |")
                print()

    print("""
---

## Two things visible in the questions themselves

The generator builds questions out of what ingestion stored, so where
ingestion went wrong the question inherits it. Both cases below are real
rows above, not hypotheticals, and both explain scores reported elsewhere.

**Table questions quoting a pipe grid.** "What does the table
`| Tier | No<br>des | Server |` show?" -- that is the stored title, and it
is the grid's first row. Fixed at ingest by #117, but only for documents
ingested after it; everything listed here predates that.

**Section questions naming something that is not a section.** "What does
the section titled 'Keywords: Earth observation data infrastructure; ...'
cover?" -- a keyword blob promoted to a heading. This is the same defect
the heading audit counts, and it is why the enumerative shape scores
against corrupt ground truth on the mixed-quality slices.

Neither is a retrieval fault. Both are why the cleanly-parsed slice
scores 100% on machine-checkable shapes while the mixed slice scores 73%.
""")


if __name__ == "__main__":
    main()
