"""qa_log_sample.py — sample documents, ask deterministic questions, score them.

Reproduces the methodology recorded in eval/corpus500_qa_log.md: questions are
generated from each document's own text as a cloze over a stated fact, so the
expected answer is known without an LLM ever being asked to invent one. Only
the answering costs money; question generation and scoring are free and
deterministic, which is what makes the run repeatable.

An answer counts correct when the expected span appears in it. `right doc`
counts how often retrieval resolved the document the question came from --
tracked separately because a wrong document with a fluent answer is the
failure mode that reads as success.

Usage:
    python scripts/qa_log_sample.py --docs 20 --questions 5 --seed 20260823
"""
from __future__ import annotations

import argparse
import json
import random
import re
import time
import urllib.request
from statistics import mean

from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:17687"
API = "http://127.0.0.1:8000/query"

# A fact worth asking about: a percentage, a decimal, a magnitude word, or a
# number of at least three digits. Bare small integers are excluded because
# they are everywhere -- a hit on "3" proves nothing about retrieval.
_FACT = re.compile(
    r"\b\d[\d,]*\.\d+\b"
    r"|\b\d[\d,]*\s*(?:%|percent|million|billion|thousand)\b"
    r"|\b\d{3}[\d,]*\b"
)
# Citation markers ("[10]", "(2021)") state nothing about the document's
# subject, and a PDF's margin text arrives as garbage like "6 2 :v iX r".
_CITATION = re.compile(r"[\[(]\s*\d[\d,\s;–-]*\s*[\])]")


def _is_prose(sent: str) -> bool:
    """Reject extraction artifacts before they become questions.

    arXiv sidebars, running headers and table gutters extract as sequences
    of stray characters. A question built over one is unanswerable, and
    scoring against it measures the parser, not retrieval.
    """
    words = sent.split()
    if len(words) < 10:
        return False
    alpha = sum(c.isalpha() or c.isspace() for c in sent) / max(len(sent), 1)
    if alpha < 0.75:
        return False
    stray = sum(1 for w in words if len(w) <= 2 and not w.isdigit())
    if stray / len(words) >= 0.3:
        return False
    # An ORCID or DOI is not a fact about the subject, and it reads as prose
    # because it sits among author affiliations.
    if re.search(r"\d{4}-\d{3}", sent):
        return False
    # "frame- est" -- a hyphenated line break the parser did not rejoin.
    # The sentence is already damaged; a cloze over it is unanswerable.
    return not re.search(r"\w- \w", sent)


_SENT = re.compile(r"(?<=[.!?])\s+")


def pick_documents(session, pool: int, n: int, seed: int) -> list[dict]:
    rows = session.run(
        """
        MATCH (r:DocRevision) WHERE r.ingested_at IS NOT NULL
        RETURN r.logical_doc_id AS doc, coalesce(r.title, r.logical_doc_id) AS title
        ORDER BY r.ingested_at DESC LIMIT $pool
        """,
        pool=pool,
    )
    docs = [dict(r) for r in rows if r.get("doc")]
    random.Random(seed).shuffle(docs)
    return docs[:n]


def make_questions(session, doc: str, title: str, k: int) -> list[dict]:
    """Cloze questions over sentences that state a specific fact."""
    rows = session.run(
        """
        MATCH (n:Section|Page|Chapter)
        WHERE n.logical_doc_id = $doc AND n.lifecycle_status = 'ACTIVE'
          AND size(coalesce(n.search_text, '')) > 300
        RETURN n.search_text AS text, coalesce(n.title, '') AS sec
        ORDER BY coalesce(n.order, 0)
        LIMIT 60
        """,
        doc=doc,
    )
    out: list[dict] = []
    seen: set[str] = set()
    for r in rows:
        for sent in _SENT.split(r["text"] or ""):
            sent = " ".join(sent.split())
            if not (60 <= len(sent) <= 240) or not _is_prose(sent):
                continue
            if _CITATION.search(sent):
                continue
            m = _FACT.search(sent)
            if not m:
                continue
            span = m.group(0).strip()
            if not any(ch.isdigit() for ch in span) or len(span) < 2:
                continue
            # A bare year is almost always a citation date or a boilerplate
            # copyright line, not a fact the document asserts.
            if re.fullmatch(r"(?:19|20)\d{2}", span):
                continue
            # A section number, not a fact. Every failure in the first run of
            # this harness was one of these: "5.3 Exact consensus", "2.2 Tool
            # Box", "5.2 Participant Evaluation" -- a heading that ran into
            # the following prose, so the cloze asked the model to infer a
            # section number from context. Unanswerable by construction, and
            # it scored as a retrieval failure, which is worse than useless.
            if m.start() == 0 or re.fullmatch(r"\d+\.\d+", span):
                continue
            if span in seen:
                continue
            cloze = (sent[: m.start()] + " ______ " + sent[m.end():]).strip()
            if len(cloze) < 50:
                continue
            seen.add(span)
            out.append({
                "question": (
                    f'In the document "{title}", fill in the blank with the exact '
                    f'value from the text: "{cloze}"'
                ),
                "expected": span,
                "section": r["sec"],
            })
            if len(out) >= k:
                return out
    return out


def ask(question: str, thread: str, timeout: int = 180) -> dict:
    body = json.dumps({
        "question": question, "user_id": "admin_001", "role": "admin",
        "tenant_id": "", "thread_id": thread,
    }).encode()
    req = urllib.request.Request(API, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
        d["_elapsed"] = time.perf_counter() - t0
        return d
    except Exception as e:
        return {"_elapsed": time.perf_counter() - t0, "_error": f"{type(e).__name__}"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=20)
    ap.add_argument("--questions", type=int, default=5)
    ap.add_argument("--pool", type=int, default=500)
    ap.add_argument("--seed", type=int, default=20260823)
    ap.add_argument("--out", default="eval/corpus500_qa_log.md")
    args = ap.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", "password123"))
    with driver.session() as s:
        docs = pick_documents(s, args.pool, args.docs, args.seed)
        plans = []
        for d in docs:
            qs = make_questions(s, d["doc"], d["title"], args.questions)
            nodes = s.run(
                "MATCH (n:Section|Page|Chapter|Region) WHERE n.logical_doc_id=$d "
                "RETURN count(n) AS c", d=d["doc"]).single()["c"]
            if qs:
                plans.append({**d, "questions": qs, "nodes": nodes})

    rows, all_times = [], []
    for i, p in enumerate(plans, 1):
        hits = right = 0
        times: list[float] = []
        for j, q in enumerate(p["questions"], 1):
            r = ask(q["question"], thread=f"qa-{p['doc']}-{j}")
            times.append(r["_elapsed"])
            ans = (r.get("answer") or "")
            if q["expected"] and q["expected"] in ans:
                hits += 1
            if r.get("document_id") == p["doc"]:
                right += 1
            print(f"  [{i}/{len(plans)}] {p['doc']} q{j} {r['_elapsed']:.0f}s "
                  f"exp={q['expected']!r} hit={q['expected'] in ans} "
                  f"doc={r.get('document_id')}", flush=True)
        all_times += times
        rows.append({
            "doc": p["title"], "score": f"{hits}/{len(p['questions'])}",
            "nodes": p["nodes"], "avg": round(mean(times)),
            "right": f"{right}/{len(p['questions'])}",
            "_h": hits, "_n": len(p["questions"]), "_r": right,
        })

    rows.sort(key=lambda r: (-r["_h"], r["doc"]))
    tot_h = sum(r["_h"] for r in rows); tot_n = sum(r["_n"] for r in rows)
    tot_r = sum(r["_r"] for r in rows)
    stamp = time.strftime("%Y-%m-%d %H:%M")

    md = [
        f"\n\n## Run {stamp} — strategy `graph_rag_vector_first`\n",
        f"{len(rows)} documents sampled (seed {args.seed}) from the most recently "
        f"ingested {args.pool}, {args.questions} questions each.\n",
        f"**Answer accuracy {tot_h}/{tot_n} ({tot_h/max(tot_n,1)*100:.0f}%) · "
        f"right document {tot_r}/{tot_n} ({tot_r/max(tot_n,1)*100:.0f}%) · "
        f"median {sorted(all_times)[len(all_times)//2]:.0f}s**\n",
        "| Document | Score | Nodes | Avg s | Right doc |",
        "|---|---|---|---|---|",
    ]
    md += [f"| {r['doc']} | {r['score']} | {r['nodes']} | {r['avg']} | {r['right']} |"
           for r in rows]
    with open(args.out, "a") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md[:5]))
    print(f"appended to {args.out}")


if __name__ == "__main__":
    main()
