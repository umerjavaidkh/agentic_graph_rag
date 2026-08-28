#!/usr/bin/env python3
"""Multi-turn retrieval: does the thread help, and does it get out of the way?

Every earlier harness gave each question its own thread, which is why none of
them caught a document named in turn 2 losing to the document from turn 1.
A real user asks in a conversation, so the conversation is the unit of test.

Each document gets one thread with three turns, chosen to pull in opposite
directions:

    T1  names the document              -> must resolve it
    T2  names nothing, follow-up        -> must STAY (the thread is the only signal)
    T3  names a DIFFERENT document      -> must SWITCH (the name outranks the thread)

T2 and T3 fail in opposite ways, so a system cannot pass both by preferring
one source of truth. Ignoring the thread breaks T2; obeying it blindly breaks
T3 -- which is the bug this suite was written to keep fixed.

Questions are cloze over spans the documents state, so answers are known
without an LLM judge. Each records its source node, so a miss is attributable
to retrieval or to generation.

    python scripts/eval_multiturn.py --docs 10 --seed 5150 --plan-only
    python scripts/eval_multiturn.py --docs 10 --seed 5150
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request

from neo4j import GraphDatabase

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.eval_retrieval_depth import (  # noqa: E402
    NEO4J_URI, API, make_questions, sample_documents,
)


def ask(question: str, thread: str, timeout: int = 240) -> dict:
    body = json.dumps({
        "question": question, "user_id": "admin_001", "role": "admin",
        "tenant_id": "", "thread_id": thread,
    }).encode()
    req = urllib.request.Request(API, body, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        d = json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception as e:
        return {"_elapsed": time.perf_counter() - t0, "_error": type(e).__name__}
    d["_elapsed"] = time.perf_counter() - t0
    return d


def build(session, docs: list[dict]) -> list[dict]:
    """One conversation per document; T3 borrows the next document's question."""
    prepared = []
    for d in docs:
        qs = make_questions(session, d["doc"], d["title"], 2, cold=False)
        cold = make_questions(session, d["doc"], d["title"], 2, cold=True)
        if len(qs) >= 1 and len(cold) >= 2:
            prepared.append({**d, "named": qs[0], "followup": cold[1]})
    convos = []
    for i, p in enumerate(prepared):
        other = prepared[(i + 1) % len(prepared)]
        if other["doc"] == p["doc"]:
            continue
        convos.append({
            "doc": p["doc"], "title": p["title"], "other": other["doc"],
            "turns": [
                ("T1 names it",        p["named"],     p["doc"]),
                ("T2 follow-up",       p["followup"],  p["doc"]),
                ("T3 names another",   other["named"], other["doc"]),
            ],
        })
    return convos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=5150)
    ap.add_argument("--plan-only", action="store_true")
    args = ap.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", "password123"))
    with driver.session() as s:
        convos = build(s, sample_documents(s, args.docs, args.seed))

    if args.plan_only:
        for c in convos:
            print(f"\n=== {c['title']}   ({c['doc']})")
            for label, q, expect_doc in c["turns"]:
                tag = "same doc" if expect_doc == c["doc"] else f"must switch to {expect_doc}"
                print(f"  [{label}] expect {q['expected']!r}  ({tag})")
                print(f"    {q['question'][:190]}")
        print(f"\n  {len(convos)} conversations x 3 turns = {len(convos)*3} questions")
        return

    rows = []
    for i, c in enumerate(convos):
        thread = f"mt{i}"          # no dots or spaces: those collapse to one shared thread
        for label, q, expect_doc in c["turns"]:
            r = ask(q["question"], thread)
            ans = r.get("answer") or ""
            ids = [x.get("id") for x in (r.get("sources") or []) if isinstance(x, dict)]
            rows.append({
                "doc": c["doc"], "turn": label, "expect_doc": expect_doc,
                "got_doc": r.get("document_id"),
                "right_doc": r.get("document_id") == expect_doc,
                "recall": q["node_id"] in ids,
                "answered": bool(q["expected"]) and q["expected"] in ans,
                "s": round(r.get("_elapsed", 0), 1),
            })
            x = rows[-1]
            print(f"  {c['title'][:22]:<24}{label:<18}{x['s']:>5}s "
                  f"doc={'Y' if x['right_doc'] else 'n'} recall={'Y' if x['recall'] else 'n'} "
                  f"ans={'Y' if x['answered'] else 'n'}", flush=True)

    def rate(turn):
        sel = [r for r in rows if r["turn"].startswith(turn)]
        return sum(r["right_doc"] for r in sel), len(sel)
    print()
    for t, meaning in (("T1", "names it        "),
                       ("T2", "follow-up, stay "),
                       ("T3", "names another   ")):
        ok, n = rate(t)
        print(f"  {t} {meaning} right document {ok}/{n}")
    n = len(rows) or 1
    print(f"\n  overall right document {sum(r['right_doc'] for r in rows)}/{n}")
    print(f"  recall@k               {sum(r['recall'] for r in rows)}/{n}")
    print(f"  answered correctly     {sum(r['answered'] for r in rows)}/{n}")
    json.dump(rows, open("/tmp/multiturn.json", "w"), indent=1)


if __name__ == "__main__":
    main()
