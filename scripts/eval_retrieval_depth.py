#!/usr/bin/env python3
"""Separate what retrieval got wrong from what generation got wrong.

Every earlier harness scored one thing: did the expected string appear in the
answer. That number cannot distinguish "the passage was never retrieved" from
"it was retrieved and the model ignored it", and it cannot see an answer that
is textually right but taken from the wrong document -- which happened six
times out of twenty-eight on IRS Publication 225, where sibling publications
share vocabulary.

So each question records the node its answer came from, and every run reports
four independent numbers:

    right_doc   retrieval resolved the document the question came from
    recall@k    the source node itself is in the retrieved set
    rank        where it landed, when it was retrieved at all
    answered    the expected span appears in the answer

recall@k and rank are properties of retrieval alone. `answered` is the only
one generation can influence, so a high recall with a low answered rate means
the model is being handed the evidence and not using it -- a different bug
from not finding it, and one the old score reported identically.

No LLM judges anywhere: questions are cloze over spans the document states,
so the expected answer is known before the system is asked.

    python scripts/eval_retrieval_depth.py --docs 25 --questions 4
    python scripts/eval_retrieval_depth.py --docs 25 --questions 4 --cold
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st
import time
import urllib.request

from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:17687"
API = "http://127.0.0.1:8000/query"

_FACT = re.compile(
    r"\b\d[\d,]*\.\d+\b"
    r"|\b\d[\d,]*\s*(?:%|percent|million|billion|thousand)\b"
    r"|\b\d{3}[\d,]*\b"
)
_CITATION = re.compile(r"[\[(]\s*\d[\d,\s;–-]*\s*[\])]")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _is_prose(sent: str) -> bool:
    """Reject extraction artifacts before they become questions.

    A question built over a PDF's margin text is unanswerable, and scoring
    against it measures the parser rather than retrieval.
    """
    words = sent.split()
    if len(words) < 10:
        return False
    if sum(c.isalpha() or c.isspace() for c in sent) / max(len(sent), 1) < 0.75:
        return False
    if sum(1 for w in words if len(w) <= 2 and not w.isdigit()) / len(words) >= 0.3:
        return False
    if re.search(r"\d{4}-\d{3}", sent):          # ORCID / DOI
        return False
    return not re.search(r"\w- \w", sent)        # hyphenated line break


def sample_documents(session, n: int, seed: int) -> list[dict]:
    """Random across the WHOLE corpus, not the most recent slice.

    Sampling by ingest recency correlates with document type -- the last 500
    here are almost all arXiv -- so it measures one family and reads as a
    corpus number.
    """
    rows = session.run(
        "MATCH (r:DocRevision) WHERE r.lifecycle_status = 'ACTIVE' "
        "RETURN r.logical_doc_id AS doc, coalesce(r.title, r.logical_doc_id) AS title"
    )
    docs = [dict(r) for r in rows if r.get("doc")]
    random.Random(seed).shuffle(docs)
    return docs[:n]


def make_questions(session, doc: str, title: str, k: int, cold: bool) -> list[dict]:
    rows = session.run(
        """
        MATCH (n:Section|Page|Chapter)
        WHERE n.logical_doc_id = $doc AND n.lifecycle_status = 'ACTIVE'
          AND size(coalesce(n.search_text, '')) > 300
        RETURN n.id AS node_id, n.search_text AS text, coalesce(n.title, '') AS sec
        ORDER BY coalesce(n.order, 0) LIMIT 60
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
            if len(span) < 2 or span in seen:
                continue
            # A match that starts mid-number. "$19,000" matched "000" because
            # \b fires after the comma, producing "$19, ______" with an
            # expected answer of "000" -- unanswerable, and it would have
            # scored as a retrieval failure.
            before = sent[: m.start()]
            if before.rstrip().endswith(",") and before.rstrip()[:-1].rstrip()[-1:].isdigit():
                continue
            if sent[m.end(): m.end() + 1] == "," and sent[m.end() + 1: m.end() + 2].isdigit():
                continue
            if re.fullmatch(r"(?:19|20)\d{2}", span):        # bare year
                continue
            if m.start() == 0 or re.fullmatch(r"\d+\.\d+", span):  # section number
                continue
            cloze = (sent[: m.start()] + " ______ " + sent[m.end():]).strip()
            if len(cloze) < 50:
                continue
            seen.add(span)
            # cold: the document is NOT named, so retrieval has to place the
            # question from its content alone. That is the harder condition
            # and the one a real user produces.
            q = (
                f'Fill in the blank with the exact value from the source: "{cloze}"'
                if cold else
                f'In the document "{title}", fill in the blank with the exact '
                f'value from the text: "{cloze}"'
            )
            out.append({"question": q, "expected": span,
                        "node_id": r["node_id"], "doc": doc})
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
    except Exception as e:
        return {"_elapsed": time.perf_counter() - t0, "_error": type(e).__name__}
    d["_elapsed"] = time.perf_counter() - t0
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=25)
    ap.add_argument("--questions", type=int, default=4)
    ap.add_argument("--seed", type=int, default=20260824)
    ap.add_argument("--cold", action="store_true",
                    help="do not name the document in the question")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    driver = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", "password123"))
    with driver.session() as s:
        plans = []
        for d in sample_documents(s, args.docs, args.seed):
            qs = make_questions(s, d["doc"], d["title"], args.questions, args.cold)
            if qs:
                plans.append(qs)

    rows = []
    # Thread ids must contain no dots or spaces: the scoper silently collapses
    # those to a single shared "default" thread, which merges unrelated
    # conversations and lets one question's document leak into the next.
    for i, qs in enumerate(plans):
        for j, q in enumerate(qs):
            r = ask(q["question"], thread=f"depth{i}x{j}")
            ans = r.get("answer") or ""
            src_ids = [s.get("id") for s in (r.get("sources") or []) if isinstance(s, dict)]
            rank = src_ids.index(q["node_id"]) + 1 if q["node_id"] in src_ids else None
            rows.append({
                "doc": q["doc"], "expected": q["expected"],
                "right_doc": r.get("document_id") == q["doc"],
                "recall": rank is not None,
                "rank": rank,
                "answered": bool(q["expected"]) and q["expected"] in ans,
                "s": round(r.get("_elapsed", 0), 1),
                "error": r.get("_error"),
            })
            print(f"  {q['doc'][:26]:<28}{rows[-1]['s']:>5}s "
                  f"doc={'Y' if rows[-1]['right_doc'] else 'n'} "
                  f"recall={'Y' if rows[-1]['recall'] else 'n'} "
                  f"rank={rank or '-'} ans={'Y' if rows[-1]['answered'] else 'n'}", flush=True)

    n = len(rows) or 1
    rd = sum(r["right_doc"] for r in rows)
    rc = sum(r["recall"] for r in rows)
    an = sum(r["answered"] for r in rows)
    ranks = [r["rank"] for r in rows if r["rank"]]
    # the diagnosis that a single score cannot make
    got_but_missed = sum(1 for r in rows if r["recall"] and not r["answered"])
    never_had = sum(1 for r in rows if not r["recall"] and not r["answered"])
    print(f"\n  questions            {n}")
    print(f"  right document       {rd}/{n} ({rd/n*100:.0f}%)")
    print(f"  recall@k (src node)  {rc}/{n} ({rc/n*100:.0f}%)")
    print(f"  answered correctly   {an}/{n} ({an/n*100:.0f}%)")
    if ranks:
        print(f"  rank of source node  median {st.median(ranks):.0f}  best 1  worst {max(ranks)}")
    print(f"\n  retrieved but not used   {got_but_missed}   <- generation lost it")
    print(f"  never retrieved          {never_had}   <- retrieval lost it")
    print(f"  median latency       {st.median([r['s'] for r in rows]):.1f}s")
    if args.out:
        json.dump(rows, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
