#!/usr/bin/env python3
"""Twenty query shapes over the five most recently ingested documents.

Ground truth comes from the graph, never from a model. Each shape declares
how it is scored, and shapes whose answers cannot be verified deterministically
are scored on retrieval only rather than given a fabricated correctness number:

    EXACT      a span the document states must appear in the answer
    SET        every member of a graph-derived set must appear (enumerative)
    COUNT      a number computed from the graph must appear
    REFUSE     the correct behaviour is to decline
    RETRIEVAL  answer correctness not machine-checkable; scored on
               right-document / recall / precision only

Metrics per question:
    right_doc    retrieval resolved the document the question came from
    recall       the source node is among the retrieved chunks (where known)
    precision    share of retrieved chunks that belong to the right document
"""
from __future__ import annotations

import argparse, json, re, statistics as st, sys, time, urllib.request
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from collections import defaultdict
from neo4j import GraphDatabase

NEO4J_URI = "bolt://localhost:17687"
API = "http://127.0.0.1:8000/query"
# Refusal phrases now come from the language profiles rather than living
# here as English. The list below was English-only, so once the system
# started refusing in the language it was asked in, a CORRECT Arabic
# refusal matched nothing and the unanswerable shape scored 0/5 -- an
# exam graded against the wrong key, and indistinguishable in the summary
# from a system that had started fabricating.
#
# Rebound in main() once --language is known; the module-level default
# keeps the English behaviour for anything importing this directly.
from src.shared.language import refusal_markers as _refusal_markers  # noqa: E402

REFUSAL = _refusal_markers("en")


# Set from --language. Omitted from the payload entirely when empty, so an
# English run sends the identical request it always did and stays
# comparable to the recorded baseline.
LANGUAGE = ""


def ask(q, thread, timeout=240):
    body = {"question": q, "user_id": "admin_001", "role": "admin",
            "tenant_id": "", "thread_id": thread}
    if LANGUAGE:
        # Without this the request takes the deployment default and the
        # scoped corpus is unreachable -- the eval would report 0% for a
        # reason that has nothing to do with retrieval.
        body["language"] = LANGUAGE
    b = json.dumps(body).encode()
    r = urllib.request.Request(API, b, {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        d = json.load(urllib.request.urlopen(r, timeout=timeout))
    except Exception as e:
        return {"_elapsed": time.perf_counter() - t0, "_error": type(e).__name__}
    d["_elapsed"] = time.perf_counter() - t0
    return d


def facts(session, doc):
    """A stated numeric fact and the node that states it."""
    rows = session.run(
        "MATCH (n:Section|Page) WHERE n.logical_doc_id=$d AND size(coalesce(n.search_text,''))>400 "
        "RETURN n.id AS id, n.search_text AS t ORDER BY coalesce(n.order,0) LIMIT 40", d=doc)
    pat = re.compile(r"\b\d[\d,]*\.\d+\b|\b\d[\d,]*\s*(?:%|percent)\b|\b\d{3}[\d,]*\b")
    out = []
    for r in rows:
        for s in re.split(r"(?<=[.!?])\s+", r["t"] or ""):
            s = " ".join(s.split())
            if not (70 <= len(s) <= 220):
                continue
            if len(s.split()) < 12 or sum(c.isalpha() or c.isspace() for c in s)/len(s) < 0.78:
                continue
            if re.search(r"\w- \w|\d{4}-\d{3}|[\[(]\s*\d", s):
                continue
            m = pat.search(s)
            if not m or m.start() == 0:
                continue
            before = s[:m.start()].rstrip()
            if before.endswith(",") or before.endswith(" ") and before.rstrip()[-1:].isdigit():
                continue
            out.append((s, m.group(0).strip(), r["id"]))
            if len(out) >= 4:
                return out
    return out


def build(session, doc, title):
    """Twenty shapes, each with graph-derived ground truth or an explicit RETRIEVAL score."""
    d = doc
    chapters = [r["t"] for r in session.run(
        "MATCH (n:Chapter) WHERE n.logical_doc_id=$d AND trim(coalesce(n.title,''))<>'' "
        "RETURN trim(n.title) AS t ORDER BY coalesce(n.order,0)", d=d)]
    tables = [dict(r) for r in session.run(
        "MATCH (n:Region) WHERE n.logical_doc_id=$d AND n.region_kind='table' "
        "AND trim(coalesce(n.title,''))<>'' RETURN n.id AS id, trim(n.title) AS t LIMIT 3", d=d)]
    figs = [dict(r) for r in session.run(
        "MATCH (n:Region) WHERE n.logical_doc_id=$d AND n.region_kind='figure' "
        "AND trim(coalesce(n.title,''))<>'' RETURN n.id AS id, trim(n.title) AS t LIMIT 2", d=d)]
    fs = facts(session, d)
    Q = []
    def add(shape, q, mode, expect=None, node=None):
        Q.append({"shape": shape, "q": q, "mode": mode, "expect": expect,
                  "node": node, "doc": d})

    if fs:
        s, span, nid = fs[0]
        add("1 fact", f'In "{title}", what value completes this statement: "{s.replace(span,"____",1)}"?',
            "EXACT", span, nid)
    if len(fs) > 1:
        s, span, nid = fs[1]
        add("18 numeric", f'In "{title}", what is the exact figure where this sentence reads "____": "{s.replace(span,"____",1)}"?',
            "EXACT", span, nid)
    if chapters:
        add("4 structural", f'In "{title}", what does the section titled "{chapters[0]}" cover?',
            "RETRIEVAL")
        add("5 enumerative", f'List every chapter heading in "{title}".', "SET", chapters[:8])
        add("7 aggregation", f'How many chapters does "{title}" contain?', "COUNT", str(len(chapters)))
        add("13 summarization", f'Summarize the chapter "{chapters[-1]}" from "{title}".', "RETRIEVAL")
    if tables:
        add("19 table", f'What does the table "{tables[0]["t"][:60]}" in "{title}" show?',
            "RETRIEVAL", None, tables[0]["id"])
    if figs:
        add("20 figure", f'What does the figure "{figs[0]["t"][:60]}" in "{title}" show?',
            "RETRIEVAL", None, figs[0]["id"])
    add("2 definition",   f'In "{title}", define the central technical term the paper introduces.', "RETRIEVAL")
    add("3 entity",       f'Who are the authors of "{title}"?', "RETRIEVAL")
    add("6 filtering",    f'Which parts of "{title}" discuss evaluation or results?', "RETRIEVAL")
    add("8 comparison",   f'In "{title}", how does the proposed approach differ from prior work?', "RETRIEVAL")
    add("10 multihop",    f'In "{title}", which method is evaluated on which dataset?', "RETRIEVAL")
    add("11 causal",      f'Why was the work in "{title}" undertaken?', "RETRIEVAL")
    add("12 thematic",    f'What does "{title}" discuss overall?', "RETRIEVAL")
    add("14 procedural",  f'What method or procedure does "{title}" describe?', "RETRIEVAL")
    add("15 requirements",f'What does "{title}" say is required to reproduce its results?', "RETRIEVAL")
    add("23 citation",    f'Where in "{title}" is the evaluation setup described?', "RETRIEVAL")
    add("24 verification",f'Does "{title}" actually report experimental results, or only propose a method?', "RETRIEVAL")
    add("28 unanswerable",f'In "{title}", what were the quarterly dividend payments to shareholders in 1987?', "REFUSE")
    add("27 ambiguous",   'What is the value?', "REFUSE")
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=int, default=5)
    ap.add_argument("--language", default="",
                    help="Scope every question to one language (e.g. ar). "
                         "Omitted from the request when unset.")
    # How many of the most-recent documents to step over first. --skip 5
    # with --docs 5 measures the second-last five, so a rerun is a fresh
    # sample rather than the batch already reported on.
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--out", default="")
    # Explicit document ids, comma separated. Lets a run target a chosen
    # slice -- cleanly-parsed documents, say -- so the score reflects
    # retrieval rather than how badly the headings came out.
    ap.add_argument("--ids", default="")
    args = ap.parse_args()
    global LANGUAGE, REFUSAL
    LANGUAGE = (args.language or "").strip().lower()
    REFUSAL = _refusal_markers(LANGUAGE or None)
    drv = GraphDatabase.driver(NEO4J_URI, auth=("neo4j", "password123"))
    with drv.session() as s:
        if args.ids:
            wanted = [x.strip() for x in args.ids.split(",") if x.strip()]
            docs = [dict(r) for r in s.run(
                "MATCH (r:DocRevision) WHERE r.logical_doc_id IN $ids "
                "AND r.lifecycle_status = 'ACTIVE' "
                "RETURN DISTINCT r.logical_doc_id AS doc, "
                "coalesce(r.title,r.logical_doc_id) AS title", ids=wanted)]
        else:
            docs = [dict(r) for r in s.run(
                "MATCH (r:DocRevision) WHERE r.ingested_at IS NOT NULL "
                "RETURN r.logical_doc_id AS doc, coalesce(r.title,r.logical_doc_id) AS title "
                "ORDER BY r.ingested_at DESC SKIP $k LIMIT $n",
                k=args.skip, n=args.docs)]
        plans = [(d, build(s, d["doc"], d["title"])) for d in docs]

    rows = []
    for di, (d, Q) in enumerate(plans):
        for qi, q in enumerate(Q):
            r = ask(q["q"], f"sh{di}x{qi}")
            ans = r.get("answer") or ""
            low = ans.lower()
            ids = [x.get("id", "") for x in (r.get("sources") or []) if isinstance(x, dict)]
            # Graph-derived chunks are scoped to the resolved document by
            # construction, but their ids are markers ("graph_outline",
            # "graph_count") rather than "<doc>::<node>". Matching on the id
            # prefix alone scored them as retrieved from the wrong document
            # and dropped precision 0.95 -> 0.89 the moment an answer started
            # coming from the hierarchy instead of from prose.
            GRAPH_MARKERS = ("graph_outline", "graph_count", "underspecified")
            same = [i for i in ids if i.startswith(q["doc"]) or i in GRAPH_MARKERS]
            # The structured verdict first, the wording only as a fallback.
            # Retrieval now declines an unplaceable question with "does not
            # say which document to look in", which matches none of the
            # REFUSAL phrases -- so a working decline scored 0/5 and read as
            # a regression. Grading prose for a signal the response carries
            # explicitly is how that happens.
            declined = bool(r.get("underspecified")) or any(p in low for p in REFUSAL)
            if q["mode"] == "EXACT":
                ok = bool(q["expect"]) and q["expect"] in ans
            elif q["mode"] == "COUNT":
                ok = q["expect"] in ans
            elif q["mode"] == "SET":
                hit = sum(1 for c in q["expect"] if c[:28].lower() in low)
                ok = hit >= max(1, int(0.8 * len(q["expect"])))
            elif q["mode"] == "REFUSE":
                ok = declined
            else:
                ok = None                      # not machine-checkable
            rows.append({
                "doc": q["doc"], "shape": q["shape"], "mode": q["mode"],
                "right_doc": r.get("document_id") == q["doc"] if q["mode"] != "REFUSE" else None,
                "recall": (q["node"] in ids) if q["node"] else None,
                # None for REFUSE: precision here means "share retrieved from
                # the EXPECTED document", and a question meant to be declined
                # has no expected document. Scoring those 0.00 dragged the
                # mean down and, worse, read as "no context retrieved" when
                # the sources were 6 and 4 -- which sent a previous round of
                # this work chasing a defect that did not exist.
                "precision": None if q["mode"] == "REFUSE"
                else ((len(same) / len(ids)) if ids else 0.0),
                "ok": ok, "s": round(r.get("_elapsed", 0), 1),
            })
            x = rows[-1]
            print(f"  {d['title'][:18]:<20}{q['shape']:<16}{q['mode']:<10}{x['s']:>5}s "
                  f"doc={'-' if x['right_doc'] is None else ('Y' if x['right_doc'] else 'n')} "
                  f"P={'-   ' if x['precision'] is None else format(x['precision'],'.2f')} ok={'-' if x['ok'] is None else ('Y' if x['ok'] else 'n')}", flush=True)

    json.dump(rows, open("/tmp/shapes.json", "w"), indent=1)
    scored = [r for r in rows if r["ok"] is not None]
    rdrows = [r for r in rows if r["right_doc"] is not None]
    rec = [r for r in rows if r["recall"] is not None]
    print("\n  === overall ===")
    print(f"  questions                {len(rows)}")
    print(f"  right document           {sum(r['right_doc'] for r in rdrows)}/{len(rdrows)}"
          f" ({sum(r['right_doc'] for r in rdrows)/max(len(rdrows),1)*100:.0f}%)")
    prec = [r["precision"] for r in rows if r["precision"] is not None]
    print(f"  precision@k (mean)       {st.mean(prec):.2f}  (over {len(prec)} scoped questions)")
    if rec:
        print(f"  recall@k (source node)   {sum(r['recall'] for r in rec)}/{len(rec)}")
    print(f"  deterministically scored {sum(r['ok'] for r in scored)}/{len(scored)}"
          f" ({sum(r['ok'] for r in scored)/max(len(scored),1)*100:.0f}%)")
    print(f"  retrieval-only shapes    {len(rows)-len(scored)} (answer not machine-checkable)")
    print(f"  median latency           {st.median(r['s'] for r in rows):.1f}s")
    per = defaultdict(list)
    for r in scored: per[r["shape"]].append(r["ok"])
    print("\n  === scored shapes ===")
    for shape, v in sorted(per.items()):
        print(f"   {shape:<18} {sum(v)}/{len(v)}")


if __name__ == "__main__":
    main()
