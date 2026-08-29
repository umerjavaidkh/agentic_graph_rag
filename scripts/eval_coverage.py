"""eval_coverage.py — can retrieval reach the whole document, or only the front?

Every other harness here samples a document's nodes in `order` and stops
at the first 40-60, which on a 34-page paper means the questions come from
the opening pages. A document could be answerable at page 3 and invisible
at page 30 and score perfectly.

This asks the same kind of question -- a cloze over a stated fact, so the
expected answer is verifiable without a model -- but draws them evenly
across the page range and reports the score by where in the document the
answer lives. The question never names the document, so retrieval has to
place it from content alone.

    docker compose exec -T app python - < scripts/eval_coverage.py -- <doc_id>
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
import uuid
from collections import defaultdict

sys.path.insert(0, "scripts")

from eval_retrieval_depth import _CITATION, _FACT, _SENT, _is_prose  # noqa: E402
from src.shared.neo4j.driver import get_neo4j_driver  # noqa: E402

BUCKETS = 8          # slices of the document to sample evenly
PER_BUCKET = 3       # questions drawn from each slice


def ask(question: str, timeout: int = 240, thread: str = "") -> dict:
    # A fresh thread per question isolates them, which is right for
    # measuring cold placement and wrong for measuring the product: the
    # design is that naming a document once pins it for the conversation.
    # `thread` opts into that, and must stay alphanumeric -- a dot collapses
    # a thread id to `default` and silently merges unrelated conversations.
    body = json.dumps({
        "question": question, "user_id": "admin_001", "role": "admin",
        "tenant_id": "", "thread_id": thread or uuid.uuid4().hex,
    }).encode()
    t0 = time.perf_counter()
    try:
        req = urllib.request.Request(
            "http://localhost:8000/query", body, {"Content-Type": "application/json"})
        out = json.load(urllib.request.urlopen(req, timeout=timeout))
    except Exception as exc:
        return {"answer": "", "_error": type(exc).__name__,
                "_elapsed": time.perf_counter() - t0}
    out["_elapsed"] = time.perf_counter() - t0
    return out


def rare_clozes_from(text: str, limit: int, rare: set[str]) -> list[tuple[str, str]]:
    """Cloze a term that is rare ACROSS THE CORPUS, not just in this document.

    _FACT only matches numbers, and the middle of this paper has almost
    none -- three of eight slices produced no question at all, which would
    have been reported as coverage.

    The first attempt filled the gap with words occurring once in this
    document, and that was wrong: "captured", "separate", "according",
    "primarily" all occur once here by chance and are not distinctive at
    all. Seven answers contained the expected word while retrieval had
    resolved a different document -- the question was measuring the string,
    not the retrieval. Corpus document frequency is the difference between
    a word that is rare and a word that merely happened not to repeat.
    """
    out: list[tuple[str, str]] = []
    for sent in _SENT.split(text or ""):
        sent = " ".join(sent.split())
        if not (60 <= len(sent) <= 240) or not _is_prose(sent):
            continue
        if _CITATION.search(sent):
            continue
        for m in re.finditer(r"\b[A-Za-z][A-Za-z]{6,}\b", sent):
            word = m.group(0)
            if word.lower() not in rare or m.start() == 0:
                continue
            cloze = (sent[: m.start()] + " ______ " + sent[m.end():]).strip()
            if len(cloze) < 50:
                continue
            out.append((cloze, word))
            break
        if len(out) >= limit:
            break
    return out


def clozes_from(text: str, limit: int) -> list[tuple[str, str]]:
    """(cloze sentence, expected span) pairs, reusing the depth harness's rules."""
    out: list[tuple[str, str]] = []
    for sent in _SENT.split(text or ""):
        sent = " ".join(sent.split())
        if not (60 <= len(sent) <= 240) or not _is_prose(sent):
            continue
        if _CITATION.search(sent):
            continue
        m = _FACT.search(sent)
        if not m:
            continue
        span = m.group(0).strip()
        if len(span) < 2 or m.start() == 0:
            continue
        if re.fullmatch(r"(?:19|20)\d{2}", span) or re.fullmatch(r"\d+\.\d+", span):
            continue
        before = sent[: m.start()]
        if before.rstrip().endswith(",") and before.rstrip()[:-1].rstrip()[-1:].isdigit():
            continue
        if sent[m.end(): m.end() + 1] == "," and sent[m.end() + 1: m.end() + 2].isdigit():
            continue
        cloze = (sent[: m.start()] + " ______ " + sent[m.end():]).strip()
        if len(cloze) < 50:
            continue
        out.append((cloze, span))
        if len(out) >= limit:
            break
    return out


def main() -> None:
    # --named puts the document in the question. Without it a miss is
    # ambiguous: retrieval may have failed to place the question rather
    # than failed to reach the page, and those are different faults.
    named = "--named" in sys.argv
    # One conversation: the FIRST question names the document, the rest name
    # nothing and must ride the thread's memory of it.
    threaded = "--thread" in sys.argv
    doc = sys.argv[-1]
    with get_neo4j_driver().session() as s:
        rows = [dict(r) for r in s.run(
            """
            MATCH (n:Section|Page|Chapter)
            WHERE n.logical_doc_id = $d AND n.lifecycle_status = 'ACTIVE'
              AND size(coalesce(n.search_text, '')) > 300
            RETURN n.id AS node_id, n.search_text AS text,
                   coalesce(n.page_start, 0) AS page
            ORDER BY page
            """, d=doc)]
        pages = s.run(
            "MATCH (n:Page) WHERE n.logical_doc_id = $d AND n.lifecycle_status = 'ACTIVE' "
            "RETURN count(*) AS c", d=doc).single()["c"]

    if not rows:
        print(f"no usable nodes for {doc}")
        return

    lo = min(r["page"] for r in rows)
    hi = max(r["page"] for r in rows)
    span = max(hi - lo, 1)
    by_bucket: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        b = min(int((r["page"] - lo) / span * BUCKETS), BUCKETS - 1)
        by_bucket[b].append(r)

    # Terms rare across the whole corpus, so the expected answer actually
    # identifies this passage. Document-local rarity is not enough: see
    # rare_clozes_from.
    from src.unstructured.retrieval.services.term_stats import CorpusTermStats
    stats = CorpusTermStats()
    with get_neo4j_driver().session() as s2:
        stats._ensure(s2)
    local: set[str] = set()
    for r in rows:
        local.update(w.lower() for w in re.findall(r"\b[A-Za-z][A-Za-z]{6,}\b", r["text"] or ""))
    # A term absent from the corpus table was pruned for being rarer than
    # 2% of documents, which is exactly what is wanted here.
    rare = {w for w in local if stats._df.get(w, 0.0) < 0.02}

    questions: list[dict] = []
    for b in range(BUCKETS):
        taken = 0
        for r in by_bucket.get(b, []):
            found = clozes_from(r["text"], PER_BUCKET - taken)
            if len(found) < PER_BUCKET - taken:
                found += rare_clozes_from(
                    r["text"], PER_BUCKET - taken - len(found), rare)
            for cloze, expected in found:
                questions.append({
                    "bucket": b, "page": r["page"], "node": r["node_id"],
                    "expected": expected,
                    # The document is NOT named: retrieval has to place the
                    # question from its content, which is the real condition.
                    "q": (
                        f'In the document "{doc[4:] if doc.startswith("doc_") else doc}", '
                        f'fill in the blank with the exact value from the text: "{cloze}"'
                        if named else
                        f'Fill in the blank with the exact value from the '
                        f'source: "{cloze}"'
                    ),
                })
                taken += 1
            if taken >= PER_BUCKET:
                break

    print(f"# Coverage — {doc}\n")
    print(f"{pages} pages, nodes spanning page {lo}-{hi}, "
          f"{len(questions)} questions drawn evenly across {BUCKETS} slices.")
    cond = ("one thread, only the first question names the document"
            if threaded else
            "document named in every question" if named else
            "document never named, fresh thread each question")
    print(f"Condition: **{cond}**.\n")
    if "--dump" in sys.argv:
        short = doc[4:] if doc.startswith("doc_") else doc
        print("Ask these in ONE conversation, in order. Only the first names "
              "the document; the rest rely on the thread holding it.\n")
        print("| # | Page | Expected | Question |")
        print("|---|---|---|---|")
        for i, q in enumerate(questions, 1):
            text = q["q"]
            if i == 1:
                text = f'In the document "{short}", {text[0].lower()}{text[1:]}'
            print(f"| {i} | {q['page']} | `{q['expected']}` | "
                  f"{text.replace('|', chr(92) + '|')} |")
        return

    print("| Slice | Pages | Q | Correct | Right doc | Source node retrieved |")
    print("|---|---|---|---|---|---|")

    per = defaultdict(lambda: {"n": 0, "ok": 0, "doc": 0, "rec": 0, "s": []})
    detail = []
    thread = "cov" + uuid.uuid4().hex if threaded else ""
    for i, q in enumerate(questions):
        text = q["q"]
        if threaded and i == 0:
            short = doc[4:] if doc.startswith("doc_") else doc
            text = f'In the document "{short}", {text[0].lower()}{text[1:]}'
        r = ask(text, thread=thread)
        ans = r.get("answer") or ""
        ids = [x.get("id", "") for x in (r.get("sources") or []) if isinstance(x, dict)]
        ok = bool(q["expected"]) and q["expected"] in ans
        right = r.get("document_id") == doc
        rec = q["node"] in ids
        p = per[q["bucket"]]
        p["n"] += 1; p["ok"] += ok; p["doc"] += right; p["rec"] += rec
        p["s"].append(r.get("_elapsed", 0))
        detail.append((q, ok, right, rec, ans))

    for b in range(BUCKETS):
        p = per.get(b)
        if not p or not p["n"]:
            continue
        ps = [q["page"] for q in questions if q["bucket"] == b]
        print(f"| {b + 1}/{BUCKETS} | {min(ps)}-{max(ps)} | {p['n']} | "
              f"{p['ok']}/{p['n']} | {p['doc']}/{p['n']} | {p['rec']}/{p['n']} |")

    tot = sum(p["n"] for p in per.values())
    print(f"\n**Overall: {sum(p['ok'] for p in per.values())}/{tot} correct, "
          f"{sum(p['doc'] for p in per.values())}/{tot} right document, "
          f"{sum(p['rec'] for p in per.values())}/{tot} source node retrieved.**\n")

    # An answer holding the expected span while retrieval resolved a
    # different document is not a pass. It means the span was not
    # distinctive, so the question was measuring the string and not the
    # retrieval -- worth seeing rather than counting.
    odd = [q for q, ok, right, _c, _a in detail if ok and not right]
    if odd:
        print(f"## Correct span, but retrieval resolved another document "
              f"({len(odd)})\n")
        print("Counted as correct above; treat them as unproven.\n")
        for q in odd:
            print(f"- **p.{q['page']}** expected `{q['expected']}`")
        print()

    misses = [(q, a) for q, ok, _r, _c, a in detail if not ok]
    if misses:
        print(f"## Misses ({len(misses)})\n")
        for q, a in misses:
            print(f"- **p.{q['page']}** expected `{q['expected']}` — "
                  f"got: {(a or '(no answer)')[:130]}")


if __name__ == "__main__":
    main()
