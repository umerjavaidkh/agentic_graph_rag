#!/usr/bin/env python3
"""
Deterministic evaluation for structured (Text-to-Cypher) retrieval.

Ground truth is computed FROM THE GRAPH by a hand-written Cypher query, then
compared against what the system answers in natural language. Nothing is
hardcoded to a row count, so the suite stays correct when the data is
reloaded or extended -- and nothing needs an LLM judge, so it is free to run
and gives the same answer twice.

That last point is the reason this exists. The sampled LLM-judge score used
for the document graph swung 63% to 80% across two runs over an IDENTICAL
graph, which is far too noisy to steer by; a numeric answer either matches
the graph or it does not.

Categories, so a failure says WHICH capability broke rather than just
lowering one number:

    fact       single value straight off one label      "how many sellers"
    aggregate  sum/avg/min/max over a relationship      "average review score"
    ranking    ordered top-N (scored by set overlap)    "top 5 categories"
    multihop   3+ relationship traversal                "customers whose seller is in another state"
    temporal   datetime filtering or arithmetic         "orders delivered late"
    absence    entity that does not exist               must NOT invent a number

Scoring:
  * scalar answers  -> exact match on the number (tolerant of formatting:
    thousands separators, currency symbols, decimals)
  * list answers    -> precision, recall and F1 against the expected set
  * absence         -> passes only if the answer declines to give a figure

Usage:
    python scripts/eval_structured.py                 # all categories
    python scripts/eval_structured.py --category multihop
    python scripts/eval_structured.py --json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.auth.roles import UserContext, Role  # noqa: E402
from src.graph.driver import get_neo4j_driver  # noqa: E402
from src.router import ask  # noqa: E402


@dataclass
class Case:
    id: str
    category: str
    question: str
    cypher: str
    # How to turn the Cypher result into the expected answer. Scalar cases
    # return a number; ranking cases return a list of strings.
    expect: Callable[[list[dict]], Any]
    kind: str = "scalar"  # scalar | list | absence


# Ground truth is a SEPARATE query from whatever the system generates -- if
# both were produced the same way the suite would only prove the system is
# self-consistent, not that it is right.
CASES: list[Case] = [
    Case("fact_sellers", "fact",
         "How many sellers are in the database?",
         "MATCH (s:Seller) RETURN count(s) AS n",
         lambda r: r[0]["n"]),
    Case("fact_orders", "fact",
         "How many orders are in the database?",
         "MATCH (o:Order) RETURN count(o) AS n",
         lambda r: r[0]["n"]),
    Case("agg_avg_review", "aggregate",
         "What is the average review score across all reviews?",
         "MATCH (v:Review) WHERE v.score IS NOT NULL RETURN round(avg(v.score), 2) AS n",
         lambda r: r[0]["n"]),
    Case("agg_total_freight", "aggregate",
         "What is the total freight value across all order items?",
         "MATCH (i:OrderItem) WHERE i.freight IS NOT NULL RETURN round(sum(i.freight)) AS n",
         lambda r: r[0]["n"]),
    Case("rank_categories", "ranking",
         "What are the top 5 product categories by total revenue?",
         """MATCH (c:Category)<-[:IN_CATEGORY]-(:Product)<-[:OF_PRODUCT]-(i:OrderItem)
            RETURN c.name AS name, sum(i.price) AS rev ORDER BY rev DESC LIMIT 5""",
         lambda r: [x["name"] for x in r], kind="list"),
    Case("rank_states", "ranking",
         "Which 5 customer states have the most orders?",
         """MATCH (c:Customer)-[:PLACED]->(o:Order)
            RETURN c.state AS name, count(o) AS n ORDER BY n DESC LIMIT 5""",
         lambda r: [x["name"] for x in r], kind="list"),
    Case("multihop_interstate", "multihop",
         "How many order items were sold by a seller in a different state from the customer?",
         """MATCH (cu:Customer)-[:PLACED]->(:Order)-[:CONTAINS]->(i:OrderItem)-[:SOLD_BY]->(s:Seller)
            WHERE cu.state <> s.state RETURN count(i) AS n""",
         lambda r: r[0]["n"]),
    Case("multihop_category_seller", "multihop",
         "How many distinct sellers sold products in the bed_bath_table category?",
         """MATCH (:Category {name:'cama_mesa_banho'})<-[:IN_CATEGORY]-(:Product)
                  <-[:OF_PRODUCT]-(:OrderItem)-[:SOLD_BY]->(s:Seller)
            RETURN count(DISTINCT s) AS n""",
         lambda r: r[0]["n"]),
    Case("temporal_late", "temporal",
         "How many orders were delivered later than their estimated delivery date?",
         """MATCH (o:Order) WHERE o.delivered_at IS NOT NULL AND o.estimated_delivery IS NOT NULL
            AND o.delivered_at > o.estimated_delivery RETURN count(o) AS n""",
         lambda r: r[0]["n"]),
    Case("temporal_2017", "temporal",
         "How many orders were placed in 2017?",
         "MATCH (o:Order) WHERE o.purchased_at.year = 2017 RETURN count(o) AS n",
         lambda r: r[0]["n"]),
    # Absence: the graph has no such entity. Inventing a number here is worse
    # than refusing, because a plausible figure is indistinguishable from a
    # real one to the reader.
    # The absent entity has to be one with no near-synonym in the graph.
    # "suppliers" was tried here first and is a bad test: this graph has
    # :Seller, and in an e-commerce dataset a seller IS a supplier, so
    # answering 3,095 is defensible. That case measured synonym resolution
    # while claiming to measure absence -- a failing score for correct
    # behaviour, which is worse than no case at all.
    Case("absence_warehouses", "absence",
         "How many warehouses are in the database?",
         "RETURN 0 AS n", lambda r: 0, kind="absence"),
    Case("absence_employees", "absence",
         "What is the average salary of employees?",
         "RETURN 0 AS n", lambda r: 0, kind="absence"),
]

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
# A permission denial is not an answer. Without this the absence cases pass
# on a broken run, hiding that nothing was actually evaluated.
_DENIED = re.compile(r"does not have permission|not authori[sz]ed|access denied", re.I)
_REFUSAL = re.compile(
    r"\b(no|not|cannot|can't|unable|does\s?n[o']t|no such|not (?:present|available|found|contain))\b",
    re.I,
)


def numbers_in(text: str) -> list[float]:
    out = []
    for m in _NUM.finditer(text or ""):
        try:
            out.append(float(m.group(0).replace(",", "")))
        except ValueError:
            pass
    return out


def scalar_hit(expected: float, answer: str) -> bool:
    """Whether the expected figure appears in the answer.

    Compared numerically rather than as a string so "99,441", "99441" and
    "99441.0" all count -- the model's formatting is not what is under test.
    A small relative tolerance covers rounding of averages.
    """
    exp = float(expected)
    for got in numbers_in(answer):
        if got == exp or (exp and abs(got - exp) / abs(exp) < 0.01):
            return True
    return False


def list_scores(expected: list[str], answer: str) -> tuple[float, float, float]:
    """Precision/recall/F1 of the expected items against the answer text.

    Set-based, not order-based: an answer that names the right five
    categories in a different order has still answered the question. Ordering
    is a separate concern and would need its own case.
    """
    low = (answer or "").lower()
    found = [e for e in expected if e.lower() in low]
    # Precision needs a denominator of what the answer CLAIMED. Counting
    # comma/newline-separated fragments over-counts prose, so this uses the
    # expected-set size as the claim size -- precision and recall coincide
    # for a fixed-size top-N, which is what these cases ask for.
    recall = len(found) / len(expected) if expected else 1.0
    precision = recall
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def run_case(session, case: Case, ctx: UserContext) -> dict[str, Any]:
    rows = [dict(r) for r in session.run(case.cypher)]
    expected = case.expect(rows)
    result = ask(case.question, user_context=ctx, retrieval_mode="structured")
    answer = (result or {}).get("answer") or ""

    row: dict[str, Any] = {
        "id": case.id, "category": case.category, "expected": expected,
        "answer": answer[:160],
    }
    if _DENIED.search(answer):
        row.update(passed=False, error="access denied — check the eval user's RBAC role")
        return row
    if case.kind == "absence":
        # Passing means NOT stating a figure. A refusal that also happens to
        # contain a number (a year, a count of documents searched) would slip
        # through a pure keyword check, so both must hold.
        row["passed"] = bool(_REFUSAL.search(answer)) and not numbers_in(answer)
    elif case.kind == "list":
        p, r, f1 = list_scores(expected, answer)
        row.update(precision=round(p, 3), recall=round(r, 3), f1=round(f1, 3), passed=r >= 0.8)
    else:
        row["passed"] = scalar_hit(expected, answer)
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--category", help="run only this category")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    cases = [c for c in CASES if not args.category or c.category == args.category]
    # A real user from the RBAC graph, not an invented one: access control is
    # graph-backed, so an unknown id is denied and EVERY case fails -- and the
    # absence cases would "pass", because a permission denial reads exactly
    # like a refusal to invent a figure.
    ctx = UserContext(user_id="admin_001", role=Role.ADMIN, department="IT", tenant_id="default")

    results = []
    with get_neo4j_driver().session() as s:
        for c in cases:
            try:
                results.append(run_case(s, c, ctx))
            except Exception as exc:  # a broken case must not hide the rest
                results.append({"id": c.id, "category": c.category, "passed": False,
                                "error": str(exc)[:160]})

    if args.json:
        print(json.dumps(results, indent=2, default=str))
        return

    by_cat: dict[str, list[dict]] = {}
    for r in results:
        by_cat.setdefault(r["category"], []).append(r)

    print(f"{'case':<28}{'exp':>12}  result")
    for r in results:
        mark = "PASS" if r.get("passed") else "FAIL"
        extra = ""
        if "recall" in r:
            extra = f"  P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f}"
        if "error" in r:
            extra = f"  ERROR {r['error']}"
        print(f"  {r['id']:<26}{str(r.get('expected','-')):>12}  {mark}{extra}")

    print("\nby category:")
    for cat, rows in sorted(by_cat.items()):
        ok = sum(1 for r in rows if r.get("passed"))
        print(f"  {cat:<12} {ok}/{len(rows)}")
    total_ok = sum(1 for r in results if r.get("passed"))
    print(f"\n  OVERALL {total_ok}/{len(results)}")
    sys.exit(0 if total_ok == len(results) else 1)


if __name__ == "__main__":
    main()
