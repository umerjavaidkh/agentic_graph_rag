#!/usr/bin/env python3
"""
Run document (Go.Data) and/or structured (Northwind) RAG eval suites against POST /query.

Environment:
  EVAL_BASE_URL   default http://localhost:8000
  EVAL_TIMEOUT    seconds per request (default 180)

Examples:
  python scripts/run_rag_eval.py --suite all
  python scripts/run_rag_eval.py --suite structured --id nw_04
  python scripts/run_rag_eval.py --suite document --output /tmp/rag_eval.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.validators import ValidationResult, validate_response  # noqa: E402

SUITE_PATHS = {
    "document": ROOT / "eval" / "document_rag_suite.json",
    "structured": ROOT / "eval" / "structured_rag_suite.json",
    "advanced": ROOT / "eval" / "advanced_structured_suite.json",
}


def load_suite(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def post_query(
    base_url: str,
    question: str,
    *,
    user_id: str,
    role: str,
    thread_id: str,
    timeout: float,
) -> dict[str, Any]:
    payload = {
        "question": question,
        "user_id": user_id,
        "role": role,
        "thread_id": thread_id,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/query",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def post_feedback_outcome(
    base_url: str,
    request_id: str,
    *,
    passed: bool,
    case_id: str,
    timeout: float = 30.0,
) -> None:
    payload = {"request_id": request_id, "passed": passed, "case_id": case_id}
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/feedback/outcome",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()


def run_case(
    base_url: str,
    suite_meta: dict[str, Any],
    case: dict[str, Any],
    *,
    timeout: float,
    dry_run: bool,
    attach_feedback: bool,
) -> dict[str, Any]:
    user_id = case.get("user_id") or suite_meta.get("default_user_id", "public_001")
    role = case.get("role") or suite_meta.get("default_role", "public")
    thread_id = f"eval_{case['id']}_{int(time.time() * 1000)}"

    record: dict[str, Any] = {
        "id": case["id"],
        "suite": suite_meta.get("suite"),
        "question": case["question"],
        "user_id": user_id,
        "role": role,
    }

    if dry_run:
        record["status"] = "skipped"
        return record

    try:
        t0 = time.perf_counter()
        response = post_query(
            base_url,
            case["question"],
            user_id=user_id,
            role=role,
            thread_id=thread_id,
            timeout=timeout,
        )
        elapsed = time.perf_counter() - t0
        validation: ValidationResult = validate_response(case, response)
        record.update(
            {
                "status": "pass" if validation.passed else "fail",
                "elapsed_s": round(elapsed, 2),
                "request_id": response.get("request_id"),
                "route_tool": response.get("route_tool"),
                "agent": response.get("agent"),
                "total_chunks": response.get("total_chunks"),
                "answer_preview": (response.get("answer") or "")[:240],
                "checks": validation.checks,
                "failures": validation.failures,
            }
        )
        if attach_feedback and response.get("request_id"):
            try:
                post_feedback_outcome(
                    base_url,
                    response["request_id"],
                    passed=validation.passed,
                    case_id=case["id"],
                )
            except Exception as exc:
                record["feedback_attach_error"] = str(exc)[:200]
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")[:500]
        record.update({"status": "error", "error": f"HTTP {e.code}: {body}"})
    except Exception as e:
        record.update({"status": "error", "error": str(e)})

    return record


def collect_cases(suite_name: str, case_id: str | None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    paths = []
    if suite_name == "all":
        paths = [SUITE_PATHS["document"], SUITE_PATHS["structured"], SUITE_PATHS["advanced"]]
    else:
        paths = [SUITE_PATHS[suite_name]]

    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for path in paths:
        meta = load_suite(path)
        for case in meta.get("cases") or []:
            if case_id and case.get("id") != case_id:
                continue
            out.append((meta, case))
    return out


def print_summary(results: list[dict[str, Any]]) -> int:
    passed = sum(1 for r in results if r.get("status") == "pass")
    failed = sum(1 for r in results if r.get("status") == "fail")
    errors = sum(1 for r in results if r.get("status") == "error")
    skipped = sum(1 for r in results if r.get("status") == "skipped")
    total = len(results)

    print()
    print(f"Results: {passed} pass, {failed} fail, {errors} error, {skipped} skipped / {total} total")
    for r in results:
        mark = {"pass": "OK", "fail": "FAIL", "error": "ERR", "skipped": "SKIP"}.get(
            r.get("status", ""), "?"
        )
        line = f"  [{mark}] {r.get('id')}"
        if r.get("failures"):
            line += f" — {'; '.join(r['failures'][:2])}"
        if r.get("error"):
            line += f" — {r['error'][:120]}"
        if r.get("status") == "fail" and r.get("answer_preview"):
            line += f"\n       answer: {r['answer_preview'][:200]}"
        print(line)

    return 0 if failed == 0 and errors == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Run RAG eval suites (document + structured).")
    parser.add_argument(
        "--suite",
        choices=["document", "structured", "advanced", "all"],
        default="all",
        help="Which suite to run (default: all = 20 document + 10 structured + 10 advanced)",
    )
    parser.add_argument("--id", help="Run a single case id (e.g. nw_04, godata_a01)")
    parser.add_argument(
        "--base-url",
        default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000"),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=float(os.environ.get("EVAL_TIMEOUT", "180")),
    )
    parser.add_argument("--output", help="Write JSON report to this path")
    parser.add_argument("--dry-run", action="store_true", help="List cases without calling API")
    parser.add_argument(
        "--attach-feedback",
        action="store_true",
        help="POST pass/fail to /feedback/outcome (requires RETRIEVAL_FEEDBACK_ENABLED on server)",
    )
    args = parser.parse_args()

    pairs = collect_cases(args.suite, args.id)
    if not pairs:
        print("No cases matched.", file=sys.stderr)
        return 2

    print(f"Running {len(pairs)} case(s) against {args.base_url} (suite={args.suite})")
    results = [
        run_case(
            args.base_url,
            meta,
            case,
            timeout=args.timeout,
            dry_run=args.dry_run,
            attach_feedback=args.attach_feedback,
        )
        for meta, case in pairs
    ]

    report = {
        "base_url": args.base_url,
        "suite": args.suite,
        "case_count": len(results),
        "results": results,
    }
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    return print_summary(results)


if __name__ == "__main__":
    raise SystemExit(main())
