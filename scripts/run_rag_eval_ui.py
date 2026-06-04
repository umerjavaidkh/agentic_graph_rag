#!/usr/bin/env python3
"""
Drive the live /chat UI with Playwright, record a video of every question, and
validate each rendered answer with eval/validators.py.

Unlike scripts/run_rag_eval.py (which posts to /query directly), this runner
exercises the real browser UI — typing each question, waiting for the rendered
answer (charts, tables, sources, meta chips), and capturing the underlying
/query JSON response off the network so the same heuristic validators apply.
The result is a single video that doubles as a pass/fail test report.

Setup (one-time, kept out of the server venv):
  python3 -m venv .venv
  .venv/bin/python -m pip install playwright
  .venv/bin/python -m playwright install chromium

Run (server must be up at EVAL_BASE_URL with the target corpus ingested):
  .venv/bin/python scripts/run_rag_eval_ui.py --suite document
  .venv/bin/python scripts/run_rag_eval_ui.py --suite all --pause 2.5
  .venv/bin/python scripts/run_rag_eval_ui.py --id godata_a01 --headed

Environment:
  EVAL_BASE_URL   default http://localhost:8000
  EVAL_TIMEOUT    seconds to wait per answer (default 180)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.validators import ValidationResult, validate_response  # noqa: E402

try:
    from playwright.sync_api import sync_playwright
except ImportError:  # pragma: no cover - guidance only
    sys.stderr.write(
        "Playwright is not installed. Install it (ideally in .venv):\n"
        "  .venv/bin/python -m pip install playwright\n"
        "  .venv/bin/python -m playwright install chromium\n"
    )
    raise SystemExit(2)

SUITE_PATHS = {
    "document": ROOT / "eval" / "document_rag_suite.json",
    "structured": ROOT / "eval" / "structured_rag_suite.json",
    "advanced": ROOT / "eval" / "advanced_structured_suite.json",
}

# Injected once; a fixed banner so the recording reads as a test report.
BANNER_JS = """
(() => {
  if (document.getElementById('eval-banner')) return;
  const bar = document.createElement('div');
  bar.id = 'eval-banner';
  bar.style.cssText = [
    'position:fixed','top:0','left:0','right:0','z-index:99999',
    'font-family:Inter,system-ui,sans-serif','font-size:14px','font-weight:600',
    'padding:10px 18px','color:#fff','background:#1e293b',
    'border-bottom:1px solid rgba(255,255,255,0.15)','letter-spacing:0.01em',
    'box-shadow:0 2px 12px rgba(0,0,0,0.4)','display:flex','gap:14px','align-items:center'
  ].join(';');
  bar.innerHTML = '<span id="eval-banner-tag">Eval UI runner</span>'
    + '<span id="eval-banner-msg" style="font-weight:500;opacity:0.9"></span>'
    + '<span id="eval-banner-result" style="margin-left:auto;font-weight:700"></span>';
  document.body.appendChild(bar);
  document.body.style.paddingTop = '46px';
})();
"""


def set_banner(page, tag: str, msg: str, result: str = "", color: str = "#1e293b") -> None:
    page.evaluate(
        """([tag, msg, result, color]) => {
            const bar = document.getElementById('eval-banner');
            if (!bar) return;
            bar.style.background = color;
            document.getElementById('eval-banner-tag').textContent = tag;
            document.getElementById('eval-banner-msg').textContent = msg;
            document.getElementById('eval-banner-result').textContent = result;
        }""",
        [tag, msg, result, color],
    )


def read_through(page, seconds: float) -> None:
    """Gently pan the chat from the question (top) to the end of the answer so a
    viewer can read long responses, then linger briefly at the bottom."""
    info = page.evaluate(
        "() => { const b=document.getElementById('chat_body');"
        " return b ? {sh:b.scrollHeight, ch:b.clientHeight} : null; }"
    )
    if not info:
        time.sleep(seconds)
        return
    extra = max(0, int(info["sh"]) - int(info["ch"]))
    page.evaluate("() => { const b=document.getElementById('chat_body'); if(b) b.scrollTop=0; }")
    if extra < 24:
        time.sleep(seconds)
        return
    steps = 24
    dwell_top = min(1.2, seconds * 0.25)
    time.sleep(dwell_top)
    pan = max(0.3, seconds - dwell_top)
    for i in range(1, steps + 1):
        page.evaluate(
            "(y) => { const b=document.getElementById('chat_body'); if(b) b.scrollTop=y; }",
            int(extra * i / steps),
        )
        time.sleep(pan / steps)


def load_suite(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def collect_cases(suite_name: str, case_id: str | None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    paths = (
        [SUITE_PATHS["document"], SUITE_PATHS["structured"], SUITE_PATHS["advanced"]]
        if suite_name == "all"
        else [SUITE_PATHS[suite_name]]
    )
    out: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for path in paths:
        meta = load_suite(path)
        for case in meta.get("cases") or []:
            if case_id and case.get("id") != case_id:
                continue
            out.append((meta, case))
    return out


def run_case(page, meta: dict[str, Any], case: dict[str, Any], idx: int, total: int,
             *, timeout_ms: int, pause_s: float, keep_history: bool) -> dict[str, Any]:
    user_id = case.get("user_id") or meta.get("default_user_id", "public_001")
    role = case.get("role") or meta.get("default_role", "public")
    cid = case["id"]
    tier = case.get("tier", "")
    progress = f"[{idx}/{total}] {cid} · {tier}"

    record: dict[str, Any] = {
        "id": cid, "suite": meta.get("suite"), "question": case["question"],
        "user_id": user_id, "role": role,
    }

    # Sidebar context (read by the page at send-time).
    page.select_option("#role", role)
    page.fill("#user_id", user_id)

    if not keep_history:
        page.click("#new_chat")
        page.wait_for_selector(".welcome-card", timeout=10_000)

    set_banner(page, progress, case["question"][:90], "running…", "#1e3a5f")
    page.fill("#q", case["question"])

    try:
        with page.expect_response(
            lambda r: r.request.method == "POST" and r.url.rstrip("/").endswith("/query"),
            timeout=timeout_ms,
        ) as resp_info:
            page.click("#send")
        response = resp_info.value
        status_code = response.status
        payload = response.json() if status_code == 200 else {}
        if status_code != 200:
            record.update({"status": "error", "error": f"HTTP {status_code}: {response.text()[:300]}"})
            set_banner(page, progress, case["question"][:90], f"ERROR {status_code}", "#7f1d1d")
            time.sleep(pause_s)
            return record

        # Wait for the assistant bubble to finish rendering.
        page.wait_for_selector(".msg.assistant.thinking", state="detached", timeout=timeout_ms)
        page.wait_for_selector(".msg.assistant:not(.thinking) .bubble", timeout=timeout_ms)

        validation: ValidationResult = validate_response(case, payload)
        passed = validation.passed
        record.update({
            "status": "pass" if passed else "fail",
            "route_tool": payload.get("route_tool"),
            "agent": payload.get("agent"),
            "answer_preview": (payload.get("answer") or "")[:240],
            "checks": validation.checks,
            "failures": validation.failures,
        })
        result_txt = "PASS" if passed else "FAIL: " + "; ".join(validation.failures[:1])
        set_banner(page, progress, case["question"][:90], result_txt,
                   "#14532d" if passed else "#7f1d1d")
        # Readable pan through the question + full answer.
        read_through(page, pause_s)
        return record
    except Exception as e:  # noqa: BLE001
        record.update({"status": "error", "error": str(e)[:300]})
        set_banner(page, progress, case["question"][:90], "ERROR", "#7f1d1d")

    time.sleep(pause_s)
    return record


def print_summary(results: list[dict[str, Any]]) -> int:
    p = sum(1 for r in results if r.get("status") == "pass")
    f = sum(1 for r in results if r.get("status") == "fail")
    e = sum(1 for r in results if r.get("status") == "error")
    print(f"\nResults: {p} pass, {f} fail, {e} error / {len(results)} total")
    for r in results:
        mark = {"pass": "OK", "fail": "FAIL", "error": "ERR"}.get(r.get("status", ""), "?")
        line = f"  [{mark}] {r.get('id')}"
        if r.get("failures"):
            line += f" — {'; '.join(r['failures'][:2])}"
        if r.get("error"):
            line += f" — {r['error'][:120]}"
        print(line)
    return 0 if f == 0 and e == 0 else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Record a UI video of the RAG eval suite.")
    parser.add_argument("--suite", choices=["document", "structured", "advanced", "all"], default="document")
    parser.add_argument("--id", help="Run a single case id (e.g. godata_a01)")
    parser.add_argument("--base-url", default=os.environ.get("EVAL_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("EVAL_TIMEOUT", "180")))
    parser.add_argument("--pause", type=float, default=2.0, help="Seconds to linger on each answer (video pacing)")
    parser.add_argument("--out-dir", help="Output directory (default eval/ui_runs/<timestamp>)")
    parser.add_argument("--headed", action="store_true", help="Show the browser window")
    parser.add_argument("--keep-history", action="store_true",
                        help="Do not reset the chat between questions (continuous scroll, shared thread)")
    args = parser.parse_args()

    pairs = collect_cases(args.suite, args.id)
    if not pairs:
        print("No cases matched.", file=sys.stderr)
        return 2

    out_dir = Path(args.out_dir) if args.out_dir else (
        ROOT / "eval" / "ui_runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    timeout_ms = int(args.timeout * 1000)

    print(f"Recording {len(pairs)} case(s) from {args.base_url} (suite={args.suite})")
    print(f"Output dir: {out_dir}")

    results: list[dict[str, Any]] = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=not args.headed)
        context = browser.new_context(
            viewport={"width": 1280, "height": 900},
            record_video_dir=str(out_dir),
            record_video_size={"width": 1280, "height": 900},
        )
        page = context.new_page()
        page.goto(f"{args.base_url.rstrip('/')}/chat", wait_until="domcontentloaded")
        page.wait_for_selector("#q", timeout=15_000)
        page.add_init_script(BANNER_JS)  # for later navigations
        page.evaluate(BANNER_JS)

        for i, (meta, case) in enumerate(pairs, start=1):
            print(f"  -> [{i}/{len(pairs)}] {case['id']}")
            results.append(run_case(
                page, meta, case, i, len(pairs),
                timeout_ms=timeout_ms, pause_s=args.pause, keep_history=args.keep_history,
            ))

        set_banner(page, "Eval UI runner", "done", "complete", "#0f172a")
        time.sleep(1.5)

        # Capture the source path BEFORE closing; the file is finalized on close.
        video_src = None
        try:
            if page.video:
                video_src = page.video.path()
        except Exception:  # noqa: BLE001
            video_src = None
        context.close()  # finalizes the .webm
        browser.close()

        final_video = out_dir / f"rag_eval_ui_{args.suite}.webm"
        if video_src and Path(video_src).exists():
            try:
                Path(video_src).replace(final_video)
                print(f"\nVideo: {final_video}")
            except Exception as e:  # noqa: BLE001
                final_video = Path(video_src)
                print(f"\nVideo: {final_video} (kept original name: {e})")
        else:
            print(f"\nVideo saved in {out_dir}")

    report = {
        "base_url": args.base_url, "suite": args.suite,
        "case_count": len(results), "results": results,
        "video": str(final_video) if pairs else None,
    }
    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Report: {report_path}")

    return print_summary(results)


if __name__ == "__main__":
    raise SystemExit(main())
