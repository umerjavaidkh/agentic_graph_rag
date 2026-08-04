#!/usr/bin/env python3
"""
scripts/fetch_sec_edgar_corpus.py — pull a real SEC EDGAR demo corpus.

Downloads 10-K/10-Q filings for a curated list of well-known public companies
over the last N years via the official SEC EDGAR Submissions API, converts
each filing's primary HTML document to PDF, and saves it under
sample_data_to_test/unstructured/sec_edgar/<TICKER>/ — ready for
POST /ingest/corpus.

SEC requires a descriptive User-Agent on every request (fair access policy):
https://www.sec.gov/os/webmaster-faq#developers
Set SEC_EDGAR_USER_AGENT before running, e.g.:

    export SEC_EDGAR_USER_AGENT="Jane Doe jane@example.com"

Usage:
    python3 scripts/fetch_sec_edgar_corpus.py --dry-run          # preview only, no network writes
    python3 scripts/fetch_sec_edgar_corpus.py --limit 1000 --years 10
    python3 scripts/fetch_sec_edgar_corpus.py --tickers AAPL MSFT --limit 50

Rate limit: SEC asks for <=10 requests/second; this script defaults to ~4/s.

macOS + weasyprint: if you see "cannot load library 'libpango-1.0-0'", run
`brew install cairo pango gdk-pixbuf libffi` first — this script also sets
DYLD_FALLBACK_LIBRARY_PATH itself as a convenience.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

# Must be set before `weasyprint` is imported so its cffi dlopen() calls can
# find Homebrew's libpango/libcairo on macOS (not on the default search path).
os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", "/opt/homebrew/lib:/usr/local/lib")

import requests  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "sample_data_to_test" / "unstructured" / "sec_edgar"

TICKER_JSON_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession_nodash}/{doc}"

FORM_TYPES = {"10-K", "10-Q"}

# ~25 well-known companies across sectors — deep history (10-K + 10-Q over
# several years per company) rather than broad-shallow coverage, per the
# "few companies, deep history" choice: better for multi-hop and temporal
# questions ("how did revenue change 2015-2024") and cross-company compares
# among names a demo audience will actually recognize.
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",       # tech
    "JPM", "GS", "BAC", "V",                        # finance
    "WMT", "COST", "TGT",                           # retail
    "JNJ", "PFE", "UNH",                             # healthcare
    "BA", "CAT", "GE",                               # industrial
    "KO", "PEP", "MCD", "NKE",                       # consumer
    "COP", "CVX",                                     # energy (XOM's ticker moved to a new
                                                        # holding-co CIK in 2026 with no filing
                                                        # history yet — COP has full history)
    "TSLA", "F",                                     # auto
]


def _session(user_agent: str) -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": user_agent, "Accept-Encoding": "gzip, deflate"})
    return s


def _load_ticker_cik_map(session: requests.Session) -> dict[str, int]:
    resp = session.get(TICKER_JSON_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return {row["ticker"].upper(): int(row["cik_str"]) for row in data.values()}


def _get_submissions(session: requests.Session, cik: int) -> dict[str, Any]:
    resp = session.get(SUBMISSIONS_URL.format(cik=cik), timeout=30)
    resp.raise_for_status()
    return resp.json()


def _yield_from_page(page: dict[str, Any], cutoff: date):
    forms = page.get("form", [])
    dates = page.get("filingDate", [])
    accns = page.get("accessionNumber", [])
    docs = page.get("primaryDocument", [])
    for form, fdate, accn, doc in zip(forms, dates, accns, docs):
        if form not in FORM_TYPES:
            continue
        if date.fromisoformat(fdate) < cutoff:
            continue
        yield form, fdate, accn, doc


def _iter_filings(
    session: requests.Session,
    cik: int,
    submissions: dict[str, Any],
    cutoff: date,
    rate_delay: float,
):
    """
    Yield (form, filingDate, accessionNumber, primaryDocument) for 10-K/10-Q
    filings back to `cutoff`. `filings.recent` only holds a rolling window —
    high-volume filers (large banks especially) can have that window cover
    just the last few weeks, with everything older paginated via
    `filings.files[]` (each a separate JSON page, newest-first). Walk those
    pages too, stopping once a page's filingTo predates the cutoff.
    """
    yield from _yield_from_page(submissions.get("filings", {}).get("recent", {}), cutoff)

    for page_ref in submissions.get("filings", {}).get("files", []):
        page_to = date.fromisoformat(page_ref["filingTo"])
        if page_to < cutoff:
            break  # pages are newest-first; nothing older is worth fetching
        resp = session.get(f"https://data.sec.gov/submissions/{page_ref['name']}", timeout=30)
        time.sleep(rate_delay)
        if not resp.ok:
            continue
        yield from _yield_from_page(resp.json(), cutoff)


def _html_to_pdf(html_bytes: bytes, base_url: str) -> bytes:
    from weasyprint import HTML

    return HTML(string=html_bytes, base_url=base_url).write_pdf()


def fetch_corpus(
    tickers: list[str],
    *,
    years: int,
    limit: int,
    per_company_limit: int,
    rate_delay: float,
    user_agent: str,
    dry_run: bool,
) -> None:
    session = _session(user_agent)
    cutoff = date.today() - timedelta(days=365 * years)

    print(f"Resolving CIKs for {len(tickers)} tickers...")
    ticker_cik = _load_ticker_cik_map(session)
    time.sleep(rate_delay)

    missing = [t for t in tickers if t not in ticker_cik]
    if missing:
        print(f"  WARNING: no CIK found for: {', '.join(missing)}")

    plan: list[tuple[str, int, str, str, str, str]] = []  # ticker, cik, form, date, accn, doc
    for ticker in tickers:
        cik = ticker_cik.get(ticker)
        if cik is None:
            continue
        submissions = _get_submissions(session, cik)
        time.sleep(rate_delay)
        company_count = 0
        for form, fdate, accn, doc in _iter_filings(session, cik, submissions, cutoff, rate_delay):
            if company_count >= per_company_limit:
                break
            if not doc:
                continue
            plan.append((ticker, cik, form, fdate, accn, doc))
            company_count += 1
        print(f"  {ticker}: {company_count} filings (10-K/10-Q, since {cutoff.isoformat()})")
        if len(plan) >= limit:
            break

    plan = plan[:limit]
    n_k = sum(1 for p in plan if p[2] == "10-K")
    n_q = sum(1 for p in plan if p[2] == "10-Q")
    total_companies = len({p[0] for p in plan})
    print(
        f"\nPlan: {len(plan)} filings ({n_k} 10-K, {n_q} 10-Q) "
        f"across {total_companies} companies, {years}-year window."
    )
    print(f"Output: {OUT_DIR}")

    if dry_run:
        print("\n--dry-run: no files downloaded. Re-run without --dry-run to fetch.")
        for ticker, _cik, form, fdate, _accn, doc in plan[:15]:
            print(f"  {ticker:6s} {form:5s} {fdate}  {doc}")
        if len(plan) > 15:
            print(f"  ... and {len(plan) - 15} more")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ok, failed = 0, 0
    for i, (ticker, cik, form, fdate, accn, doc) in enumerate(plan, 1):
        accn_nodash = accn.replace("-", "")
        doc_url = ARCHIVE_URL.format(cik=cik, accession_nodash=accn_nodash, doc=doc)
        out_path = OUT_DIR / ticker / f"{ticker}_{form}_{fdate}.pdf"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if out_path.exists():
            ok += 1
            continue
        try:
            resp = session.get(doc_url, timeout=60)
            resp.raise_for_status()
            pdf_bytes = _html_to_pdf(resp.content, base_url=doc_url)
            out_path.write_bytes(pdf_bytes)
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"  [{i}/{len(plan)}] FAILED {ticker} {form} {fdate}: {exc}")
        if i % 25 == 0 or i == len(plan):
            print(f"  [{i}/{len(plan)}] {ok} ok, {failed} failed")
        time.sleep(rate_delay)

    print(f"\nDone: {ok} PDFs written to {OUT_DIR}, {failed} failed.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="Tickers to pull (default: curated 25).")
    p.add_argument("--years", type=int, default=10, help="How many years of history per company (default: 10).")
    p.add_argument("--limit", type=int, default=1000, help="Total filing cap across all companies (default: 1000).")
    p.add_argument("--per-company-limit", type=int, default=60, help="Max filings per company (default: 60).")
    p.add_argument("--rate-delay", type=float, default=0.25, help="Seconds between SEC requests (default: 0.25 = 4/s).")
    p.add_argument("--dry-run", action="store_true", help="Show the download plan without fetching/writing files.")
    args = p.parse_args()

    user_agent = os.environ.get("SEC_EDGAR_USER_AGENT")
    if not user_agent:
        print(
            "ERROR: set SEC_EDGAR_USER_AGENT first, e.g.:\n"
            '  export SEC_EDGAR_USER_AGENT="Your Name your-email@domain.com"\n'
            "SEC EDGAR requires a descriptive User-Agent on every request and will "
            "reject/rate-limit requests without one.",
            file=sys.stderr,
        )
        sys.exit(1)

    fetch_corpus(
        [t.upper() for t in args.tickers],
        years=args.years,
        limit=args.limit,
        per_company_limit=args.per_company_limit,
        rate_delay=args.rate_delay,
        user_agent=user_agent,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
