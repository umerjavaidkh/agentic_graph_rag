#!/usr/bin/env python3
"""
audit_language.py — what the language-share threshold should actually be.

`LANGUAGE_SHARE_THRESHOLD` decides whether a document goes to the default
language or to another one. It ships at 0.05, and 0.05 is a guess: the
design says outright that the right value has to be measured against real
bilingual documents rather than picked in advance.

This measures it. Two failure modes bound the answer from opposite sides,
and they are not symmetric:

  * TOO LOW  -- a scanned English document whose OCR emits a handful of
    stray Arabic-looking glyphs crosses the line and leaves the English
    corpus. No English query reaches it again, and nothing looks broken.
  * TOO HIGH -- a genuinely bilingual document stays in English. Its
    Arabic half is then unreachable by an Arabic query.

The first is worse, because it moves documents nobody was thinking about
and the symptom is silence. So the number wants to sit above the noise
floor of English documents and below the share of real bilingual ones,
and the gap between those two populations is what this prints.

Reads PDFs directly rather than the ingested graph, so it can be run
before deciding whether to ingest anything.

Usage:
    python scripts/audit_language.py FILE.pdf [FILE.pdf ...]
    python scripts/audit_language.py --dir sample_data_to_test/unstructured/corpus10
    python scripts/audit_language.py --dir DIR --per-page
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.shared.config.settings import LANGUAGE_SHARE_THRESHOLD  # noqa: E402
from src.shared.language import detect_language, script_shares  # noqa: E402


def _page_texts(path: Path) -> list[str]:
    """Page text, by whichever extractor this environment has.

    Deliberately not the project's parser: the question here is what the
    document contains, not how well the pipeline reads it, and a parser
    bug would be indistinguishable from a language signal.
    """
    try:
        import fitz  # PyMuPDF

        with fitz.open(path) as doc:
            return [page.get_text() or "" for page in doc]
    except ImportError:
        pass
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            return [(p.extract_text() or "") for p in pdf.pages]
    except ImportError:
        raise SystemExit("need PyMuPDF or pdfplumber to read PDFs")


def _report(path: Path, per_page: bool) -> tuple[str, dict[str, float]]:
    pages = _page_texts(path)
    full = "\n".join(pages)
    shares = script_shares(full)
    verdict = detect_language(full)

    top = sorted(shares.items(), key=lambda kv: -kv[1])
    summary = ", ".join(f"{code}={share:.4f}" for code, share in top) or "none"
    print(f"\n{path.name}")
    print(f"  pages={len(pages)}  letters-scored-from={len(full):,} chars")
    print(f"  non-default shares: {summary}")
    print(f"  detect_language -> {verdict}   (threshold {LANGUAGE_SHARE_THRESHOLD})")

    if per_page and pages:
        by_page = [script_shares(t) for t in pages]
        for code, _ in top:
            vals = sorted(s.get(code, 0.0) for s in by_page)
            n = len(vals)
            q = lambda p: vals[min(n - 1, int(p * n))]  # noqa: E731
            print(
                f"    {code} per page: min={vals[0]:.3f} p25={q(.25):.3f} "
                f"median={statistics.median(vals):.3f} p75={q(.75):.3f} max={vals[-1]:.3f}"
            )
            for bound in (0.01, 0.05, 0.25, 0.50, 0.90):
                hits = sum(1 for v in vals if v > bound)
                print(f"      pages over {bound:<5}: {hits:4}/{n}")
    return verdict, shares


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="*", type=Path)
    parser.add_argument("--dir", type=Path, help="every .pdf under this directory")
    parser.add_argument("--per-page", action="store_true", help="per-page distribution")
    args = parser.parse_args()

    paths = list(args.files)
    if args.dir:
        paths += sorted(args.dir.rglob("*.pdf"))
    if not paths:
        parser.error("give some PDFs, or --dir")

    results = []
    for path in paths:
        try:
            results.append((path, *_report(path, args.per_page)))
        except Exception as exc:  # a corrupt file must not end the audit
            print(f"\n{path.name}\n  SKIPPED: {exc}")

    # The gap between the two populations is the whole point: the threshold
    # belongs inside it, and if there is no gap no threshold can separate them.
    print("\n" + "=" * 64)
    non_default = [max(sh.values(), default=0.0) for _, verdict, sh in results if verdict != "en"]
    default = [max(sh.values(), default=0.0) for _, verdict, sh in results if verdict == "en"]
    print(f"documents read: {len(results)}   non-default: {len(non_default)}   default: {len(default)}")
    if default:
        print(f"  highest non-default share among DEFAULT-language docs: {max(default):.4f}  <- noise floor")
    if non_default:
        print(f"  lowest  non-default share among OTHER-language docs:   {min(non_default):.4f}  <- signal floor")
    if default and non_default:
        floor, signal = max(default), min(non_default)
        if signal > floor:
            print(f"  GAP: {floor:.4f} .. {signal:.4f}   threshold belongs inside it")
            print(f"  suggested: {(floor + signal) / 2:.4f}   (current: {LANGUAGE_SHARE_THRESHOLD})")
        else:
            print("  NO GAP: the populations overlap on this metric.")
            print("  A share threshold cannot separate them; the rule needs another signal")
            print("  (per-page share, or a run-length test) before it can be trusted.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
