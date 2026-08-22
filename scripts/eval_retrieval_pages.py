#!/usr/bin/env python
"""Retrieval-only accuracy: does the retriever return the page the answer is on?

No chat model and no LLM judge -- one embedding call per question, which is
where the cost stops. This measures the half of answer quality that
generation cannot fix: if the right page is never retrieved, no model can
answer from it.

Ground truth is read out of the PDF at run time rather than written down
here. A phrase that cannot be located, or that appears on more than a few
pages, is reported as UNGROUNDED and excluded from the score instead of
being scored against a page someone guessed.

    python scripts/eval_retrieval_pages.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import fitz  # noqa: E402

from src.shared.auth.roles import Role, UserContext  # noqa: E402
from src.unstructured.retrieval.retriever import DocumentRAGRetriever  # noqa: E402

CORPUS = Path("sample_data_to_test/unstructured/corpus10")
SUITE = Path("eval/corpus10_retrieval_suite.json")
#: A phrase on more than this many pages does not identify a page.
MAX_GROUND_TRUTH_PAGES = 4


def pages_containing(pdf: Path, phrase: str) -> list[int]:
    with fitz.open(pdf) as doc:
        return [i + 1 for i, page in enumerate(doc)
                if phrase.lower() in page.get_text().lower()]


def retrieved_pages(result: dict) -> set[int]:
    """Every page number the retrieval result points at, however nested."""
    pages: set[int] = set()

    def walk(obj):
        if isinstance(obj, dict):
            for key in ("pdf_page", "page_start", "page"):
                value = obj.get(key)
                if isinstance(value, int):
                    pages.add(value)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)

    walk(result)
    return pages


def main() -> int:
    cases = json.loads(SUITE.read_text())["cases"]
    retriever = DocumentRAGRetriever()
    context = UserContext(user_id="admin_001", role=Role.ADMIN, tenant_id="default")

    hits = misses = ungrounded = 0
    print(f"{'doc':<24}{'expected':<12}{'retrieved':<28}result")
    for case in cases:
        pdf = CORPUS / f"{case['doc']}.pdf"
        truth = pages_containing(pdf, case["answer_phrase"]) if pdf.exists() else []
        if not truth or len(truth) > MAX_GROUND_TRUTH_PAGES:
            ungrounded += 1
            why = "not in pdf" if not truth else f"on {len(truth)} pages"
            print(f"{case['doc']:<24}{'-':<12}{'-':<28}UNGROUNDED ({why})")
            continue

        result = retriever.hybrid_retrieve(case["q"], user_context=context)
        got = retrieved_pages(result)
        ok = bool(got & set(truth))
        hits += ok
        misses += not ok
        shown = ",".join(str(p) for p in sorted(got)[:8]) or "(none)"
        print(f"{case['doc']:<24}{str(truth)[:11]:<12}{shown[:27]:<28}{'HIT' if ok else 'MISS'}")

    scored = hits + misses
    print(f"\n{hits}/{scored} pages retrieved correctly"
          f"{f' ({100*hits//scored}%)' if scored else ''}"
          f" · {ungrounded} ungrounded (excluded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
