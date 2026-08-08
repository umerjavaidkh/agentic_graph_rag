"""page_validation.py — page-level coverage scoring, the same "measured,
not assumed" idea as ontology_validation.py but scoped down to a single
page instead of the whole document: a document can pass every document-
level check (text/NER/embedding coverage, page-number continuity) while
individual pages inside it collapsed to near-empty content, and nothing
today surfaces which ones.

Pure scoring logic only -- no Neo4j/blob/parser calls live here, so it's
testable with plain strings; document/page_report.py is the impure wiring
layer that re-parses the stored source and queries Neo4j, then calls in
here per page.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models import DKGNode
    from .ir import DocumentIR

# Below this fraction of the raw page's word count actually landing in the
# graph, a page is flagged for re-extraction rather than silently passed.
PAGE_COVERAGE_PASS_THRESHOLD = 0.5


def _word_count(text: str) -> int:
    return len((text or "").split())


@dataclass
class PageReport:
    page_number: int
    raw_chars: int
    graph_chars: int
    coverage: float
    entity_count: int
    edge_count: int
    status: str  # PASS | RE_EXTRACT_REQUIRED | REPARSE_MISMATCH

    def as_dict(self) -> dict[str, Any]:
        return {
            "page_number": self.page_number,
            "raw_chars": self.raw_chars,
            "graph_chars": self.graph_chars,
            "coverage": round(self.coverage, 4),
            "entity_count": self.entity_count,
            "edge_count": self.edge_count,
            "status": self.status,
        }


def score_page(
    *,
    page_number: int,
    raw_text: str,
    graph_text: str,
    entity_count: int,
    edge_count: int,
) -> PageReport:
    """One page's coverage: how much of its raw extracted text actually
    made it into the graph. Word-count ratio, not char-count -- robust to
    whitespace/formatting cleanup the structural builder applies, same
    tolerance reasoning as ontology_validation's fuzzy title match."""
    raw_words = _word_count(raw_text)
    graph_words = _word_count(graph_text)

    if raw_words == 0:
        # Nothing on the source page to cover. Graph text existing anyway
        # means the raw re-parse and what's in the graph disagree on this
        # page's content (different page count/offset between the stored
        # source and the ingested revision) -- worth flagging, not a pass.
        coverage = 1.0
        status = "PASS" if graph_words == 0 else "REPARSE_MISMATCH"
    else:
        coverage = min(1.0, graph_words / raw_words)
        status = "PASS" if coverage >= PAGE_COVERAGE_PASS_THRESHOLD else "RE_EXTRACT_REQUIRED"

    return PageReport(
        page_number=page_number,
        raw_chars=len(raw_text or ""),
        graph_chars=len(graph_text or ""),
        coverage=coverage,
        entity_count=entity_count,
        edge_count=edge_count,
        status=status,
    )


def summarize_pages(pages: list[PageReport]) -> dict[str, Any]:
    if not pages:
        return {"page_count": 0, "avg_coverage": 0.0, "pages_failing": 0, "requires_reprocessing": False}
    avg_coverage = sum(p.coverage for p in pages) / len(pages)
    failing = [p for p in pages if p.status != "PASS"]
    return {
        "page_count": len(pages),
        "avg_coverage": round(avg_coverage, 4),
        "pages_failing": len(failing),
        "requires_reprocessing": len(failing) > 0,
    }


def check_construction_coverage(ir: "DocumentIR", nodes: list["DKGNode"]) -> dict[str, Any]:
    """Runs immediately after Axis1StructuralBuilder.build(), while `ir`
    (the raw per-page extraction) and `nodes` (what the structural builder
    actually produced) are both still in hand -- catches a total or
    partial structural collapse (e.g. every heading heuristic missing on a
    uniform-font document, so the whole document becomes one "Preamble"
    section and a run of near-empty Page nodes) at construction time
    instead of only when someone happens to open the on-demand page-
    validation panel later.

    Needs no re-parse and no Neo4j round trip -- unlike page_report.py's
    wiring (which re-parses a stored source file after the fact), this
    reuses the exact `ir`/`nodes` that this ingestion run already produced,
    so it can't disagree with what actually got ingested. `entity_count`/
    `edge_count` are always 0 here: Axis-2 hasn't run yet at this point in
    the pipeline, so those columns are meaningless for this pass (still
    reported for shape-compatibility with the page_report.py callers that
    read this same PageReport shape after Axis-2 has run)."""
    from ..models import NodeType

    raw_text_by_page = {p.page: p.text for p in ir.pages}
    graph_text_by_page = {
        n.page_start: (n.text or "")
        for n in nodes
        if n.type in (NodeType.PAGE, NodeType.PAGE.value)
    }
    all_page_numbers = sorted(set(raw_text_by_page) | set(graph_text_by_page))
    pages = [
        score_page(
            page_number=pno,
            raw_text=raw_text_by_page.get(pno, ""),
            graph_text=graph_text_by_page.get(pno, ""),
            entity_count=0,
            edge_count=0,
        )
        for pno in all_page_numbers
    ]
    return {"pages": [p.as_dict() for p in pages], "summary": summarize_pages(pages)}
