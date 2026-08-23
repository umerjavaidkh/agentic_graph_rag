"""query_plan.py — classify a question by SHAPE, and emit the plan for it.

Domains differ in vocabulary; query shapes do not. "What does section 4.2
say" and "what does clause 8 say" are the same retrieval problem, and a
router built on shapes serves every document type the corpus holds.

The plan is a value object, deliberately: everything above it is judgment,
everything below it is deterministic execution. A query can be replayed by
replaying its plan, which is the property that was missing when every
quality gate read green through real failures.

Classification here is regex over phrasing, not an LLM call. It costs
nothing, it is deterministic, and it reuses the predicates the strategies
were already using individually -- the change is that the decision now
happens once, in one place, instead of five times across five strategies
that each resolved their own document.

Two rules matter more than the routing itself, and both come from measured
failures on this corpus:

  * An ENUMERATIVE question must never be answered from a top-k cut. Asked
    for the table of contents of a 9-chapter, 21-section document, the
    hybrid path returned 8 headings out of 30, out of order, and phrased
    with full confidence. `LIMIT 6` on the lexical queries did that.

  * A STRUCTURAL question must never go through the vector index. "Section
    4.2" is a graph address, not a semantic concept, and the nearest
    neighbours of an address are other addresses. Worse here than in
    general: Page nodes are 1.0% embedded and Region nodes 0%, so the
    vector index cannot see 51% of this corpus's content at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .query_intent import (
    is_enumeration_question,
    is_overview_question,
    is_synthesis_question,
    is_toc_question,
)


class Shape(str, Enum):
    STRUCTURAL = "structural"      # a graph address: section/page/box/figure N
    ENUMERATIVE = "enumerative"    # "every", "all", "list" -- exhaustive
    THEMATIC = "thematic"          # "overall", "main risks" -- whole-document
    FACTOID = "factoid"            # the default: hybrid recall + rerank


# "Table of contents" is enumerative AND structural: it asks for the whole
# hierarchy, in document order. It is listed here rather than inferred so
# the intent is explicit at the call site.
_TOC_LIKE = re.compile(r"\b(table\s+of\s+contents|contents\s+page|outline\s+of)\b", re.I)

# A reference to a numbered structural unit. Deliberately broad across the
# vocabularies the corpus actually contains -- section/clause/article for
# legal and standards documents, box/figure/table for reports.
_ADDRESS = re.compile(
    r"\b(?:section|clause|article|chapter|part|annex|appendix|box|figure|fig\.?|table|page)\s*"
    r"(?:no\.?\s*)?\d+(?:\.\d+)*\b",
    re.I,
)


# The shared `is_enumeration_question` requires the literal "list all" /
# "enumerate" / "name all" set, so "list every clause mentioning indemnity"
# fell through to a top-k factoid plan -- the exact failure the exhaustive
# rule exists to prevent. Supplemented here rather than widened in
# query_intent.py, because that predicate also drives ranking weights and
# fetch limits elsewhere; broadening it would change behaviour for shapes
# this router does not own yet.
_EXHAUSTIVE_ASK = re.compile(
    r"\b(?:list|show|find|give\s+me|identify|extract)\b[^.?]*\b(?:every|all|each)\b"
    r"|\bevery\s+\w+\s+(?:that|which|mentioning|containing|with|referring)\b"
    r"|\ball\s+(?:the\s+)?\w+s\s+(?:that|which|mentioning|containing|with)\b",
    re.I,
)


@dataclass(frozen=True)
class RetrievalPlan:
    """What to run, how wide, and how to order it. Replayable by construction."""

    shape: Shape
    use_vectors: bool
    exhaustive: bool                  # no top-k truncation; recall is the contract
    limit: int
    document_order: bool              # order by position, not relevance score
    address: Optional[str] = None     # the structural unit named, if any
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_structural(self) -> bool:
        return self.shape is Shape.STRUCTURAL


# An exhaustive answer must be able to hold a whole document's hierarchy.
# This corpus's largest document has 887 content nodes; the cap exists to
# bound a pathological query, not to trim a normal one.
EXHAUSTIVE_LIMIT = 1000


def classify(query: str, *, default_limit: int = 8) -> RetrievalPlan:
    """The shape of `query`, and the plan that fits it."""
    q = query or ""

    if _TOC_LIKE.search(q):
        return RetrievalPlan(
            shape=Shape.STRUCTURAL,
            use_vectors=False,
            exhaustive=True,
            limit=EXHAUSTIVE_LIMIT,
            document_order=True,
            notes=("table-of-contents: whole hierarchy, document order, no truncation",),
        )

    address = _ADDRESS.search(q)
    if address:
        return RetrievalPlan(
            shape=Shape.STRUCTURAL,
            use_vectors=False,
            exhaustive=False,
            limit=default_limit,
            document_order=True,
            address=address.group(0),
            notes=("named a structural address; resolved on the hierarchy, not by similarity",),
        )

    if is_enumeration_question(q) or _EXHAUSTIVE_ASK.search(q):
        return RetrievalPlan(
            shape=Shape.ENUMERATIVE,
            use_vectors=True,
            exhaustive=True,
            limit=EXHAUSTIVE_LIMIT,
            document_order=True,
            notes=("exhaustive: a top-k cut here answers 3 of 17 and sounds certain",),
        )

    # The shared `is_toc_question` also fires on "all sections ...", which
    # is an enumeration with a content filter, not a request for the
    # hierarchy. It is checked below the enumerative branch for that reason:
    # routing it as structural would switch the vector channel off on a
    # query whose whole point is matching "mention risk" semantically.
    if is_toc_question(q):
        return RetrievalPlan(
            shape=Shape.STRUCTURAL,
            use_vectors=False,
            exhaustive=True,
            limit=EXHAUSTIVE_LIMIT,
            document_order=True,
            notes=("hierarchy request (weaker signal than explicit TOC phrasing)",),
        )

    if is_synthesis_question(q) or is_overview_question(q):
        return RetrievalPlan(
            shape=Shape.THEMATIC,
            use_vectors=True,
            exhaustive=False,
            limit=max(default_limit, 16),
            document_order=True,
            notes=("whole-document: wants breadth over depth",),
        )

    return RetrievalPlan(
        shape=Shape.FACTOID,
        use_vectors=True,
        exhaustive=False,
        limit=default_limit,
        document_order=False,
        notes=("default hybrid recall, reranked",),
    )
