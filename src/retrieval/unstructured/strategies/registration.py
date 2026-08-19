"""registration.py — builds the shared service bundle and registers unstructured strategies.

Imported once (by mixins/hybrid.py) as a side-effecting bootstrap step, so
`get_unstructured(key)` has something to resolve. Strategies are added here
incrementally as each is migrated (Part A2/A3) — this is not the final
shape; A4 moves this bundle-building into DocumentRAGRetriever itself once
all strategies have landed and the mixins are retired.

The shared services hold no per-request state (no driver, no cached
session) — every method takes `session`/`tenant_id` explicitly — so a
single process-wide instance of each is safe to reuse across every strategy
and every request.
"""
from __future__ import annotations

from ....shared.neo4j.driver import get_neo4j_driver
from ...strategy_registry import register_unstructured
from ..executor import DocumentQueryExecutor
from ..services.chapter_summary import ChapterSummaryService
from ..services.document_resolver import DocumentResolver
from ..services.formatter import ResponseFormatter
from ..services.graph_seeds import GraphSeedService
from ..services.lexical import LexicalService
from ..services.ranking import RankingService
from ..services.reranker import RerankerService
from .box import BoxStrategy
from .filing_date import FilingDateStrategy
from .full_hybrid import FullHybridStrategy
from .page import PageStrategy
from .subsection import SubsectionStrategy
from .toc import TocStrategy

ranking = RankingService()
graph_seeds = GraphSeedService(ranking)
document_resolver = DocumentResolver(graph_seeds)
lexical = LexicalService(ranking, document_resolver)
chapter_summaries = ChapterSummaryService()
formatter = ResponseFormatter()
reranker = RerankerService()
exec_ = DocumentQueryExecutor()

register_unstructured("subsection_tree", lambda: SubsectionStrategy(document_resolver, formatter, exec_))
register_unstructured("structural_box_list", lambda: BoxStrategy(document_resolver, formatter, exec_))
register_unstructured("structural_page", lambda: PageStrategy(document_resolver, formatter))
register_unstructured("structural_toc", lambda: TocStrategy(document_resolver, formatter, exec_))
register_unstructured("structural_filing_date", lambda: FilingDateStrategy(document_resolver, formatter))
register_unstructured(
    "graph_rag_hybrid",
    lambda: FullHybridStrategy(
        get_neo4j_driver(),
        ranking,
        graph_seeds,
        lexical,
        formatter,
        document_resolver,
        chapter_summaries,
        reranker=reranker,
    ),
)
