"""
tests/test_unstructured_services_unit.py — parity between extracted services and their source mixins.

Part of the loosely-coupled retrieval refactor (Part A). These services were
extracted verbatim from mixins/{ranking,graph_seeds,document_resolver,
lexical}.py — this file proves the extraction didn't silently change
behavior, by calling both the old mixin method and the new service method
with identical inputs and asserting identical output.

Scope: covers the pure-logic methods (no live Neo4j session required —
query parsing, scoring, merging, keyword/phrase extraction). The Cypher-
querying methods (vector_seed, graph_expand, resolve_document_for_query,
structural_*_retrieve) are copied verbatim too, but their real regression
gate is the live 40-question eval suite (scripts/run_rag_eval.py), run
against the real docker-compose app once each strategy is wired in — a
unit-level DB fixture would either need a live Neo4j or a hand-built fake
graph, neither of which would catch more than the eval suite already does
for this kind of "same Cypher, moved file" extraction.

Run with:
    python -m pytest tests/test_unstructured_services_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

# Drop any stale fake stub a previously-collected test file may have left in
# sys.modules (several test files stub src.auth/src.graph/src.retrieval/
# neo4j/fastapi as a bare types.ModuleType for their own narrow needs and
# never restore them) — this file needs the real packages. A real module
# always has __file__ or __path__; a hand-built stub has neither. Imports
# below are all done at module top (not deferred into test functions) so
# this one cleanup, run once at collection time, protects every symbol this
# file needs — a deferred/function-local import would re-resolve via
# whatever sys.modules state exists when that specific test *runs* (after
# other test files' own run-phase side effects), not just at collection.
for _mod_name in list(sys.modules):
    if (
        _mod_name == "src.auth"
        or _mod_name.startswith("src.auth.")
        or _mod_name == "src.graph"
        or _mod_name.startswith("src.graph.")
        or _mod_name == "src.retrieval"
        or _mod_name.startswith("src.retrieval.")
        or _mod_name in ("neo4j", "neo4j.exceptions", "fastapi")
    ):
        _mod = sys.modules[_mod_name]
        if getattr(_mod, "__file__", None) is None and getattr(_mod, "__path__", None) is None:
            del sys.modules[_mod_name]

from src.retrieval.unstructured.mixins.ranking import RankingMixin
from src.retrieval.unstructured.mixins.document_resolver import DocumentResolverMixin
from src.retrieval.unstructured.mixins.graph_seeds import GraphSeedsMixin
from src.retrieval.unstructured.mixins.lexical import LexicalRetrievalMixin
from src.retrieval.unstructured.services.ranking import RankingService
from src.retrieval.unstructured.services.graph_seeds import GraphSeedService
from src.retrieval.unstructured.services.document_resolver import DocumentResolver
from src.retrieval.unstructured.services.lexical import LexicalService
from src.retrieval.unstructured.services.formatter import ResponseFormatter, access_denied_response


class _OldRanking(RankingMixin):
    pass


class _OldDocumentResolver(DocumentResolverMixin):
    pass


QUESTIONS = [
    "What is the electronic version ISBN of the Go.Data annual report 2021?",
    "Which network deployed fellows to Greece, Malta, Moldova, and Kosovo?",
    "Contrast proximity tracing tools vs. Go.Data as categorized by WHO.",
    "How many countries and territories were supported during 2020-2021?",
    "What does the compliance policy say about whistleblowing procedures?",
]


# ── RankingService vs RankingMixin ────────────────────────────────────────


def test_query_keywords_parity():
    old, new = _OldRanking(), RankingService()
    for q in QUESTIONS:
        assert old._query_keywords(q) == new._query_keywords(q)


def test_content_keywords_from_query_parity():
    old, new = _OldRanking(), RankingService()
    for q in QUESTIONS:
        assert old._content_keywords_from_query(q) == new._content_keywords_from_query(q)


def test_search_phrases_from_query_parity():
    old, new = _OldRanking(), RankingService()
    for q in QUESTIONS:
        assert old._search_phrases_from_query(q) == new._search_phrases_from_query(q)


def test_relevance_boost_parity():
    old, new = _OldRanking(), RankingService()
    cases = [
        ("Introduction", "Some section text about Go.Data deployment.", ["deployment", "go.data"]),
        ("Page 12", "", []),
        ("Findings", "Greece Malta Moldova Kosovo fellows network", ["greece", "network", "fellows"]),
    ]
    for title, text, keywords in cases:
        assert old._relevance_boost(title, text, keywords) == new._relevance_boost(title, text, keywords)


def test_merge_and_rank_parity():
    old, new = _OldRanking(), RankingService()
    vector_hits = [{"id": "s1", "title": "Intro", "text": "hello world", "score": 0.8}]
    fulltext_hits = [{"id": "s2", "title": "Body", "text": "greece kosovo", "score": 3.2}]
    graph_hits = [
        {"id": "s3", "title": "Related", "text": "network", "seed_id": "s1", "hops": 1,
         "edge_weight": 0.7, "rel_type": "MENTIONS"}
    ]
    seed_scores = {"s1": 0.8}
    for q in QUESTIONS:
        old_out = old._merge_and_rank(q, vector_hits, fulltext_hits, graph_hits, seed_scores, limit=5)
        new_out = new._merge_and_rank(q, vector_hits, fulltext_hits, graph_hits, seed_scores, limit=5)
        assert old_out == new_out


def test_merge_retrieval_chunks_parity():
    old, new = _OldRanking(), RankingService()
    primary = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.5}]
    extra = [{"id": "b", "score": 0.5}, {"id": "c", "score": 0.7}]
    assert old._merge_retrieval_chunks(primary, extra) == new._merge_retrieval_chunks(primary, extra)


def test_contrast_term_groups_parity():
    old, new = _OldRanking(), RankingService()
    for q in QUESTIONS:
        assert old._contrast_term_groups(q) == new._contrast_term_groups(q)


# ── GraphSeedService.fulltext_query vs GraphSeedsMixin._fulltext_query ────


def test_fulltext_query_parity():
    class _OldGraphSeeds(RankingMixin, GraphSeedsMixin):
        pass

    old = _OldGraphSeeds()
    new = GraphSeedService(RankingService())
    for q in QUESTIONS:
        assert old._fulltext_query(q) == new.fulltext_query(q)


# ── DocumentResolver pure methods vs DocumentResolverMixin ────────────────


def test_document_match_terms_parity():
    old, new = _OldDocumentResolver(), DocumentResolver(GraphSeedService(RankingService()))
    for q in QUESTIONS:
        assert old._document_match_terms(q) == new.document_match_terms(q)


def test_doc_name_terms_parity():
    old, new = _OldDocumentResolver(), DocumentResolver(GraphSeedService(RankingService()))
    for q in QUESTIONS:
        assert old._doc_name_terms(q) == new.doc_name_terms(q)


def test_logical_id_from_node_id_parity():
    old, new = _OldDocumentResolver(), DocumentResolver(GraphSeedService(RankingService()))
    ids = ["doc_rag_document:r1::section_1_2", "plain_id", "", "a:b:c"]
    for nid in ids:
        assert old._logical_id_from_node_id(nid) == new._logical_id_from_node_id(nid)


# ── LexicalService.enrich_chunk_text_for_facts vs LexicalRetrievalMixin ───


def test_enrich_chunk_text_for_facts_parity():
    class _OldLexical(LexicalRetrievalMixin):
        pass

    old = _OldLexical()
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    new = LexicalService(RankingService(), resolver)
    cases = [
        ("Contacts", "Visit https://example.org/go-data for more info."),
        ("Plain", "No links here."),
        ("Multi", "See https://a.example and https://b.example for details."),
    ]
    for title, text in cases:
        assert old._enrich_chunk_text_for_facts(title, text) == new.enrich_chunk_text_for_facts(title, text)


# ── ResponseFormatter / access_denied_response vs PoliciesMixin ──────────


def test_format_response_matches_shape():
    from unittest.mock import MagicMock

    formatter = ResponseFormatter()
    ctx = MagicMock(user_id="public_001")
    ctx.role.value = "public"
    items = [
        {"id": "s1", "title": "Intro", "text": "hello", "score": 0.812345, "related": ["x"]},
        {"id": "s2", "title": "Page 3", "text": "world", "score": 0.5, "related": [], "pdf_page": 3},
    ]
    out = formatter.format("q", items, ctx=ctx)
    assert out["query"] == "q"
    assert out["total_available"] == 2
    assert out["_access_level"] == "public"
    assert out["_user_id"] == "public_001"
    assert out["chunks"][0]["score"] == 0.812
    assert out["chunks"][1]["pdf_page"] == 3
    assert "pdf_page" not in out["chunks"][0]


def test_access_denied_response_none_when_allowed():
    from unittest.mock import MagicMock

    rbac = MagicMock()
    rbac.can_query_knowledge_area.return_value = True
    ctx = MagicMock(user_id="admin_001")
    assert access_denied_response(rbac, "q", ctx) is None


def test_access_denied_response_dict_when_denied():
    from unittest.mock import MagicMock

    rbac = MagicMock()
    rbac.can_query_knowledge_area.return_value = False
    ctx = MagicMock(user_id="public_001")
    ctx.role.value = "public"
    out = access_denied_response(rbac, "q", ctx)
    assert out is not None
    assert out["chunks"][0]["id"] == "access_denied"
    assert out["_user_id"] == "public_001"
