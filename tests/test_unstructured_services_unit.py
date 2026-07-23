"""
tests/test_unstructured_services_unit.py — behavioral tests for the extracted retrieval services.

Part of the loosely-coupled retrieval refactor (Part A). These services
(RankingService, GraphSeedService, DocumentResolver, LexicalService,
ResponseFormatter) were extracted from mixins/{ranking,graph_seeds,
document_resolver,lexical,policies}.py — that extraction was verified via
parity tests comparing old-mixin-output vs new-service-output while both
existed side by side (all passed, see git history for
scripts/run_rag_eval.py 40/40 confirmation and the original parity test
run). Now that the old mixins are deleted (Part A4), this file tests the
services directly on their own behavioral properties rather than against
a no-longer-existing "old" implementation.

Scope: covers the pure-logic methods (no live Neo4j session required —
query parsing, scoring, merging, keyword/phrase extraction). The Cypher-
querying methods (vector_seed, graph_expand, resolve_document_for_query,
structural_*_retrieve) are covered by the live 40-question eval suite
(scripts/run_rag_eval.py) run against the real docker-compose app, not by
a unit-level DB fixture.

Run with:
    python -m pytest tests/test_unstructured_services_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from src.retrieval.unstructured.services.document_resolver import DocumentResolver
from src.retrieval.unstructured.services.formatter import ResponseFormatter, access_denied_response
from src.retrieval.unstructured.services.graph_seeds import GraphSeedService
from src.retrieval.unstructured.services.lexical import LexicalService
from src.retrieval.unstructured.services.ranking import RankingService

QUESTIONS = [
    "What is the electronic version ISBN of the Go.Data annual report 2021?",
    "Which network deployed fellows to Greece, Malta, Moldova, and Kosovo?",
    "Contrast proximity tracing tools vs. Go.Data as categorized by WHO.",
    "How many countries and territories were supported during 2020-2021?",
    "What does the compliance policy say about whistleblowing procedures?",
]


# ── RankingService ─────────────────────────────────────────────────────


def test_query_keywords_excludes_short_tokens():
    ranking = RankingService()
    kws = ranking._query_keywords("Which network deployed fellows to Greece and Kosovo?")
    assert "to" not in kws  # below the 3-char minimum
    assert "greece" in kws
    assert "kosovo" in kws
    assert len(kws) <= 18  # capped


def test_content_keywords_from_query_includes_anchors_and_bigrams():
    ranking = RankingService()
    kws = ranking._content_keywords_from_query("Go.Data annual report 2021")
    assert "go.data" in kws
    assert "annual report" in kws  # adjacent bigram
    assert len(kws) <= 18  # capped


def test_search_phrases_from_query_prefers_longer_phrases_first():
    ranking = RankingService()
    phrases = ranking._search_phrases_from_query("Greece Malta Moldova Kosovo network deployed fellows")
    assert phrases  # non-empty for a real question
    # Sorted by length descending (longest n-gram phrases before single words).
    assert len(phrases[0]) >= len(phrases[-1])
    assert len(phrases) <= 14  # capped


def test_relevance_boost_rewards_keyword_matches_and_named_sections():
    ranking = RankingService()
    base = ranking._relevance_boost("Page 12", "", [])
    named_with_keywords = ranking._relevance_boost(
        "Findings", "Greece Malta Moldova Kosovo fellows network", ["greece", "network", "fellows"]
    )
    assert named_with_keywords > base


def test_merge_and_rank_dedupes_by_id_and_respects_limit():
    ranking = RankingService()
    vector_hits = [{"id": "s1", "title": "Intro", "text": "hello world", "score": 0.8}]
    fulltext_hits = [{"id": "s1", "title": "Intro", "text": "hello world", "score": 3.2}]  # same id, different source
    graph_hits = [
        {"id": "s3", "title": "Related", "text": "network", "seed_id": "s1", "hops": 1,
         "edge_weight": 0.7, "rel_type": "MENTIONS"}
    ]
    seed_scores = {"s1": 0.8}
    out = ranking._merge_and_rank(
        "test query", vector_hits, fulltext_hits, graph_hits, seed_scores, limit=1
    )
    assert len(out) == 1  # limit respected
    ids = {item["id"] for item in ranking._merge_and_rank(
        "test query", vector_hits, fulltext_hits, graph_hits, seed_scores, limit=5
    )}
    assert len(ids) == len({"s1", "s3"})  # s1 merged from two sources into one entry


def test_merge_retrieval_chunks_dedupes_sorts_and_caps_at_8():
    ranking = RankingService()
    primary = [{"id": "a", "score": 0.9}, {"id": "b", "score": 0.5}]
    extra = [{"id": "b", "score": 0.5}, {"id": "c", "score": 0.7}]
    out = ranking._merge_retrieval_chunks(primary, extra)
    assert [c["id"] for c in out] == ["a", "c", "b"]  # sorted by score desc, "b" deduped not doubled

    many_primary = [{"id": str(i), "score": float(i)} for i in range(10)]
    out_capped = ranking._merge_retrieval_chunks(many_primary, [])
    assert len(out_capped) == 8


def test_contrast_term_groups_only_for_contrast_questions():
    ranking = RankingService()
    assert ranking._contrast_term_groups(QUESTIONS[0]) == []  # not a contrast question
    groups = ranking._contrast_term_groups(QUESTIONS[2])  # "Contrast X vs. Y"
    assert len(groups) == 2  # one group per side of the comparison


# ── GraphSeedService ────────────────────────────────────────────────────


def test_fulltext_query_quotes_multiword_phrases():
    service = GraphSeedService(RankingService())
    q = service.fulltext_query("Which network deployed fellows to Greece and Kosovo?")
    assert '"' in q  # multi-word phrases are quoted for Lucene
    assert " OR " in q


def test_fulltext_query_falls_back_to_truncated_question_with_no_keywords():
    service = GraphSeedService(RankingService())
    assert service.fulltext_query("") == ""


# ── DocumentResolver pure methods ───────────────────────────────────────


def test_document_match_terms_capped_at_6():
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    for q in QUESTIONS:
        terms = resolver.document_match_terms(q)
        assert len(terms) <= 6
        assert "table" not in terms  # generic structural word filtered out


def test_doc_name_terms_prefers_proper_nouns():
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    terms = resolver.doc_name_terms("Which network deployed fellows to Greece and Kosovo?")
    assert "greece" in terms
    assert "kosovo" in terms
    assert len(terms) <= 6


def test_doc_name_terms_strips_trailing_possessive():
    """Regression: "JPMorgan's" extracted whole (with the apostrophe-s)
    never matches the document's own text, which says "JPMorgan Chase &
    Co." or "the Firm", not the possessive form -- CONTAINS scored zero
    everywhere and the query's one real anchor silently contributed
    nothing. Verified live: "What does Item 9A report about the
    effectiveness of JPMorgan's internal controls?" resolved to an
    unrelated WHO report instead of the JPM 10-K that has that exact
    section, purely because the possessive form matched nothing."""
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    terms = resolver.doc_name_terms("What does JPMorgan's 10-K say about risk?")
    assert "jpmorgan" in terms
    assert "jpmorgan's" not in terms


def test_document_match_terms_strips_trailing_possessive():
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    terms = resolver.document_match_terms(
        "What does Item 9A report about the effectiveness of JPMorgan's internal controls?"
    )
    assert "jpmorgan" in terms
    assert "jpmorgan's" not in terms


def test_doc_name_terms_excludes_generic_long_words():
    """Regression: a prior version fell back to "any word >= 6 chars that
    isn't a stopword" as a document-name candidate. In a multi-document
    corpus with one much larger document, ordinary content vocabulary
    ("employees", "conflicts", "benefits", "discussed") isn't distinctive —
    it just appears more often in whichever document has the most content —
    so resolve_document_for_query_strict's raw-occurrence scoring picked
    the wrong document purely because it was bigger. Verified live: an
    83-section policy document's own "how can I avoid conflicts when it
    comes to gifts and benefits" question resolved to an unrelated
    638-section 10-K instead, before this fix."""
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    terms = resolver.doc_name_terms(
        "How can employees avoid conflicts when it comes to gifts and benefits?"
    )
    assert terms == []

    terms = resolver.doc_name_terms("What is discussed on page 6 of this document?")
    assert terms == []


def test_doc_name_terms_excludes_structural_references_and_their_glosses():
    """Regression: a structural reference like "Note 3 (Commitments and
    Contingencies)" names a location WITHIN the document already under
    discussion, not a different document. Standard footnote/item titles are
    boilerplate shared across most filings in a corpus, so left unstripped
    the mid-sentence-capitalization scan picked them up as document-naming
    anchors and resolve_document_for_query_strict's raw-occurrence scoring
    matched whichever unrelated document merely used those generic terms
    more. Verified live: this single-handedly overrode a correct
    conversation document hint on an AMZN 10-Q, resolving to an unrelated
    JPM 10-K instead purely because JPM's "Note 3" happens to also be
    titled with common financial/legal vocabulary."""
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    assert resolver.doc_name_terms("What does Note 3 (Commitments and Contingencies) discuss?") == []
    assert resolver.doc_name_terms("What is Box 9 about?") == []
    # Letter-suffixed SEC item numbering ("Item 9A") must strip the same way.
    assert resolver.doc_name_terms("What does Item 9A (Controls and Procedures) report?") == []

    # A real proper noun alongside a structural reference must still surface.
    terms = resolver.doc_name_terms(
        "What does Note 7 (Segment Information) report about Amazon's business segments?"
    )
    assert "amazon" in terms or "amazon's" in terms


def test_logical_id_from_node_id_extracts_prefix():
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    assert resolver._logical_id_from_node_id("doc_rag_document:r1::section_1_2") == "doc_rag_document"
    assert resolver._logical_id_from_node_id("plain_id") == "plain_id"
    assert resolver._logical_id_from_node_id("") is None


# ── LexicalService ──────────────────────────────────────────────────────


def test_enrich_chunk_text_for_facts_appends_extracted_urls():
    resolver = DocumentResolver(GraphSeedService(RankingService()))
    lexical = LexicalService(RankingService(), resolver)

    plain = lexical.enrich_chunk_text_for_facts("Plain", "No links here.")
    assert plain == "No links here."

    with_url = lexical.enrich_chunk_text_for_facts(
        "Contacts", "Visit https://example.org/go-data for more info."
    )
    assert "[Extracted URLs]" in with_url
    assert "https://example.org/go-data" in with_url


# ── ResponseFormatter / access_denied_response ──────────────────────────


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
