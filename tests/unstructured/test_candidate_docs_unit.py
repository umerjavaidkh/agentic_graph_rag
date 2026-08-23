"""tests/unstructured/test_candidate_docs_unit.py — vector-first document scoping.

The property under test is the one the graph resolver lost as the corpus
grew: the cost and the ranking of "which documents is this about" must not
depend on how many documents exist.

Run with:
    python -m pytest tests/unstructured/test_candidate_docs_unit.py -v
"""
from __future__ import annotations

from src.unstructured.retrieval.services.candidate_docs import (
    MIN_RELATIVE_SCORE,
    PROBE_K,
    CandidateDocService,
)


class _FakeStore:
    """Returns a fixed ranked list, and records the top_k it was asked for."""

    def __init__(self, hits):
        self._hits = hits
        self.calls: list[dict] = []

    def query_with_docs(self, embedding, top_k=10, *, filters=None):
        self.calls.append({"top_k": top_k, "filters": filters})
        return self._hits[:top_k]


def _svc(hits):
    store = _FakeStore(hits)
    return CandidateDocService(embed=lambda t: [0.0], vector_store=store), store


def _hit(doc, rank):
    return (f"{doc}:rev1::node{rank}", 1.0 - rank / 1000.0, doc)


def _ranked(placements, total=None):
    """A ranked hit list where each document sits at the ranks given.

    Rank is position in the returned list -- that is what the service reads
    -- so gaps are filled with one-hit filler documents. Building fixtures
    any other way lets an intended rank silently differ from the real one.
    """
    at = {r: doc for doc, ranks in placements.items() for r in ranks}
    total = total if total is not None else max(at) + 1
    return [_hit(at.get(r, f"doc_filler_{r}"), r) for r in range(total)]


def test_one_strong_hit_beats_many_weak_ones():
    """The Go.Data failure, in miniature.

    A large document mentioning the topic many times but never well must not
    outrank the document that actually answers the question. Summing
    reciprocal ranks gets this wrong -- sixty mediocre chunks out-total one
    excellent chunk -- which is why the document scores on its best chunk.
    """
    svc, _ = _svc(_ranked({"doc_right": [0], "doc_bulky": list(range(40, 100))}))

    ranked = svc.candidates("go.data annual report")

    assert ranked[0].document_id == "doc_right"
    assert ranked[0].hits == 1


def test_volume_cannot_overtake_a_clearly_better_best_chunk():
    """Corroboration breaks near-ties; it must not reverse a clear ordering."""
    svc, _ = _svc(_ranked({"doc_right": [0], "doc_bulky": list(range(30, 200))}))

    assert svc.candidates("q")[0].document_id == "doc_right"


def test_corroboration_separates_documents_with_comparable_best_chunks():
    """Several good chunks are better evidence than one, all else equal."""
    svc, _ = _svc(_ranked({"doc_thin": [0], "doc_corroborated": list(range(1, 30))}))

    assert svc.candidates("q")[0].document_id == "doc_corroborated"


def test_probe_k_is_constant_so_cost_does_not_track_corpus_size():
    """The pass must ask for a fixed number of chunks, never a share of the corpus."""
    svc, store = _svc([_hit("doc_a", r) for r in range(500)])

    svc.candidates("anything")

    assert store.calls[0]["top_k"] == PROBE_K


def test_result_is_unchanged_by_documents_beyond_the_probe():
    """Ingesting more documents must not change the answer for an unrelated query.

    The same top-ranked chunks are returned; thousands of further documents
    rank below them and are never examined.
    """
    core = [_hit("doc_right", 0), _hit("doc_other", 1)]
    small, _ = _svc(list(core))
    huge, _ = _svc(core + [_hit(f"doc_filler_{i}", 2 + i) for i in range(5_000)])

    assert [c.document_id for c in small.candidates("q")][:2] == ["doc_right", "doc_other"]
    assert huge.candidates("q")[0].document_id == "doc_right"


def test_candidate_set_stays_small():
    """A short list is the point; a long one restores the cost being removed."""
    svc, _ = _svc([_hit(f"doc_{i}", i) for i in range(100)])

    assert len(svc.candidates("q")) <= 8


def test_weak_candidates_are_pruned_relative_to_the_winner():
    svc, _ = _svc([_hit("doc_top", 0)] + [_hit(f"doc_tail_{i}", 150 + i) for i in range(5)])

    ranked = svc.candidates("q")

    assert ranked[0].document_id == "doc_top"
    assert all(c.relative >= MIN_RELATIVE_SCORE for c in ranked)


def test_empty_query_expresses_no_opinion():
    svc, store = _svc([_hit("doc_a", 0)])

    assert svc.candidates("   ") == []
    assert store.calls == []  # never pays for an embedding it cannot use


def test_hits_without_a_document_are_skipped_not_counted():
    svc, _ = _svc([("orphan", 0.9, None), _hit("doc_real", 1)])

    ranked = svc.candidates("q")

    assert [c.document_id for c in ranked] == ["doc_real"]


def test_candidate_ids_are_shaped_for_multi_document_scoping():
    svc, _ = _svc([_hit("doc_a", 0), _hit("doc_b", 1)])

    ids = svc.candidate_ids("q")

    assert ids == ["doc_a", "doc_b"]
    assert all(isinstance(i, str) for i in ids)
