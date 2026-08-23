"""candidate_docs.py — pick the documents a question is about, in time that
does not grow with the corpus.

The graph-side resolver answers this by walking every document's subtree and
applying a regex per content node (`DocumentResolver._scored_documents`).
That is O(corpus): measured at 14.36s across 998 documents, paid on every
query before any retrieval starts, and it grows with each ingest.

The vector index already answers the same question. One ANN lookup returns
the `probe_k` nearest content chunks; the documents those chunks belong to
are the candidate set. ANN search is sub-linear in collection size, and
`probe_k` is a constant, so both the lookup and everything downstream that
scopes to the result stay flat as documents are added.

This is a *scoping* pass, not an answer. It narrows the corpus to a handful
of documents that the existing lexical/graph/hybrid retrieval then searches
exactly as it does today -- via `node_scope_cypher_multi`, which already
exists for precisely this shape of input.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ....shared.config.settings import EMBEDDING_MODEL, MULTI_TENANCY_ENABLED
from ....shared.model_providers.factory import get_embedding_provider
from ....shared.storage.vector.factory import get_vector_store

# How many nearest chunks to pull before aggregating them into documents.
# Constant on purpose -- this is the number that keeps the pass independent
# of corpus size, so it must never be derived from the document count. Large
# enough that a document answering the question with a single strong chunk
# still appears; small enough that Qdrant serves it from the HNSW graph
# without a full rescan.
PROBE_K = 200

# Most documents any one question is allowed to be about. The point of the
# pass is to hand the graph a short list; a long one restores the cost it
# was meant to remove.
MAX_CANDIDATE_DOCS = 8

# Reciprocal-rank constant, the standard 60.
RRF_K = 60

# A document scores on its BEST chunk, not on the sum of its chunks.
#
# Summing was tried first and is wrong: a document contributing sixty
# mediocre chunks accumulates more total reciprocal rank than the document
# holding the single best one, so sheer size wins again -- the identical
# failure that `log1p(cnt)` was bolted onto the graph resolver to contain,
# and that still let an unrelated arXiv paper come within 1.21x of the right
# document on "Go.Data annual report". Max-pooling over a document's chunks
# is the standard way to rank documents from chunk embeddings precisely
# because it is immune to length.
#
# Corroboration still counts for something -- several good chunks are better
# evidence than one -- so it is a multiplicative bonus, normalised by
# log1p(PROBE_K) so it can never exceed this factor. A document therefore
# cannot overtake one with a better best-chunk by volume alone; the bonus
# only separates documents whose best chunks are already close.
CORROBORATION_WEIGHT = 0.5

# A candidate must reach this share of the winner's score to stay in the
# set. Relative, not absolute: scores are fusion ranks, so their scale
# depends on how many chunks a document contributed, never on the corpus.
MIN_RELATIVE_SCORE = 0.15


@dataclass(frozen=True)
class DocCandidate:
    document_id: str
    score: float
    relative: float
    hits: int
    best_rank: int


class CandidateDocService:
    """Vector-first document scoping."""

    def __init__(self, embed=None, vector_store=None):
        # Injected for tests; resolved lazily in production so importing this
        # module never constructs a client.
        self._embed = embed
        self._vector_store = vector_store

    def _embedding(self, text: str) -> list[float]:
        if self._embed is not None:
            return self._embed(text)
        resp = get_embedding_provider().embeddings(
            model=EMBEDDING_MODEL, input=(text or "")[:8000]
        )
        return list(resp.data[0].embedding)

    def _store(self):
        return self._vector_store if self._vector_store is not None else get_vector_store()

    def candidates(
        self,
        query: str,
        tenant_id: str = "",
        *,
        probe_k: int = PROBE_K,
        limit: int = MAX_CANDIDATE_DOCS,
    ) -> list[DocCandidate]:
        """Documents this query is plausibly about, best first.

        Empty when the query is empty or the vector store returns nothing --
        an empty result means "no opinion", and the caller should fall back
        rather than treat it as "no documents match".
        """
        if not (query or "").strip():
            return []

        filters = {"tenant_id": tenant_id} if (MULTI_TENANCY_ENABLED and tenant_id) else None
        hits = self._store().query_with_docs(
            self._embedding(query), top_k=probe_k, filters=filters
        )
        if not hits:
            return []

        agg: dict[str, dict] = {}
        for rank, (_node_id, _score, doc_id) in enumerate(hits):
            if not doc_id:
                continue
            slot = agg.setdefault(doc_id, {"hits": 0, "best_rank": rank})
            slot["hits"] += 1
            slot["best_rank"] = min(slot["best_rank"], rank)

        if not agg:
            return []

        span = math.log1p(max(probe_k, 1))
        for slot in agg.values():
            base = 1.0 / (RRF_K + slot["best_rank"] + 1)
            corroboration = math.log1p(slot["hits"] - 1) / span if span else 0.0
            slot["score"] = base * (1.0 + CORROBORATION_WEIGHT * corroboration)

        ranked = sorted(agg.items(), key=lambda kv: kv[1]["score"], reverse=True)
        top = ranked[0][1]["score"] or 1.0
        out = [
            DocCandidate(
                document_id=doc_id,
                score=v["score"],
                relative=v["score"] / top,
                hits=v["hits"],
                best_rank=v["best_rank"],
            )
            for doc_id, v in ranked
        ]
        return [c for c in out if c.relative >= MIN_RELATIVE_SCORE][:limit]

    def candidate_ids(
        self, query: str, tenant_id: str = "", *, limit: int = MAX_CANDIDATE_DOCS
    ) -> list[str]:
        """Just the ids, shaped for `node_scope_cypher_multi`'s `$doc_ids`."""
        return [c.document_id for c in self.candidates(query, tenant_id, limit=limit)]
