"""term_stats.py — how many documents use a word, so "generic" is measured not guessed.

A question can be perfectly well formed, match the corpus superbly, and
still be unanswerable: "What are the key findings?" names no document and
is answerable in every one of them. Retrieval is not malfunctioning there
-- there is simply no referent to retrieve against.

Three retrieval statistics were measured against this and all three
failed, which is the reason this file exists rather than a threshold on
one of them:

  - top-document concentration  -> would decline 102 of 105 real questions
  - top raw similarity          -> "What are the limitations?" scores 0.83,
                                   higher than most real questions
  - similarity margin over the
    runner-up document          -> inverts (generic median 0.064 vs real
                                   0.027)

The distinguishing property is not about the match at all. It is that
"findings", "value", "conclusion", "limitations" are document-STRUCTURE
vocabulary occurring in nearly every document, while a real question
carries at least one term that occurs in few. That is document frequency,
measured here over the corpus itself so it adapts to whatever has been
ingested -- a hand-written list of generic words would be exactly the
overfitting this is meant to avoid, and would be wrong for a corpus where
"limitations" IS the distinguishing term.

Measured over 110 generated questions and 12 generic ones: real questions
top out at 0.042, generic ones bottom out at 0.093. The threshold sits in
that gap.
"""
from __future__ import annotations

import threading
import time
from typing import Any, Optional

from ....shared.unicode_text import words as word_tokens

# Terms rarer than this are dropped from the table and answer 0.0 on
# lookup. Only terms near the decision threshold need an accurate figure,
# so this bounds memory to the common vocabulary without changing any
# verdict: a pruned term is below the threshold either way.
_PRUNE_BELOW = 0.02

# Per-document text sampled when building the table. Document frequency
# only asks whether a term appears in a document at all, so a prefix of
# each node is enough to place common vocabulary; it under-counts every
# term uniformly, which shifts both populations together and leaves the
# gap between them intact.
_SAMPLE_NODES = 40_000
_SAMPLE_CHARS = 1_200


class CorpusTermStats:
    """Share of corpus documents containing each common term.

    Rebuilt on a TTL rather than on ingest: document frequency over
    hundreds of documents moves slowly, and a stale table degrades the
    verdict gently instead of failing it.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._df: dict[str, float] = {}
        self._built_at = 0.0
        self._doc_count = 0

    def _build(self, session: Any) -> None:
        docsets: dict[str, set[str]] = {}
        rows = session.run(
            """
            MATCH (n:Section|Page|Chapter)
            WHERE n.lifecycle_status = 'ACTIVE'
              AND size(coalesce(n.search_text, '')) > 200
            RETURN n.logical_doc_id AS d,
                   substring(n.search_text, 0, $chars) AS t
            LIMIT $limit
            """,
            chars=_SAMPLE_CHARS,
            limit=_SAMPLE_NODES,
        )
        for r in rows:
            doc = r.get("d")
            if not doc:
                continue
            docsets.setdefault(doc, set()).update(
                word_tokens(r.get("t") or "", min_length=3, hyphens=True)
            )

        total = len(docsets)
        if not total:
            self._df, self._doc_count, self._built_at = {}, 0, time.time()
            return

        counts: dict[str, int] = {}
        for words in docsets.values():
            for w in words:
                counts[w] = counts.get(w, 0) + 1
        floor = _PRUNE_BELOW * total
        self._df = {w: c / total for w, c in counts.items() if c >= floor}
        self._doc_count = total
        self._built_at = time.time()

    def _ensure(self, session: Any) -> None:
        if self._df and (time.time() - self._built_at) < self._ttl:
            return
        with self._lock:
            if self._df and (time.time() - self._built_at) < self._ttl:
                return
            self._build(session)

    def document_frequency(self, session: Any, term: str) -> float:
        self._ensure(session)
        return self._df.get((term or "").lower(), 0.0)

    def min_term_frequency(self, session: Any, keywords: list[str]) -> Optional[float]:
        """Document frequency of the query's RAREST term.

        Morphological variants are collapsed first and scored by the form
        the corpus actually uses. Keyword extraction emits a word and its
        stem together ("challenges", "challeng"), and the stem is often
        absent from prose -- taking the minimum across them read
        "challenges" as a rare term on a df of 0.009 and let the question
        through. They denote one term and must count once.
        """
        words = [k for k in (keywords or []) if k and k.isalpha()]
        if not words:
            return None
        self._ensure(session)
        if not self._df:
            return None

        groups: list[list[str]] = []
        for k in words:
            for g in groups:
                if any(k.startswith(x[:5]) or x.startswith(k[:5]) for x in g):
                    g.append(k)
                    break
            else:
                groups.append([k])
        return min(max(self._df.get(w, 0.0) for w in g) for g in groups)
