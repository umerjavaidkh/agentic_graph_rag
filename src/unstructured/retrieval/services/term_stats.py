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

from ....shared.config.settings import DEFAULT_LANGUAGE
from ....shared.neo4j.tenancy import language_filter, tenant_filter
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

# Fewest documents a language needs before its frequency table is allowed
# an opinion. Derived from the separation this file already measures, not
# picked: real questions top out at 0.042 and generic ones bottom out at
# 0.093, so the table has to resolve a gap of about 0.05. Document
# frequency is a proportion over n documents, so its finest possible
# distinction is 1/n -- at 20 documents that is 0.05, exactly the width of
# the gap, and every value lands on one side or the other by accident. 30
# gives the measurement room to be a measurement.
#
# Below this the table returns no opinion rather than a wrong one, which
# the caller already handles: `min_term_frequency` returning None means
# "cannot judge", and the gate answers normally instead of declining.
#
# This is what stops per-language partitioning from breaking a young
# corpus. With one Arabic document every term in it has df = 1.0 -- not
# because those words are generic, but because there is nothing to be
# generic against -- and a table that reported that would decline every
# Arabic question as underspecified.
_MIN_DOCS_FOR_DF = 30


class CorpusTermStats:
    """Share of corpus documents containing each common term.

    Rebuilt on a TTL rather than on ingest: document frequency over
    hundreds of documents moves slowly, and a stale table degrades the
    verdict gently instead of failing it.
    """

    def __init__(
        self, ttl_seconds: int = 3600, min_docs: int = _MIN_DOCS_FOR_DF
    ) -> None:
        self._ttl = ttl_seconds
        # Injectable so a test can say which of the two things it is
        # testing: the frequency arithmetic, or the policy about when
        # that arithmetic is trustworthy. A fixture of ten documents is
        # exercising the first and should not have to satisfy the second.
        self._min_docs = min_docs
        self._lock = threading.Lock()
        # One table per language. A single shared table cannot work once a
        # second language exists: with 998 English documents and one
        # Arabic one, every Arabic term sits at df = 0.001 and is pruned
        # below the floor, so the whole Arabic vocabulary reads as
        # maximally rare. Measured before this existed -- 0 Arabic terms
        # survived in a 6,919-term table.
        self._tables: dict[str, dict[str, float]] = {}
        self._built_at: dict[str, float] = {}
        self._doc_counts: dict[str, int] = {}

    def _build(self, session: Any, language: str, tenant_id: str) -> None:
        docsets: dict[str, set[str]] = {}
        rows = session.run(
            f"""
            MATCH (n:Section|Page|Chapter)
            WHERE n.lifecycle_status = 'ACTIVE'
              AND size(coalesce(n.search_text, '')) > 200
              AND {tenant_filter("n")} AND {language_filter("n")}
            RETURN n.logical_doc_id AS d,
                   substring(n.search_text, 0, $chars) AS t
            LIMIT $limit
            """,
            chars=_SAMPLE_CHARS,
            limit=_SAMPLE_NODES,
            language=language,
            tenant_id=tenant_id,
        )
        for r in rows:
            doc = r.get("d")
            if not doc:
                continue
            docsets.setdefault(doc, set()).update(
                word_tokens(r.get("t") or "", min_length=3, hyphens=True)
            )

        key = (tenant_id, language)
        total = len(docsets)
        self._doc_counts[key] = total
        self._built_at[key] = time.time()
        if not total:
            self._tables[key] = {}
            return

        counts: dict[str, int] = {}
        for words in docsets.values():
            for w in words:
                counts[w] = counts.get(w, 0) + 1
        floor = _PRUNE_BELOW * total
        self._tables[key] = {w: c / total for w, c in counts.items() if c >= floor}

    def _ensure(self, session: Any, language: str, tenant_id: str) -> None:
        key = (tenant_id, language)
        built = self._built_at.get(key, 0.0)
        if key in self._tables and (time.time() - built) < self._ttl:
            return
        with self._lock:
            built = self._built_at.get(key, 0.0)
            if key in self._tables and (time.time() - built) < self._ttl:
                return
            self._build(session, language, tenant_id)

    def _has_opinion(self, key: tuple) -> bool:
        """Whether this language has enough documents to judge rarity.

        A table built from too few documents is not a weak measurement, it
        is a different one: with a single document every term in it has
        df = 1.0, which reads exactly like "this word is generic" and is
        nothing of the sort.
        """
        return self._doc_counts.get(key, 0) >= self._min_docs

    def document_frequency(
        self,
        session: Any,
        term: str,
        language: str = DEFAULT_LANGUAGE,
        tenant_id: str = "",
    ) -> float:
        self._ensure(session, language, tenant_id)
        table = self._tables.get((tenant_id, language), {})
        return table.get((term or "").lower(), 0.0)

    def min_term_frequency(
        self,
        session: Any,
        keywords: list[str],
        language: str = DEFAULT_LANGUAGE,
        tenant_id: str = "",
    ) -> Optional[float]:
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
        key = (tenant_id, language)
        self._ensure(session, language, tenant_id)
        table = self._tables.get(key) or {}
        if not table:
            return None
        # None means "cannot judge", and the caller answers normally on it.
        # That is the right answer for a corpus too small to have generic
        # vocabulary yet -- see _MIN_DOCS_FOR_DF.
        if not self._has_opinion(key):
            return None

        groups: list[list[str]] = []
        for k in words:
            for g in groups:
                if any(k.startswith(x[:5]) or x.startswith(k[:5]) for x in g):
                    g.append(k)
                    break
            else:
                groups.append([k])
        return min(max(table.get(w, 0.0) for w in g) for g in groups)
