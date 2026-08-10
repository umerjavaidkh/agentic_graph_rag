"""ranking.py — chunk merging, scoring, and pinning, shared by retrieval strategies.

Extracted verbatim from mixins/ranking.py (RankingMixin) as part of the
loosely-coupled retrieval refactor. Fully self-contained — no dependency on
any other service or on `self.driver`/`self.rbac`/etc, so it's the first
service in construction order.
"""
from __future__ import annotations

import re
from typing import Optional

from ..query_intent import (
    CONTRAST_COMPARE_RE as _CONTRAST_COMPARE_RE,
    KEYWORD_STOP as _KEYWORD_STOP,
    MONTH_YEAR_RE as _MONTH_YEAR_RE,
    PHRASE_STOP as _PHRASE_STOP,
    is_enumeration_question,
)
from ..text_utils import _query_anchor_terms


# Suffix rules for _morphological_stem, ordered LONGEST FIRST so the most
# specific rule wins ("abbreviations" -> "abbreviation" via "s", but
# "implemented" -> "implement" via "ed" rather than mangling it via "d").
# Covers the three morphology classes that actually break CONTAINS matching
# between a question and a document: plurals (hospitals/Hospital), verb
# inflections (implemented/implement), and demonyms or derived adjectives
# (Jordanian/Jordan, Spanish/Spain-ish, Portuguese/Portugal-ese) -- the last
# of which no plain plural stemmer would catch, and which questions about
# countries and regions produce constantly.
# "ies" strips to the bare consonant stem ("countries" -> "countr") rather
# than restoring "country": the stem only has to be a prefix of BOTH forms
# for CONTAINS to find them, and "countri" would match "countries" while
# silently missing "country".
_STEM_SUFFIXES = ("ians", "ies", "ian", "ing", "ese", "ish", "ed", "es", "s")
# Don't touch short words: below this length, suffix stripping is far more
# likely to destroy a word ("data" -> "dat", "uses" -> "us") than to recover
# a real base form.
_STEM_MIN_SOURCE_LEN = 6
# And never emit a stub as a stem -- a 3-character CONTAINS fragment matches
# a huge amount of unrelated text.
_STEM_MIN_RESULT_LEN = 4


# Structural nouns every document type shares, optionally followed by an
# identifier ("Table A2", "Figure 1", "Annex 3", "list of abbreviations").
# Matched case-insensitively so lowercase phrasing scopes retrieval exactly
# as capitalized phrasing does.
_STRUCTURAL_REF_RE = re.compile(
    # "fig\." precedes "fig" so the abbreviated form claims its own period --
    # otherwise "fig" matches first and the trailing "." blocks the
    # identifier from being read ("Fig. 1" would yield "fig" with no number).
    r"\b(list\s+of\s+\w+|table\s+of\s+contents|table|figure|fig\.|fig|annex|appendix|"
    r"exhibit|section|chapter|box|note|glossary|abbreviations|references|"
    r"bibliography)\s*([A-Za-z]?\d+(?:\.\d+)?)?",
    re.I,
)
# A document writes one surface form and the question uses the other; both
# must be searched. Only genuine spelling variants of the SAME structural
# noun belong here -- this is not a place for topical synonyms.
_STRUCTURAL_SYNONYMS = {
    "figure": ("figure", "fig."),
    "fig": ("fig.", "figure"),
    "fig.": ("fig.", "figure"),
    "appendix": ("appendix", "annex"),
    "annex": ("annex", "appendix"),
    "abbreviations": ("abbreviations", "list of abbreviations"),
}


class RankingService:
    def _merge_and_rank(
        self,
        query: str,
        vector_hits: list[dict],
        fulltext_hits: list[dict],
        graph_hits: list[dict],
        seed_scores: dict[str, float],
        limit: int,
        *,
        lexical_hits: Optional[list[dict]] = None,
        chapter_summary_hits: Optional[list[dict]] = None,
        synthesis: bool = False,
        chapter_summary_boost: bool = False,
    ) -> list[dict]:
        merged: dict[str, dict] = {}

        def _upsert(item: dict, score: float, source: str, related: Optional[list] = None) -> None:
            cid = item.get("id") or ""
            if not cid:
                return
            rel = related or item.get("related") or []
            if cid in merged:
                merged[cid]["score"] = max(float(merged[cid]["score"]), score)
                merged[cid]["sources"].add(source)
                for r in rel:
                    if r and r not in merged[cid]["related"]:
                        merged[cid]["related"].append(r)
            else:
                merged[cid] = {
                    "id": cid,
                    "title": item.get("title") or cid,
                    "text": item.get("text") or "",
                    "page_start": item.get("page_start"),
                    "score": score,
                    "related": list(rel),
                    "sources": {source},
                }

        is_contrast = bool(_CONTRAST_COMPARE_RE.search(query or ""))
        vector_weight = 1.15 if synthesis and not is_contrast else 1.0
        graph_weight = 1.2 if synthesis and not is_contrast else 1.0
        if is_contrast:
            lexical_weight = 1.1
        elif synthesis:
            lexical_weight = 0.82
        else:
            lexical_weight = 1.0

        for item in vector_hits:
            _upsert(item, float(item.get("score", 0.0)) * vector_weight, "vector")

        max_ft = max((float(h.get("score", 0.0)) for h in fulltext_hits), default=1.0) or 1.0
        for item in fulltext_hits:
            norm = float(item.get("score", 0.0)) / max_ft
            weighted = norm * 0.92
            # Page nodes are never embedded (only Chapter/Section get
            # embeddings in Axis2 enrichment — see SEMANTIC_NODE_TYPES in
            # semantic/axis2.py), so a Page can never appear in vector_hits
            # at all, regardless of how directly it answers the question.
            # Without this, fulltext's normalized ceiling (0.92) plus that
            # structural exclusion means Page content routinely loses to
            # Section-level vector/graph hits that are merely topically
            # related, even when the Page is the one place with the actual
            # answer (e.g. a signature page listing names no Section
            # captures coherently). Applies to every document, not this one.
            if item.get("node_label") == "Page":
                weighted *= 1.3
            _upsert(item, weighted, "fulltext")

        for item in graph_hits:
            seed_id = item.get("seed_id") or ""
            base = seed_scores.get(seed_id, 0.55)
            hop_decay = 0.88 ** int(item.get("hops", 1))
            edge_w = float(item.get("edge_weight", 0.75))
            rel = item.get("rel_type")
            _upsert(
                item,
                base * hop_decay * edge_w * graph_weight,
                "graph",
                [rel] if rel else [],
            )

        for item in lexical_hits or []:
            src = "phrase" if "phrase_search" in (item.get("related") or []) else "keyword"
            _upsert(item, float(item.get("score", 0.85)) * lexical_weight, src)

        # Chapter rollup summaries only matter for "what does this document/
        # chapter discuss" style questions (chapter_summary_boost — a
        # broader, dedicated detector than `synthesis`, see
        # is_overview_question in query_intent.py) — for everything else a
        # short paragraph summary is more likely to dilute a specific-fact
        # answer than help it, so it's barely boosted (still eligible, just
        # unlikely to outrank an actual on-topic chunk) rather than fully
        # excluded.
        chapter_summary_weight = 1.25 if chapter_summary_boost else 0.4
        for item in chapter_summary_hits or []:
            _upsert(item, float(item.get("score", 0.9)) * chapter_summary_weight, "chapter_summary")

        keywords = self._query_keywords(query)
        for item in merged.values():
            item["score"] = float(item["score"]) * self._relevance_boost(
                item.get("title") or "",
                item.get("text") or "",
                keywords,
            )

        ranked = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        out: list[dict] = []
        for item in ranked[:limit]:
            sources = sorted(item.pop("sources", {"graph"}))
            item["related"] = item.get("related") or []
            if sources:
                item["related"] = list(dict.fromkeys([*item["related"], f"via:{','.join(sources)}"]))
            out.append(item)
        return out

    def _contrast_term_groups(self, query: str) -> list[list[str]]:
        """For compare/contrast questions, one keyword group per side of the comparison."""
        if not _CONTRAST_COMPARE_RE.search(query or ""):
            return []
        parts = re.split(
            r"\b(?:versus|vs\.?|compared\s+to|against)\b",
            query or "",
            maxsplit=1,
            flags=re.I,
        )
        if len(parts) >= 2:
            groups: list[list[str]] = []
            for part in parts[:2]:
                kws = [
                    k
                    for k in self._content_keywords_from_query(part)
                    if len(k) >= 4 and k not in _KEYWORD_STOP
                ]
                if kws:
                    groups.append(kws[:4])
            if len(groups) >= 2:
                return groups
        q = (query or "").lower()
        groups = []
        for token in self._query_keywords(query):
            if len(token) >= 5 and token not in {
                "contrast", "compare", "comparison", "between", "versus",
            }:
                groups.append([token])
        return groups[:2] if len(groups) >= 2 else []

    @staticmethod
    def _text_matches_term_groups(text: str, groups: list[list[str]]) -> bool:
        if len(groups) < 2:
            return False
        norm = (text or "").lower().replace(" ", "").replace(".", "")
        for group in groups:
            if not any(g.lower().replace(" ", "").replace(".", "") in norm for g in group):
                return False
        return True

    def scope_phrases_from_query(self, query: str) -> list[str]:
        """Short proper-noun phrases from the query that may name a SCOPE the
        document partitions its content by ("International Upstream", "U.S.
        Downstream", "Note 15").

        Distinct from _search_phrases_from_query, which builds long 6-8 word
        n-grams spanning the whole question. Those are good at pinning a
        chunk that restates the question almost verbatim, but they never
        match a document that states the same fact in its own words -- and
        crucially they cannot isolate a scope, because the scope terms are
        diluted among six other tokens. Verified live on a 10-K: the question
        "International Upstream net oil-equivalent production" produced only
        long phrases, none of which appear contiguously anywhere in the
        filing, so lexical retrieval contributed nothing and vector search
        returned a DIFFERENT segment's table with the same row labels -- the
        answer was a confidently wrong number, not a miss.

        Extraction is deliberately permissive (any run of capitalized tokens,
        2-3 words), because the caller applies a document-frequency filter
        that is far better at telling a scope from boilerplate than any
        pattern could be: "Chevron" or "Annual Report" appear in most nodes
        and get dropped, while "International Upstream" appears in few and
        survives. Same generic-vs-distinctive IDF reasoning this repo already
        uses for entity anchoring (semantic/axis2.py) and lexical ranking.
        """
        if not query:
            return []
        phrases: list[str] = []
        # Structural references ("list of abbreviations", "table A2", "annex
        # 1") name a scope just as precisely as a proper noun does, but users
        # type them in lowercase, so the capitalization rule below never sees
        # them -- verified live: "What is in the List of Abbreviations?"
        # returned all 24 entries while the identical lowercase question
        # answered "this document does not cover" it. The vocabulary is
        # document-STRUCTURE words (every document has sections, tables and
        # figures), never document content, so it carries no corpus
        # assumptions. Both surface forms of the abbreviated ones are emitted
        # ("Figure 1" / "Fig. 1") because a document picks one and the
        # question picks the other independently -- the same live failure:
        # asking "Figure 1" missed a region titled "Fig. 1:".
        for m in _STRUCTURAL_REF_RE.finditer(query):
            noun, ident = m.group(1).lower(), (m.group(2) or "").strip()
            for form in _STRUCTURAL_SYNONYMS.get(noun, (noun,)):
                phrases.append(f"{form} {ident}".strip() if ident else form)
        # Runs of capitalized/abbreviated tokens: "U.S. Downstream",
        # "International Upstream", "Note 15". Case is meaningful here, so
        # this reads the raw query rather than the lowercased forms the other
        # phrase helpers work from.
        for run in re.findall(
            r"\b(?:[A-Z][\w.&'’-]*|\d{1,3})(?:\s+(?:[A-Z][\w.&'’-]*|\d{1,3})){1,2}\b",
            query,
        ):
            tokens = run.split()
            for size in (3, 2):
                for i in range(len(tokens) - size + 1):
                    phrase = " ".join(tokens[i : i + size])
                    if any(c.isalpha() for c in phrase):
                        phrases.append(phrase)
        # Preserve order, drop duplicates case-insensitively.
        seen: set[str] = set()
        out: list[str] = []
        for p in phrases:
            key = p.lower()
            if key in seen or key in _KEYWORD_STOP:
                continue
            seen.add(key)
            out.append(p)
        return out[:8]

    def _precision_pin_patterns(self, query: str) -> list[str]:
        """Long query-derived phrases used to pin compact high-signal chunks."""
        min_len = 10 if is_enumeration_question(query) else 8
        patterns = [
            p for p in self._search_phrases_from_query(query) if len(p) >= min_len
        ]
        return list(dict.fromkeys(patterns))[:8]

    def _pin_precision_lexical_chunks(
        self,
        query: str,
        items: list[dict],
        lexical_hits: list[dict],
        *,
        limit: int,
    ) -> list[dict]:
        """
        Pin compact, high-signal lexical hits (e.g. Region facts, network lists)
        that vector search often ranks below broad sections.
        """
        patterns = self._precision_pin_patterns(query)
        if not patterns:
            return items

        pinned: list[dict] = []
        for hit in lexical_hits:
            text = (hit.get("text") or "").lower()
            if any(p in text for p in patterns):
                pinned.append(hit)
        if not pinned:
            return items

        pinned.sort(key=lambda h: len(h.get("text") or ""))

        seen: set[str] = set()
        out: list[dict] = []
        for hit in pinned[:3]:
            cid = hit.get("id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(
                {
                    "id": cid,
                    "title": hit.get("title") or cid,
                    "text": hit.get("text") or "",
                    "page_start": hit.get("page_start"),
                    "score": float(hit.get("score", 1.5)) + 0.55,
                    "related": list(
                        dict.fromkeys([*(hit.get("related") or []), "via:precision_pin"])
                    ),
                }
            )

        for item in items:
            cid = item.get("id")
            if cid and cid not in seen:
                out.append(item)
            if len(out) >= limit:
                break
        return out[:limit]

    def _pin_contrast_lexical_chunks(
        self,
        query: str,
        items: list[dict],
        lexical_hits: list[dict],
        *,
        limit: int,
    ) -> list[dict]:
        """
        Contrast questions need chunks that mention BOTH sides named in the query.
        Vector-only ranking often returns executive-summary pages and drops the intro contrast.
        """
        groups = self._contrast_term_groups(query)
        if len(groups) < 2:
            return items

        pinned: list[dict] = []
        for hit in lexical_hits:
            if self._text_matches_term_groups(hit.get("text") or "", groups):
                pinned.append(hit)

        if not pinned:
            return items

        # Prefer the smallest Section chunk (figure callouts are often on one intro section).
        pinned.sort(
            key=lambda h: (
                0 if (h.get("related") or []) and "keyword" in str(h.get("related")) else 1,
                len(h.get("text") or ""),
            )
        )

        seen: set[str] = set()
        out: list[dict] = []
        for hit in pinned[:2]:
            cid = hit.get("id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(
                {
                    "id": cid,
                    "title": hit.get("title") or cid,
                    "text": hit.get("text") or "",
                    "page_start": hit.get("page_start"),
                    "score": float(hit.get("score", 1.5)) + 0.5,
                    "related": list(
                        dict.fromkeys([*(hit.get("related") or []), "via:contrast_pin"])
                    ),
                }
            )

        for item in items:
            cid = item.get("id")
            if cid and cid not in seen:
                out.append(item)
            if len(out) >= limit:
                break
        return out[:limit]

    def _pin_firmwide_summary_chunks(
        self,
        items: list[dict],
        financial_summary_hits: list[dict],
        *,
        limit: int,
    ) -> list[dict]:
        """
        Pin a document's firmwide financial-summary section(s) to the top.

        For a firmwide financial-metric question (gated by the caller via
        is_firmwide_financial_metric_question) the authoritative figure lives
        in a summary section that vector cosine buries under short segment
        tables repeating the metric name. Pinning forces it into context so
        synthesis answers the firm total, not a segment's. Sections that hold
        actual figures ('$'/digits) and are not flagged low-confidence rank
        first.

        Pins the whole candidate set (already small and precise — at most 5,
        title-matched to a fixed list of summary-section headings by
        FinancialSummaryService, not a broad search) rather than truncating
        to the top couple: different summary sections cover different
        metrics (e.g. "Selected Financial Data" has EPS/assets while ROE
        only appears in "Executive Overview"), so keeping just the
        shortest/most-figure-dense ones by a text-length heuristic can
        silently drop the one section that actually answers the question —
        verified live on a "return on equity" question that needed the
        longer Executive Overview section, not the shorter tables ranked
        above it under the old top-2 cap.
        """
        if not financial_summary_hits:
            return items

        def _rank_key(h: dict) -> tuple:
            text = (h.get("text") or "")
            low_conf = "[low confidence extract]" in text.lower()
            has_figures = ("$" in text) or any(ch.isdigit() for ch in text)
            return (low_conf, not has_figures, len(text))

        pinned = sorted(financial_summary_hits, key=_rank_key)

        seen: set[str] = set()
        out: list[dict] = []
        for hit in pinned:
            cid = hit.get("id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append(
                {
                    "id": cid,
                    "title": hit.get("title") or cid,
                    "text": hit.get("text") or "",
                    "page_start": hit.get("page_start"),
                    "score": float(hit.get("score", 1.0)) + 10.0,
                    "related": list(
                        dict.fromkeys(
                            [*(hit.get("related") or []), "via:financial_summary_pin"]
                        )
                    ),
                }
            )

        for item in items:
            cid = item.get("id")
            if cid and cid not in seen:
                out.append(item)
            if len(out) >= limit:
                break
        return out[:limit]

    def _pin_scope_chunks(
        self,
        items: list[dict],
        scope_hits: list[dict],
        *,
        limit: int,
    ) -> list[dict]:
        """
        Pin chunks belonging to the scope (segment/region/note) the question
        actually named, ahead of same-shaped chunks from sibling scopes.

        Ranking alone cannot fix this, which is why it is a pin: the correct
        chunk was retrievable the whole time and still never reached the
        context window, because every sibling segment's table scores nearly
        identically on a metric name they all share. Verified live on a 10-K
        -- a question about International Upstream's liquids production was
        answered with a different segment's figure, and querying the
        document's own verbatim sentence did not surface its own chunk.

        Ordered like the other pins: chunks carrying actual figures first,
        shorter (denser) ones ahead of long ones -- a scope's summary table
        answers a metric question better than the prose section that merely
        mentions it.
        """
        if not scope_hits:
            return items

        def _rank_key(h: dict) -> tuple:
            text = h.get("text") or ""
            low_conf = "[low confidence extract]" in text.lower()
            has_figures = ("$" in text) or any(ch.isdigit() for ch in text)
            return (low_conf, not has_figures, -float(h.get("score") or 0.0), len(text))

        seen: set[str] = set()
        out: list[dict] = []
        for hit in sorted(scope_hits, key=_rank_key):
            cid = hit.get("id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            out.append({
                "id": cid,
                "title": hit.get("title") or cid,
                "text": hit.get("text") or "",
                "page_start": hit.get("page_start"),
                "score": float(hit.get("score", 1.0)) + 10.0,
                "related": list(
                    dict.fromkeys([*(hit.get("related") or []), "via:scope_pin"])
                ),
            })

        for item in items:
            cid = item.get("id")
            if cid and cid not in seen:
                out.append(item)
            if len(out) >= limit:
                break
        return out[:limit]

    def _pin_keyword_leader(
        self,
        items: list[dict],
        lexical_hits: list[dict],
        *,
        limit: int,
    ) -> list[dict]:
        """
        Pin the single best keyword match so a narrow factual question can
        still reach the one chunk that answers it.

        structural_keyword_retrieve already ranks by idf, so its first hit is
        the chunk matching the query's RAREST terms -- precisely the chunk a
        specific factual question is about. But lexical scores enter
        _merge_and_rank around 1.0 while vector hits come out of the
        relevance boost around 4-5, so that leader is routinely ranked off
        the end of the context window by broadly-similar prose.

        Verified live: "Which Jordanian hospitals implemented Go.Data in
        January 2021?" returned "this document does not cover" while
        structural_keyword_retrieve ranked the very page holding all eight
        hospitals FIRST -- the answer was retrieved and then discarded before
        synthesis ever saw it. Only the leader is pinned (not the whole
        lexical list), so this stays a precision instrument: it costs exactly
        one slot, and only when a keyword search found something at all.
        """
        if not lexical_hits or not items:
            return items
        leader = lexical_hits[0]
        leader_id = leader.get("id")
        if not leader_id or any(i.get("id") == leader_id for i in items[:limit]):
            return items

        pinned = {
            "id": leader_id,
            "title": leader.get("title") or leader_id,
            "text": leader.get("text") or "",
            "page_start": leader.get("page_start"),
            "score": float(leader.get("score", 1.0)) + 10.0,
            "related": list(
                dict.fromkeys([*(leader.get("related") or []), "via:keyword_leader"])
            ),
        }
        return ([pinned] + [i for i in items if i.get("id") != leader_id])[:limit]

    def _search_phrases_from_query(self, query: str) -> list[str]:
        """
        Build document-agnostic search phrases from the question (dates + word n-grams).
        Keeps light stopwords (of, the, at) so phrases align with PDF sentence wording.
        """
        q = (query or "").lower()
        phrases: list[str] = []

        for m in _MONTH_YEAR_RE.finditer(q):
            phrases.append(f"{m.group(1).lower()} {m.group(2)}")

        _light_stop = _PHRASE_STOP - frozenset({
            "of", "at", "in", "on", "to", "and", "or", "for", "by", "with", "from",
        })
        tokens: list[str] = []
        for anchor in _query_anchor_terms(query):
            for w in re.findall(r"[\w']+", anchor):
                if len(w) >= 2 and w not in tokens:
                    tokens.append(w)
        for w in re.findall(r"[\w']+", q):
            if len(w) <= 2 or w in _light_stop:
                continue
            if w not in tokens:
                tokens.append(w)

        for n in range(min(7, len(tokens)), 2, -1):
            for i in range(len(tokens) - n + 1):
                phrase = " ".join(tokens[i : i + n])
                if len(phrase) >= 8:
                    phrases.append(phrase)

        for w in tokens:
            if len(w) >= 5:
                phrases.append(w)

        seen: set[str] = set()
        ordered: list[str] = []
        for p in sorted(phrases, key=len, reverse=True):
            pl = p.lower().strip()
            if pl and pl not in seen:
                seen.add(pl)
                ordered.append(pl)
        return ordered[:14]

    @staticmethod
    def _morphological_stem(word: str) -> Optional[str]:
        """Shortest safe base form of `word`, or None when no rule applies.

        Every lexical predicate in this file matches with CONTAINS, which is
        exact-substring: a question asking about "hospitals" or "Jordanian"
        never matches a document that writes "Hospital" and "Jordan", because
        neither of those CONTAINS the longer query word. Verified live -- the
        question "Which Jordanian hospitals implemented Go.Data in January
        2021?" was answered "this document does not cover" by a graph that,
        one question earlier, had correctly listed all eight of those
        hospitals from the same table. Two morphological misses in one
        question ("jordanian" vs "Jordan", "hospitals" vs "Hospital") were
        enough to lose it entirely.

        Stemming the QUERY side (rather than indexing stems) is what makes
        this work with CONTAINS instead of against it: the stem is always a
        prefix of the original, so CONTAINS(stem) matches a strict superset
        of what CONTAINS(original) matched -- "hospital" finds both
        "hospital" and "hospitals". Recall can only improve, never regress,
        and precision is still handled downstream by the existing idf
        weighting (a stem that matches too much simply weighs little).

        Plain English suffix rules, no wordlist and no per-corpus vocabulary,
        so this behaves identically on any document in any domain. The
        minimum stem length keeps short words ("data", "site") untouched,
        where suffix stripping is far likelier to destroy a word than to
        recover its base form.
        """
        w = (word or "").lower()
        if len(w) < _STEM_MIN_SOURCE_LEN or not w.isalpha():
            return None
        for suffix in _STEM_SUFFIXES:
            if w.endswith(suffix) and len(w) - len(suffix) >= _STEM_MIN_RESULT_LEN:
                return w[: -len(suffix)]
        return None

    def _content_keywords_from_query(self, query: str) -> list[str]:
        """
        Distinct content terms for AND-style overlap scoring (corpus-agnostic).

        Derived entirely from the question: proper-noun/acronym anchors, month-year
        dates, content tokens, hyphen/space variants, and adjacent bigrams. No
        per-document or per-topic vocabulary is injected here.
        """
        q = (query or "").lower()
        keywords: list[str] = []

        for anchor in _query_anchor_terms(query):
            if anchor not in keywords:
                keywords.append(anchor)

        for m in _MONTH_YEAR_RE.finditer(q):
            keywords.append(f"{m.group(1).lower()} {m.group(2)}")
            keywords.append(m.group(2))

        # Hyphenated terms in the query: add joined / spaced variants generically
        # (e.g. "case-control" → "case control", "casecontrol") to survive PDF wording.
        for hyph in re.findall(r"[a-z]+(?:-[a-z]+)+", q):
            keywords.append(hyph)
            keywords.append(hyph.replace("-", " "))
            keywords.append(hyph.replace("-", ""))

        for w in re.findall(r"[\w']+", q):
            if len(w) <= 2 or w in _PHRASE_STOP:
                continue
            if w not in keywords:
                keywords.append(w)
            # The stem REPLACES nothing -- it is added alongside, because a
            # stem is a prefix of the word and therefore matches everything
            # the word did plus its other inflections. Keeping both lets the
            # idf weighting prefer the exact form when the document uses it.
            stem = self._morphological_stem(w)
            if stem and stem not in keywords:
                keywords.append(stem)

        words = [
            w
            for w in re.findall(r"[\w']+", q)
            if len(w) >= 4 and w not in _KEYWORD_STOP
        ]
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i + 1]}"
            if bigram not in keywords:
                keywords.append(bigram)

        return list(dict.fromkeys(keywords))[:18]

    @staticmethod
    def _merge_retrieval_chunks(primary: list[dict], extra: list[dict]) -> list[dict]:
        merged = list(primary)
        seen = {c["id"] for c in merged if c.get("id")}
        for item in extra:
            cid = item.get("id")
            if not cid or cid in seen:
                continue
            seen.add(cid)
            merged.append(item)
        merged.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
        return merged[:8]

    def _query_keywords(self, question: str) -> list[str]:
        terms = re.findall(r"[\w'-]{3,}", (question or "").lower())
        return [t for t in terms if t not in _KEYWORD_STOP][:18]

    def _relevance_boost(self, title: str, text: str, keywords: list[str]) -> float:
        """Boost named sections and chunks that match more query terms."""
        boost = 1.0
        if title.strip() and not re.match(r"^Page\s+\d+$", title.strip(), re.I):
            boost *= 1.08
        hay = f"{title} {text}".lower()
        if keywords:
            hits = sum(1 for k in keywords if k in hay)
            boost *= 1.0 + min(0.45, 0.07 * hits)
        return boost
