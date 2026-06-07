"""Document RAG retriever — ranking."""
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


class RankingMixin:
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
        synthesis: bool = False,
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
            _upsert(item, norm * 0.92, "fulltext")

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

