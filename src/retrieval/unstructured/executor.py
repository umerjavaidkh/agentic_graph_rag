from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ...conversation.clarification import format_clarification_answer


_SECTION_NUM_RE = re.compile(r"\b(\d+(?:\.\d+){1,3})\b")
# Covers financial-statement footnotes ("Note 3"), SEC filing items (which
# use a letter-suffixed numbering convention Note references don't --
# "Item 9A. Controls and Procedures", "Item 7A", "Item 1B"), textbook
# worked examples ("Example 2.8") -- unlike Note/Item, a textbook's own
# heading is "Example 2.8" (word BEFORE a DOTTED number, not the number
# alone), so the returned "example 2.8" combined string still lines up
# with subsection.py's `title STARTS WITH` match, which a bare "2.8"
# extracted by _SECTION_NUM_RE alone would not (that title doesn't start
# with the number) -- and textbook chapter references ("Chapter 15"),
# same bare-integer shape as Note/Item. Fourth regression, found via the
# physics-textbook stress test: "List every Check Your Understanding
# question in Chapter 15." had no structural reference recognized at all,
# so it fell through to generic hybrid retrieval, which returned unrelated
# pages (Preface, Chapter 1, random page numbers) instead of anything from
# Chapter 15 -- verified live against the real ingested textbook.
# \d+(?:\.\d+)*[a-z]? covers all four shapes ("3", "7A", "2.8", "15") with
# one pattern -- the added `(?:\.\d+)*` is additive (zero-or-more), so
# existing Note/Item matches are unaffected.
_STRUCTURAL_NUM_RE = re.compile(r"\b(note|item|example|chapter)\s+(?:no\.?\s*)?(\d+(?:\.\d+)*[a-z]?)\b", re.I)
_SUBSECTION_CUE_RE = re.compile(r"\b(sub\s*sections?|subsections?|under\s+this\s+section)\b", re.I)
_BOX_LIST_CUE_RE = re.compile(r"\b(list|show|enumerate|all)\b.{0,20}\bbox(?:es)?\b", re.I)
_BOX_RE = re.compile(r"\bbox\s+(\d{1,3})\b", re.I)


@dataclass
class DocClarification:
    kind: str
    prompt: str
    options: list[dict[str, Any]]


class DocumentQueryExecutor:
    """Generic helpers for document ambiguity + subsection requests."""

    def parse_section_number(self, query: str) -> Optional[str]:
        # Financial-statement footnotes ("Note 3 — Commitments and
        # Contingencies") are stored as Section nodes titled "Note N — ...",
        # not dotted numeric headings — checked first since a bare "3" here
        # wouldn't match _SECTION_NUM_RE anyway (no dot), but a query could
        # in principle contain both forms.
        m = _STRUCTURAL_NUM_RE.search(query or "")
        if m:
            return f"{m.group(1).lower()} {m.group(2).lower()}"
        m = _SECTION_NUM_RE.search(query or "")
        return m.group(1) if m else None

    def is_subsection_request(self, query: str) -> bool:
        q = query or ""
        if _STRUCTURAL_NUM_RE.search(q):
            return True
        return bool(_SUBSECTION_CUE_RE.search(q)) or ("sub section" in q.lower())

    def has_multiple_structural_references(self, query: str) -> bool:
        """True when a query names more than one distinct Note/Item/Example/
        Chapter reference ("How does Chapter 9 relate to Chapter 11?").

        parse_section_number only ever returns the FIRST match -- a query
        naming two chapters silently narrows retrieval to just the first
        one, so a genuine cross-chapter question gets answered from a
        single chapter's content alone and the model correctly (from its
        narrow context) reports the other chapter as "not covered." Callers
        use this to route to a dedicated multi-reference lookup (see
        parse_all_section_numbers) instead of confidently answering wrong
        from a single match. Verified live: "How does the treatment of
        momentum in Chapter 9 relate to angular momentum in Chapter 11?"
        against the real ingested textbook returned only Chapter 9's
        sections and answered "does not cover" the Chapter 11 relationship,
        before this check existed.
        """
        return len(self.parse_all_section_numbers(query)) > 1

    def parse_all_section_numbers(self, query: str) -> list[str]:
        """Every distinct Note/Item/Example/Chapter reference in a query,
        in first-seen order (unlike parse_section_number, which only
        returns the first). Used for comparison-style questions naming
        more than one reference.
        """
        seen: list[str] = []
        seen_set: set[str] = set()
        for g1, g2 in _STRUCTURAL_NUM_RE.findall(query or ""):
            combined = f"{g1.lower()} {g2.lower()}"
            if combined not in seen_set:
                seen_set.add(combined)
                seen.append(combined)
        return seen

    def is_box_list_request(self, query: str) -> bool:
        q = query or ""
        return bool(_BOX_LIST_CUE_RE.search(q)) or bool(re.search(r"\bbox\s+headings?\b", q, re.I))

    def extract_box_numbers(self, text: str) -> list[int]:
        nums: list[int] = []
        for m in _BOX_RE.finditer(text or ""):
            try:
                nums.append(int(m.group(1)))
            except Exception:
                continue
        # de-dupe while keeping order
        return list(dict.fromkeys(nums))

    def parse_box_number(self, query: str) -> Optional[int]:
        """Return Box number if query mentions a specific Box N."""
        nums = self.extract_box_numbers(query or "")
        return nums[0] if nums else None

    def build_doc_choice_clarification(
        self,
        *,
        original_question: str,
        documents: list[dict[str, str]],
    ) -> DocClarification:
        opts = [
            {
                "id": d["id"],
                "label": d.get("title") or d["id"],
                "detail": "",
                "aliases": [d.get("title", "").lower(), d["id"].lower()],
            }
            for d in documents
            if d.get("id")
        ]
        prompt = (
            "I found multiple documents you can query. Which document should I use for this question?"
        )
        return DocClarification(
            kind="document_choice",
            prompt=format_clarification_answer(prompt, opts),
            options=opts,
        )

