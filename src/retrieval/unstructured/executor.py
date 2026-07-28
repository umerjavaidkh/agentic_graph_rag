from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ...conversation.clarification import format_clarification_answer


_SECTION_NUM_RE = re.compile(r"\b(\d+(?:\.\d+){1,3})\b")
# Covers financial-statement footnotes ("Note 3"), SEC filing items (which
# use a letter-suffixed numbering convention Note references don't --
# "Item 9A. Controls and Procedures", "Item 7A", "Item 1B"), and textbook
# worked examples ("Example 2.8") -- unlike Note/Item, a textbook's own
# heading is "Example 2.8" (word BEFORE a DOTTED number, not the number
# alone), so the returned "example 2.8" combined string still lines up
# with subsection.py's `title STARTS WITH` match, which a bare "2.8"
# extracted by _SECTION_NUM_RE alone would not (that title doesn't start
# with the number). \d+(?:\.\d+)*[a-z]? covers all three shapes ("3",
# "7A", "2.8") with one pattern -- the added `(?:\.\d+)*` is additive
# (zero-or-more), so existing Note/Item matches are unaffected.
_STRUCTURAL_NUM_RE = re.compile(r"\b(note|item|example)\s+(?:no\.?\s*)?(\d+(?:\.\d+)*[a-z]?)\b", re.I)
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

