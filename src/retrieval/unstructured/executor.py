from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Optional

from ...conversation.clarification import format_clarification_answer


_SECTION_NUM_RE = re.compile(r"\b(\d+(?:\.\d+){1,3})\b")
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
        m = _SECTION_NUM_RE.search(query or "")
        return m.group(1) if m else None

    def is_subsection_request(self, query: str) -> bool:
        q = query or ""
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

