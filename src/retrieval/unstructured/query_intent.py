"""Question-shape detectors for document RAG routing."""
from __future__ import annotations

import re

from ...document.page_numbers import parse_page_number_from_query
from .text_utils import query_anchor_terms
from .visual_retrieval import parse_visual_intent

_SYNTHESIS_RE = re.compile(
    r"\b(synthesi[sz]|structural map|escalat|pathway|flowchart|flow chart|"
    r"compare|contrast|relationship between|trace how|build a .{0,20}map|"
    r"how .{0,40} connect|map showing)\b",
    re.I,
)

_ENUMERATION_RE = re.compile(
    r"\b(list\s+all|enumerate|name\s+all|distinct)\b",
    re.I,
)

_CONTRAST_COMPARE_RE = re.compile(
    r"\b(contrast|compare|comparison|versus|vs\.?)\b",
    re.I,
)

_KEYWORD_STOP = frozenset({
    "what", "which", "where", "when", "that", "this", "with", "from", "into",
    "have", "been", "were", "they", "their", "there", "about", "under", "based",
    "specific", "according", "should", "would", "could", "document", "text",
    "showing", "single", "show", "build", "does", "explicitly", "detailed",
})

_TOC_RE = re.compile(
    r"\b(table\s+of\s+contents?|\btoc\b|list\s+(?:all\s+)?(?:the\s+)?contents?|"
    r"show\s+(?:me\s+)?(?:the\s+)?contents?|provide\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?toc)\b",
    re.I,
)

_PAGE_QUERY_RE = re.compile(
    r"\b(?:fetch|get|show|retrieve|read|content|text|everything|all)\b.{0,50}\bpage\b|"
    r"\bpage\s+[\wivxlcdm\-]+\s+(?:of|from|in)\b|"
    r"\bcontent\s+(?:from|on|of)\s+(?:pdf\s+)?page\b|"
    r"\bwhat\s+(?:is|does)\s+(?:pdf\s+)?page\s+",
    re.I,
)

_VISUAL_PAGE_RE = re.compile(
    r"\bvisual\s+content\b|"
    r"\b(?:all\s+)?(?:the\s+)?(?:images?|figures?|figs?\.?|diagrams?|charts?|photos?|pictures?|visuals?)\b.{0,40}\bpage\b|"
    r"\bpage\b.{0,40}\b(?:images?|figures?|visual|diagram)\b|"
    r"\b(?:tell\s+me|describe|explain).{0,60}\b(?:image|figure|diagram)\b|"
    r"\babout\s+(?:that|the)\s+(?:image|figure|diagram)\b",
    re.I,
)

_FIG_CAPTION_RE = re.compile(
    r"(?:Fig\.?|Figure)\s*(\d+(?:\.\d+)?)\s*[:.]\s*([^\n]+)",
    re.I,
)

_FACT_LOOKUP_RE = re.compile(
    r"\b(?:url|link|website|web\s*site|portal|email|e-mail|hyperlink)\b|"
    r"\bwhat\s+is\s+the\s+(?:url|link|website|address|portal)\b|"
    r"\b(?:which|into\s+which|how\s+many|when\s+did|who\s+hosted)\b|"
    r"\b(?:translated|translation|languages?|hosted|host|workshop)\b",
    re.I,
)

_PHRASE_STOP = _KEYWORD_STOP | frozenset({
    "url", "link", "website", "portal", "email", "address", "http", "https",
    "into", "which", "what", "when", "who", "how", "many", "much",
    "the", "for", "has", "been", "was", "were", "does", "did", "are", "any",
    "whose", "that", "this", "with", "from", "than", "then", "also", "only",
    "name", "list", "give", "tell", "say", "ask",
})

_MONTH_YEAR_RE = re.compile(
    r"\b(january|february|march|april|may|june|july|august|september|october|"
    r"november|december)\s+(20\d{2}|19\d{2})\b",
    re.I,
)


def is_synthesis_question(query: str) -> bool:
    return bool(_SYNTHESIS_RE.search(query or ""))


def is_enumeration_question(query: str) -> bool:
    return bool(_ENUMERATION_RE.search(query or ""))


def is_toc_question(query: str) -> bool:
    return bool(_TOC_RE.search(query or ""))


def is_page_question(query: str) -> bool:
    pdf_page, doc_page = parse_page_number_from_query(query)
    if pdf_page is not None or doc_page:
        return True
    return bool(_PAGE_QUERY_RE.search(query or ""))


def is_fact_lookup_question(query: str) -> bool:
    return bool(_FACT_LOOKUP_RE.search(query or ""))


def is_visual_page_question(query: str) -> bool:
    """Page-scoped question focused on figures/images, not plain page text."""
    if not _VISUAL_PAGE_RE.search(query or ""):
        return False
    pdf_page, doc_page = parse_page_number_from_query(query)
    if pdf_page is not None or doc_page:
        return True
    intent = parse_visual_intent(query)
    return intent.wants_image or intent.pdf_page is not None


# Re-export for ranking / lexical modules
__all__ = [
    "CONTRAST_COMPARE_RE",
    "FIG_CAPTION_RE",
    "KEYWORD_STOP",
    "MONTH_YEAR_RE",
    "PHRASE_STOP",
    "is_enumeration_question",
    "is_fact_lookup_question",
    "is_page_question",
    "is_synthesis_question",
    "is_toc_question",
    "is_visual_page_question",
    "query_anchor_terms",
]

# Alias exports used by mixins (match old private names)
CONTRAST_COMPARE_RE = _CONTRAST_COMPARE_RE
FIG_CAPTION_RE = _FIG_CAPTION_RE
KEYWORD_STOP = _KEYWORD_STOP
PHRASE_STOP = _PHRASE_STOP
MONTH_YEAR_RE = _MONTH_YEAR_RE
