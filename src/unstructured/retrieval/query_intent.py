"""Question-shape detectors for document RAG routing."""
from __future__ import annotations

import re

from ...shared.language import intent_alternations, normalize_arabic

from ..document.page_numbers import parse_page_number_from_query
from .text_utils import query_anchor_terms
from .visual_retrieval import parse_visual_intent

def _with_profiles(shape: str, english: str) -> str:
    """The English pattern, plus every registered profile's for this shape.

    Unioned rather than selected between, and therefore needing no
    language argument. A pattern written in Arabic script cannot match
    ASCII and an English one cannot match Arabic, so the alternation is
    free of cross-language false positives -- and English's own accurate
    set is untouched, which is what the design asks for.

    Threading a language into all 77 shape checks would have been the
    branch the design forbids, wearing a parameter as a disguise.
    """
    extra = intent_alternations(shape)
    if not extra:
        return english
    # Each side gets its own non-capturing group. The English patterns end
    # in `\b(...)\b` and the profile ones do not; without grouping the
    # trailing boundary would bind to the first Arabic alternative and
    # quietly change what English matches.
    return "|".join(f"(?:{part})" for part in (english, *extra))


def _shape_query(query: str) -> str:
    """The query in the space the profile patterns are written in.

    Safe to apply unconditionally: the Arabic normalizer only rewrites
    Arabic code points, so it is the identity on English -- asserted in
    tests/shared/test_arabic_normalization_unit.py rather than assumed.
    """
    return normalize_arabic(query or "")


_SYNTHESIS_RE = re.compile(
    _with_profiles(
        "synthesis",
    r"\b(synthesi[sz]|structural map|escalat|pathway|flowchart|flow chart|"
    r"compare|contrast|relationship between|trace how|build a .{0,20}map|"
    r"how .{0,40} connect|map showing)\b",
    ),
    re.I,
)

_ENUMERATION_RE = re.compile(
    _with_profiles(
        "enumeration",
    r"\b(list\s+all|enumerate|name\s+all|distinct|what\s+are\s+all|"
    r"which\s+(?:\w+\s+){0,3}?(?:examples|sections|pages|items|cases|instances)\b|"
    r"all\s+(?:the\s+)?(?:examples|sections|instances|cases))\b",
    ),
    re.I,
)
# "Which worked examples apply the standard-error formula" reads exactly
# like "which section discusses X" in surface form, but wants ALL matching
# instances, not the single best one -- the plural noun after "which" is
# the signal (a document-agnostic heuristic, not tied to "worked example"
# specifically). Verified live: this phrasing previously fell through to
# is_enumeration_question's original literal "list all/enumerate/name
# all/distinct" set entirely, so full_hybrid.py's existing fetch_limit
# bump for enumeration questions (see retrieve()) never activated, and
# only 1 of 3 structurally-parallel "Worked Example N" sections survived
# ranking into the final top-8 context -- not a ranking bug in the
# individual scores, just this detector never firing for the phrasing.

# "What does this document/chapter discuss overall" — deliberately separate
# from _SYNTHESIS_RE (which is tuned for compare/contrast/structural-map
# phrasing and also drives unrelated retrieval weighting elsewhere in
# ranking.py) so this narrower detector can gate the chapter-summary
# rollup feature without changing behavior for any existing question shape.
_OVERVIEW_RE = re.compile(
    _with_profiles(
        "overview",
    r"\bwhat\s+(?:does|is|are)\s+(?:this|the)\b.{0,30}\b"
    r"(?:document|filing|report|chapter|section|10-?k|10-?q|annual\s+report)\b"
    r".{0,40}\b(?:discuss|about|cover|contain|address)\b|"
    r"\bwhat\s+is\s+this\s+(?:document|filing|report)\s+about\b|"
    r"\bsummar(?:y|ize|ise)\s+(?:this|the)\s+(?:document|filing|report|chapter)\b|"
    r"\b(?:overview|gist|summary)\s+of\s+(?:this|the)\s+(?:document|filing|report|chapter)\b",
    ),
    re.I,
)

_CONTRAST_COMPARE_RE = re.compile(
    r"\b(contrast|compare|comparison|versus|vs\.?)\b",
    re.I,
)

# When was this filing actually SUBMITTED to the SEC — distinct from the
# period-end date the filing covers ("for the quarter ended March 29,
# 2026"). The real filing/submission date is usually not printed anywhere
# in the PDF body at all (EDGAR's "Filed:" stamp lives in the filing's
# HTML/index wrapper, not the document itself) — retrieval was pulling
# whichever date-heavy MD&A chunk ranked highest and the LLM picked the most
# prominent date in it (almost always the period-end date), not the actual
# answer. See strategies/filing_date.py, which answers this from
# DocRevision.source_filename instead of guessing from prose. Deliberately
# narrow: "period ended"/"quarter ended" phrasing must NOT match here, since
# that's a different, legitimately-in-the-text question.
_FILING_DATE_RE = re.compile(
    r"\bfiling\s+date\b|\bdate\s+(?:of\s+)?filing\b|\bdate\s+filed\b|"
    r"\bfiled\s+on\b|\bwhen\s+(?:was|is)\s+(?:this|it|the)\b[^.?]{0,30}\bfiled\b",
    re.I,
)

# A question about a company's OWN firmwide financial metric (net earnings,
# revenue, EPS, ROE, book value, total assets, …) that names no specific
# business segment or geographic region. The authoritative figure lives in a
# summary section (Executive/Financial Overview, Financial Highlights,
# Consolidated Statements), but that long narrative loses vector-cosine
# ranking to short, focused segment tables that literally repeat the metric
# name as a row label — so a plain "net earnings for 2025" drifts to a
# *segment's* net earnings. full_hybrid.py uses this to pull the firmwide
# summary section into the candidate pool and pin it (see
# FinancialSummaryService and RankingService._pin_firmwide_summary_chunks).
_FIRMWIDE_METRIC_RE = re.compile(
    r"\b(net\s+earnings|net\s+income|net\s+revenues?|total\s+(?:net\s+)?revenues?|"
    r"earnings\s+per\s+(?:common\s+)?share|diluted\s+eps|\beps\b|"
    r"return\s+on\s+(?:average\s+)?(?:common\s+)?equity|\broe\b|"
    r"return\s+on\s+(?:average\s+)?assets|\broa\b|"
    r"book\s+value(?:\s+per\s+(?:common\s+)?share)?|"
    r"total\s+assets|provision\s+for\s+credit\s+losses|"
    r"effective\s+(?:income\s+)?tax\s+rate)\b",
    re.I,
)

# A quarter-by-quarter metric question (net income/revenue/EPS Q1-Q4). The
# authoritative figures live in the "Selected Quarterly Financial Data
# (Unaudited)" table — a standardized 10-K disclosure (former Reg S-K Item
# 302) — but it's a terse, number-dense table with a generic wrapping
# section title ("Supplementary information"), so it loses vector-cosine
# ranking to narrative annual-summary sections that merely mention the same
# metric name once. full_hybrid.py uses this to pull that table into the
# candidate pool and pin it (see FinancialSummaryService.
# fetch_quarterly_for_document and the shared _pin_firmwide_summary_chunks
# pinner, reused as-is since its mechanics aren't firmwide-specific).
_QUARTERLY_METRIC_RE = re.compile(
    r"\b(net\s+earnings|net\s+income|net\s+revenues?|total\s+(?:net\s+)?revenues?|"
    r"earnings\s+per\s+(?:common\s+)?share|diluted\s+eps|\beps\b|earnings)\b",
    re.I,
)
_QUARTERLY_PERIOD_RE = re.compile(
    r"\bquarter(?:ly)?\b|\bq[1-4]\b|"
    r"\b(?:first|second|third|fourth)\s+quarter\b",
    re.I,
)

# If any of these appear, the question is explicitly scoped to a segment,
# business line, or geography — the firmwide summary is NOT what it wants, so
# the boost must not fire. Segment names are Goldman-flavored but the generic
# "by segment / by region / operating segment" phrasing is issuer-agnostic.
_SEGMENT_SCOPE_RE = re.compile(
    r"\b(global\s+banking|asset\s*&?\s*(?:and\s+)?wealth|platform\s+solutions|"
    r"consumer\s+banking|by\s+(?:geographic\s+)?region|geographic|"
    r"per\s+segment|by\s+segment|each\s+segment|which\s+segment|"
    r"operating\s+segment|business\s+segment)\b",
    re.I,
)

_KEYWORD_STOP = frozenset({
    # Original project-specific additions.
    "what", "which", "where", "when", "that", "this", "with", "from", "into",
    "have", "been", "were", "they", "their", "there", "about", "under", "based",
    "specific", "according", "should", "would", "could", "document", "text",
    "showing", "single", "show", "build", "does", "explicitly", "detailed",
    # Common English stopwords (3+ letters — shorter ones are already
    # dropped by _query_keywords' length-3 regex) that were simply missing
    # here, diluting full-text/lexical relevance scoring on every question
    # containing them (not specific to any one document or query) — e.g.
    # "who" and "are" were absent even though "what"/"which"/"were" were
    # already filtered.
    "who", "how", "are", "was", "the", "and", "but", "for", "not", "all",
    "any", "can", "her", "him", "his", "she", "you", "your", "our", "out",
    "off", "over", "then", "than", "such", "some", "each", "other", "only",
    "own", "too", "very", "will", "just", "now", "these", "those", "here",
    "more", "once", "through", "until", "while", "whom", "why", "being",
    "having", "both", "few", "nor", "same",
    # Generic directional/temporal connector words from a question's own
    # natural phrasing ("break down across the year") rather than its
    # actual subject matter — low information value as search keywords
    # regardless of document domain, and IDF weighting alone doesn't
    # reliably discount them: a formal document's real subject-matter
    # vocabulary (e.g. "net", "income" in a financial filing) is often
    # itself very common throughout that document, while these connector
    # words happen to cluster in narrative/prose sections and be
    # comparatively rarer — inverting the intended weighting rather than
    # fixing it. "year"/"years" specifically: the actual year number
    # (e.g. "2016") is already captured separately via date-pattern
    # matching, so the bare word isn't needed as its own keyword. "break"
    # is part of the same phrasal verb ("break down") as "down" above —
    # its rarity as a literal word in formal financial prose (vs. its very
    # common use in casual questions) makes it get an outsized IDF weight
    # in structural_keyword_retrieve, letting one incidental match swamp
    # several genuinely on-topic term matches (verified: JPM 10-K query
    # "...quarterly net income break down across the year..." — "break"
    # alone had ~5x the weight of "quarterly", the next-rarest real term).
    "down", "across", "away", "back", "year", "years", "break",
})

_TOC_RE = re.compile(
    _with_profiles(
        "toc",
    r"\b(table\s+of\s+contents?|\btoc\b|list\s+(?:all\s+)?(?:the\s+)?contents?|"
    r"show\s+(?:me\s+)?(?:the\s+)?contents?|provide\s+(?:me\s+)?(?:all\s+)?(?:the\s+)?toc|"
    # Structural-outline phrasing that doesn't say "contents"/"toc" but
    # means the same thing -- "what sections does this doc have", "list
    # every heading" -- was falling through to the relevance-ranked hybrid
    # strategy instead of TocStrategy, which is the only one that walks
    # the graph in actual document order rather than by match score.
    r"(?:list|show)\s+(?:me\s+)?(?:all\s+|every\s+)?(?:the\s+)?(?:sections?|headings?)\b|"
    r"what\s+(?:sections?|headings?)\s+(?:does|do|is|are)\b|"
    r"what\s+(?:are\s+the|is\s+the)\s+(?:sections?|headings?))\b",
    ),
    re.I,
)

_PAGE_QUERY_RE = re.compile(
    _with_profiles(
        "page",
    r"\b(?:fetch|get|show|retrieve|read|content|text|everything|all)\b.{0,50}\bpage\b|"
    r"\bpage\s+[\wivxlcdm\-]+\s+(?:of|from|in)\b|"
    r"\bcontent\s+(?:from|on|of)\s+(?:pdf\s+)?page\b|"
    r"\bwhat\s+(?:is|does)\s+(?:pdf\s+)?page\s+",
    ),
    re.I,
)

_FIG_NUMBER_RE = re.compile(r"\b(?:fig\.?|figure)\s*\d+(?:\.\d+)?\b", re.I)

_VISUAL_PAGE_RE = re.compile(
    r"\bvisual\s+content\b|"
    r"\b(?:fig\.?|figure)\s*\d+(?:\.\d+)?\b|"
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
    return bool(_SYNTHESIS_RE.search(_shape_query(query)))


def is_overview_question(query: str) -> bool:
    return bool(_OVERVIEW_RE.search(_shape_query(query)))


def is_filing_date_question(query: str) -> bool:
    return bool(_FILING_DATE_RE.search(query or ""))


def is_firmwide_financial_metric_question(query: str) -> bool:
    """A firmwide financial-metric question that names no segment/region."""
    q = query or ""
    return bool(_FIRMWIDE_METRIC_RE.search(q)) and not _SEGMENT_SCOPE_RE.search(q)


def is_quarterly_breakdown_question(query: str) -> bool:
    """A quarter-by-quarter metric question (net income/revenue/EPS Q1-Q4)."""
    q = query or ""
    return bool(_QUARTERLY_PERIOD_RE.search(q)) and bool(_QUARTERLY_METRIC_RE.search(q))


def is_enumeration_question(query: str) -> bool:
    return bool(_ENUMERATION_RE.search(_shape_query(query)))


def is_toc_question(query: str) -> bool:
    return bool(_TOC_RE.search(_shape_query(query)))


def is_page_question(query: str) -> bool:
    pdf_page, doc_page = parse_page_number_from_query(query)
    if pdf_page is not None or doc_page:
        return True
    return bool(_PAGE_QUERY_RE.search(_shape_query(query)))


def is_fact_lookup_question(query: str) -> bool:
    return bool(_FACT_LOOKUP_RE.search(query or ""))


def is_visual_page_question(query: str) -> bool:
    """Page-scoped question focused on figures/images, not plain page text."""
    if not _VISUAL_PAGE_RE.search(query or ""):
        return False
    pdf_page, doc_page = parse_page_number_from_query(query)
    if pdf_page is not None or doc_page:
        return True
    # A bare "what does Figure 1 show" has no page number to give, but a
    # named figure is still resolvable — the page gets found by figure
    # number instead (see PageStrategy._resolve_pdf_page_by_figure_number).
    if _FIG_NUMBER_RE.search(query or ""):
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
    "is_overview_question",
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
