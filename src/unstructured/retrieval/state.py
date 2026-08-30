from typing import TypedDict, List, Dict, Optional
from ...shared.auth.roles import UserContext

class ESGState(TypedDict, total=False):
    question: str
    keywords: List[str]
    retrieved_context: Dict
    answer: str
    sources: List[Dict]
    query_type: str
    user_context: Optional[UserContext]
    # Which language corpus to search. Declared here for the same reason
    # `claims` below is: LangGraph drops any key the schema does not name,
    # so an undeclared `language` would be discarded on the way in and every
    # query would silently search the default corpus.
    language: Optional[str]
    focus_section_id: Optional[str]
    parent_section_id: Optional[str]
    document_id: Optional[str]
    prior_context: Optional[Dict]
    low_confidence: bool
    confidence_note: Optional[str]
    # Set when the question named no document and carried too few content
    # words to have implied one, so retrieval declined to guess. Declared
    # here for the same reason as `claims` below -- an undeclared key is
    # dropped, and the answer would arrive looking like any other answer.
    underspecified: bool
    # Which claim each source supports. Declared here because LangGraph drops
    # any key a node returns that the state schema does not name -- the value
    # was being computed and silently discarded.
    claims: List[Dict]
    # Documents this query might have meant, when the resolver declined to
    # pick between them. Declared for the same reason `claims` above is: a
    # node returned it, the schema did not name it, and LangGraph discarded
    # it without a word -- the picker rendered nothing and looked like a
    # resolver that had simply given up.
    document_candidates: List[Dict]
    skip_structured_guard: bool
    strategy: Optional[str]
    _autofix_agent: Optional[str]