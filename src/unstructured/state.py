from typing import TypedDict, List, Dict, Optional
from ..auth.roles import UserContext

class ESGState(TypedDict, total=False):
    question: str
    keywords: List[str]
    retrieved_context: Dict
    answer: str
    sources: List[Dict]
    query_type: str
    user_context: Optional[UserContext]
    focus_section_id: Optional[str]
    parent_section_id: Optional[str]
    prior_context: Optional[Dict]