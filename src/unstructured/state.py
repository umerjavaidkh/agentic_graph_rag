from typing import TypedDict, List, Dict, Optional
from ..auth.roles import UserContext

class ESGState(TypedDict, total=False):
    question: str
    keywords: List[str]
    retrieved_context: Dict
    answer: str
    sources: List[Dict]
    user_context: Optional[UserContext]