from typing import TypedDict, List, Dict

class ESGState(TypedDict):
    question: str
    keywords: List[str]
    retrieved_context: Dict
    answer: str
    sources: List[Dict]