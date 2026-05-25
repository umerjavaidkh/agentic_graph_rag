from langgraph.graph import StateGraph, END
from agent.state import ESGState
from agent.retriever import ESGComplianceRetriever
import re
from typing import List
import openai
import os

retriever = ESGComplianceRetriever()
openai.api_key = os.getenv("OPENAI_API_KEY")

# ─────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────
MIN_CONFIDENCE_SCORE = 0.30   # below this → low confidence warning to LLM
MIN_CHUNKS_REQUIRED  = 1      # below this → no_context branch

TOC_PHRASES = {
    "table of contents", "full table of contents", "list all sections",
    "document structure", "show all sections", "all sections",
    "list sections", "what sections", "what chapters", "what topics",
    "overview of", "structure of",
}

STRUCTURAL_PHRASES = {
    "what is in chapter", "what is in section", "contents of chapter",
    "contents of section", "show chapter", "show section",
    "summarize chapter", "summarize section",
}


# ─────────────────────────────────────────
# QUERY CLASSIFIER
# ─────────────────────────────────────────
def classify_query(question: str) -> str:
    """
    Returns: 'toc' | 'structural' | 'semantic'
    Uses substring matching — no LLM cost.
    """
    q = question.lower().strip()

    # TOC: contains any TOC phrase anywhere in question
    if any(phrase in q for phrase in TOC_PHRASES):
        return "toc"

    # Structural: asking about a specific chapter/section by name
    if any(phrase in q for phrase in STRUCTURAL_PHRASES):
        return "structural"

    return "semantic"


# ─────────────────────────────────────────
# NODES
# ─────────────────────────────────────────
def retrieve_node(state: ESGState):
    question    = state["question"]
    query_type  = classify_query(question)

    if query_type == "toc":
        context  = retriever.get_all_sections()
        keywords = ["toc"]

    elif query_type == "structural":
        # Pure graph traversal — no embedding needed
        context  = retriever.get_all_sections()
        keywords = ["structural"]

    else:
        # Semantic: multi-hop for complex questions
        context  = retriever.multi_hop_retrieve(
            query=question,
            limit=5,
            hops=2,
        )
        keywords = _extract_keywords(question)

    return {
        "retrieved_context": context,
        "keywords":          keywords,
        "sources":           context["chunks"],
        "query_type":        query_type,
    }


def generate_node(state: ESGState):
    chunks     = state["retrieved_context"]["chunks"]
    query_type = state.get("query_type", "semantic")

    if not chunks:
        return {"answer": "I couldn't find relevant compliance information in the document."}

    # ── Confidence check (semantic queries only) ──────────────
    low_confidence = False
    if query_type == "semantic":
        top_score = max((c.get("score", 0) for c in chunks), default=0)
        if top_score < MIN_CONFIDENCE_SCORE:
            low_confidence = True

    context_text = "\n\n".join([
        f"[Section: {c['title']}]\n{c['text']}"
        for c in chunks
    ])

    # ── System prompt adapts to query type + confidence ───────
    if query_type == "toc":
        system_prompt = (
            "You are a document navigator. List the sections clearly and concisely "
            "based only on the provided section titles and page numbers."
        )
    elif low_confidence:
        system_prompt = (
            "You are an ESG Compliance Analyst for STRATEC. "
            "The retrieved sections may not be a perfect match for the question. "
            "Answer as best you can using ONLY the provided sections, and clearly "
            "state if the exact answer is not present. Cite section titles."
        )
    else:
        system_prompt = (
            "You are an ESG Compliance Analyst for STRATEC. "
            "Answer using ONLY the provided document sections. Cite section titles. "
            "If the answer is not in the context, say so clearly."
        )

    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"Context:\n{context_text}\n\nQuestion: {state['question']}"},
        ],
        temperature=0.1,
    )

    return {
        "answer":         response.choices[0].message.content,
        "low_confidence": low_confidence,
    }


def should_continue(state: ESGState):
    chunks = state["retrieved_context"]["chunks"]
    if len(chunks) < MIN_CHUNKS_REQUIRED:
        return "no_context"
    return "generate"


# ─────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────
def _extract_keywords(question: str) -> List[str]:
    stopwords = {
        "what", "are", "the", "that", "this", "with", "for", "and", "or",
        "in", "of", "to", "a", "an", "is", "it", "section", "document",
        "from", "do", "does", "did", "can", "could", "would", "should",
        "will", "have", "has", "had", "be", "been", "being", "members",
        "ask", "themselves", "their", "they", "we", "our", "us", "me", "my", "i",
    }
    cleaned = re.sub(r'[^a-z\s]', '', question.lower())
    return [w for w in cleaned.split() if w not in stopwords and len(w) > 2][:5]


# ─────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────
workflow = StateGraph(ESGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_conditional_edges(
    "retrieve",
    should_continue,
    {"no_context": END, "generate": "generate"},
)
workflow.add_edge("generate", END)

# NO MEMORY — each query is independent
esg_agent = workflow.compile()