"""
retrieval/structured/graph.py — Structured agent wrapper (LangGraph).

Uses the structured retriever and synthesizes a natural-language answer.
"""

from langgraph.graph import END, StateGraph

from .retriever import StructuredRetriever
from ...config.prompts import load_prompt
from ...config.settings import CHAT_MODEL
from ...model_providers.factory import get_model_provider
from .query_intent import analytics_result_limit
from .state import StructuredState

provider = get_model_provider()
retriever = StructuredRetriever()
LLM_MODEL = CHAT_MODEL

SCHEMA_PHRASES = {
    "what data", "what tables", "what is available",
    "show schema", "what can i query", "what labels",
    "what nodes", "database structure",
}


def _is_schema_query(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in SCHEMA_PHRASES)


def retrieve_node(state: StructuredState):
    question = state["question"]
    user_context = state.get("user_context")

    if _is_schema_query(question):
        context = retriever.get_schema()
        strategy = "schema"
    else:
        lim = analytics_result_limit(question, 5)
        context = retriever.retrieve(question, limit=lim, user_context=user_context)
        strategy = context.get("strategy", "text2cypher")

    cypher = None
    if context.get("chunks"):
        cypher = context["chunks"][0].get("cypher")

    return {
        "retrieved_context": context,
        "sources": context.get("chunks", []),
        "strategy": strategy,
        "cypher_generated": cypher,
        "keywords": [],
    }


def generate_node(state: StructuredState):
    chunks = state["retrieved_context"].get("chunks", [])
    question = state["question"]

    if not chunks:
        return {"answer": "I couldn't find relevant data for that query.", "low_confidence": False}

    has_error = any(c.get("id") == "error" for c in chunks)
    if has_error:
        err_chunk = next((c for c in chunks if c.get("id") == "error"), None)
        err_text = (err_chunk or {}).get("text") or "The database query failed."
        return {
            "answer": (
                "I couldn't run that query successfully.\n\n"
                f"{err_text}\n\n"
                "Try rephrasing the question or narrowing the filter."
            ),
            "low_confidence": True,
        }

    context_lines = []
    for i, c in enumerate(chunks, 1):
        title = c.get("title", "Result")
        text = c.get("text", "")
        score = c.get("score")
        cypher = c.get("cypher", "")

        meta = f"#{i}"
        if score is not None:
            meta += f" | Score: {score}"
        if cypher:
            meta += f" | Query: {cypher}"
        context_lines.append(f"{meta}\nTitle: {title}\n{text}")

    context_text = "\n\n".join(context_lines)
    system_prompt = load_prompt("structured_synthesis", context=context_text, question=question)
    response = provider.chat_completion(
        model=LLM_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(question)},
        ],
        max_tokens=600,
    )
    return {"answer": response.choices[0].message.content.strip(), "low_confidence": False}


def should_continue(state: StructuredState):
    return "generate"


workflow = StateGraph(StructuredState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_conditional_edges("retrieve", should_continue, {"generate": "generate"})
workflow.add_edge("generate", END)

structured_agent = workflow.compile()

