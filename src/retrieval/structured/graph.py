"""
retrieval/structured/graph.py — Structured agent wrapper (LangGraph).

Uses the structured retriever and synthesizes a natural-language answer.
"""

from langgraph.graph import END, StateGraph

from .retriever import StructuredRetriever
from ...config.prompts import load_prompt
from ...config.settings import (
    STRUCTURED_FAST_ANSWER,
    STRUCTURED_MODEL,
    STRUCTURED_SYNTHESIS_LONG_MAX_TOKENS,
    STRUCTURED_SYNTHESIS_MAX_TOKENS,
)
from .query_intent import estimate_structured_synthesis_max_tokens
from ...model_providers.factory import get_model_provider
from .query_intent import analytics_result_limit
from .state import StructuredState

provider = get_model_provider()
retriever = StructuredRetriever()
LLM_MODEL = STRUCTURED_MODEL

SCHEMA_PHRASES = {
    "what data", "what tables", "what is available",
    "show schema", "what can i query", "what labels",
    "what nodes", "database structure",
}


def _is_schema_query(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in SCHEMA_PHRASES)


def _build_fast_structured_answer(
    chunks: list[dict], strategy: str, question: str = ""
) -> str:
    """Format Cypher row / multistep chunks without an LLM synthesis pass."""
    if strategy == "schema" and chunks:
        return (chunks[0].get("text") or "").strip()

    parts: list[str] = []
    for chunk in chunks:
        if chunk.get("id") in ("access_denied", "error"):
            continue
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        title = (chunk.get("title") or "").strip()
        if title and title.lower() not in text.lower()[:120]:
            parts.append(f"{title}\n{text}")
        else:
            parts.append(text)

    if not parts:
        return "No matching records were found in the business database for that query."

    count = len(parts)
    header = (question or "").strip()
    intro = f"Found {count} result{'s' if count != 1 else ''} from the database query."
    if header:
        return f"{header}\n\n{intro}\n\n" + "\n\n".join(parts)
    return f"{intro}\n\n" + "\n\n".join(parts)


def _should_fast_structured_answer(chunks: list[dict], strategy: str) -> bool:
    if not STRUCTURED_FAST_ANSWER:
        return False
    if strategy in ("schema", "text2cypher", "multistep") and chunks:
        if any(c.get("id") in ("access_denied", "error") for c in chunks):
            return False
        if strategy == "multistep" and any(
            str(c.get("id", "")).endswith("_error") for c in chunks
        ):
            return False
        return True
    return False


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
    retrieved_context = state.get("retrieved_context") or {}
    chunks = retrieved_context.get("chunks", [])
    question = state["question"]

    if (retrieved_context.get("mode") or "") == "needs_clarification":
        # Return the clarification prompt as-is (no LLM synthesis).
        if chunks:
            return {"answer": (chunks[0].get("text") or "").strip(), "low_confidence": False}
        return {
            "answer": "I need one clarification before I can answer that.",
            "low_confidence": False,
        }

    if not chunks:
        return {
            "answer": "No matching records were found in the business database for that query.",
            "low_confidence": False,
        }

    denied = next((c for c in chunks if c.get("id") == "access_denied"), None)
    if denied:
        return {
            "answer": (denied.get("text") or "Access denied for structured data.").strip(),
            "low_confidence": False,
        }

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

    strategy = state.get("strategy") or retrieved_context.get("strategy") or "text2cypher"
    if _should_fast_structured_answer(chunks, strategy):
        return {
            "answer": _build_fast_structured_answer(chunks, strategy, question),
            "low_confidence": False,
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
        max_tokens=estimate_structured_synthesis_max_tokens(
            question,
            chunk_count=len(chunks),
            default_max=STRUCTURED_SYNTHESIS_MAX_TOKENS,
            long_max=STRUCTURED_SYNTHESIS_LONG_MAX_TOKENS,
        ),
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

