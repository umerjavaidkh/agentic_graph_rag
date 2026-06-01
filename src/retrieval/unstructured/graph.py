"""
retrieval/unstructured/graph.py — simplified unstructured agent.

Does semantic retrieval over embedded Sections and synthesizes an answer using
the `document_default` prompt.
"""

from langgraph.graph import END, StateGraph

from .retriever import DocumentRAGRetriever
from ...config.prompts import load_prompt
from ...config.settings import CHAT_MODEL
from ...model_providers.factory import get_model_provider
from .state import ESGState

retriever = DocumentRAGRetriever()
provider = get_model_provider()


def retrieve_node(state: ESGState):
    question = state["question"]
    user_context = state.get("user_context")

    context = retriever.hybrid_retrieve(query=question, limit=8, user_context=user_context)
    return {
        "retrieved_context": context,
        "keywords": [],
        "sources": context.get("chunks", []),
        "query_type": "semantic",
    }


def generate_node(state: ESGState):
    retrieved = state.get("retrieved_context", {}) or {}
    chunks = retrieved.get("chunks", []) or []

    if not chunks:
        return {
            "answer": "I couldn't find relevant information in the document knowledge graph.",
            "low_confidence": False,
        }

    context_lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        title = c.get("title", "Result")
        text = (c.get("text") or "").strip()
        if not text:
            continue
        context_lines.append(f"[Chunk {i}] {title}\n{text}")
    context_text = "\n\n".join(context_lines)

    system_prompt = load_prompt("document_default", context=context_text, question=state["question"])
    response = provider.chat_completion(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(state["question"])},
        ],
        temperature=0.1,
    )
    return {"answer": response.choices[0].message.content, "low_confidence": False}


def should_continue(state: ESGState):
    return "generate"


workflow = StateGraph(ESGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_conditional_edges("retrieve", should_continue, {"generate": "generate"})
workflow.add_edge("generate", END)

esg_agent = workflow.compile()

