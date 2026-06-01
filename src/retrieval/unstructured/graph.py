"""
retrieval/unstructured/graph.py — Neo4j Graph RAG agent.

Vector seed + graph expansion + LLM synthesis.
"""

from langgraph.graph import END, StateGraph

from ...routing import is_structured_data_question
from .retriever import (
    DocumentRAGRetriever,
    is_page_question,
    is_synthesis_question,
    is_toc_question,
    is_visual_page_question,
)
from ...config.prompts import load_prompt
from ...config.settings import CHAT_MODEL, RETRIEVAL_FINAL_LIMIT
from ...model_providers.factory import get_model_provider
from .state import ESGState

retriever = DocumentRAGRetriever()
provider = get_model_provider()


def retrieve_node(state: ESGState):
    question = state["question"]
    user_context = state.get("user_context")

    limit = max(RETRIEVAL_FINAL_LIMIT, 12) if is_synthesis_question(question) else RETRIEVAL_FINAL_LIMIT
    context = retriever.hybrid_retrieve(
        query=question,
        limit=limit,
        user_context=user_context,
    )
    strategy = context.get("strategy", "graph_rag")
    return {
        "retrieved_context": context,
        "keywords": [],
        "sources": context.get("chunks", []),
        "query_type": strategy,
    }


def generate_node(state: ESGState):
    question = state["question"]
    retrieved = state.get("retrieved_context", {}) or {}
    chunks = retrieved.get("chunks", []) or []

    if is_structured_data_question(question):
        return {
            "answer": (
                "This question is about the business database (products, orders, customers), "
                "not ingested PDF documents. Re-run with structured access (e.g. regular_001 or "
                "compliance_001) so the system can query product and category data."
            ),
            "low_confidence": False,
        }

    if not chunks:
        return {
            "answer": "I could not find relevant information in the ingested documents.",
            "low_confidence": False,
        }

    context_lines: list[str] = []
    for i, c in enumerate(chunks, 1):
        title = c.get("title", "Result")
        text = (c.get("text") or "").strip()
        if not text:
            continue
        rel = c.get("related") or []
        rel_note = f" (graph: {', '.join(rel)})" if rel else ""
        context_lines.append(f"[Chunk {i}] {title}{rel_note}\n{text}")
    context_text = "\n\n".join(context_lines)

    if is_toc_question(state["question"]):
        prompt_name = "document_toc"
    elif is_visual_page_question(state["question"]):
        prompt_name = "document_visual"
    elif is_page_question(state["question"]):
        prompt_name = "document_page"
    elif is_synthesis_question(state["question"]):
        prompt_name = "document_synthesis"
    else:
        prompt_name = "document_default"
    system_prompt = load_prompt(prompt_name, context=context_text, question=state["question"])
    response = provider.chat_completion(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(state["question"])},
        ],
        temperature=0.1,
        max_tokens=1400 if (
            is_toc_question(state["question"])
            or is_page_question(state["question"])
            or is_visual_page_question(state["question"])
        ) else 600,
    )
    return {"answer": response.choices[0].message.content.strip(), "low_confidence": False}


def should_continue(state: ESGState):
    return "generate"


workflow = StateGraph(ESGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_conditional_edges("retrieve", should_continue, {"generate": "generate"})
workflow.add_edge("generate", END)

esg_agent = workflow.compile()

