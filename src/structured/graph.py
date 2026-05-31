"""
structured/graph.py — LangGraph agent for structured graph queries.

Handles:
  - Aggregation queries  (total, count, average, sum)
  - Lookup queries       (find product X, show orders for customer Y)
  - Semantic queries     (find products similar to seafood)
  - Schema queries       (what data is available?)
"""
from langgraph.graph import StateGraph, END
from .state import StructuredState
from .retriever import StructuredRetriever
from ..config.prompts import load_prompt
from ..config.settings import CHAT_MODEL, STRUCTURED_FAST_ANSWER
from ..model_providers.factory import get_model_provider
from .fast_answer import try_tabular_answer
from .query_intent import analytics_result_limit
from .clarification import needs_structured_clarification
from ..conversation.clarification import format_clarification_answer

provider = get_model_provider()
retriever = StructuredRetriever()
LLM_MODEL = CHAT_MODEL


# ─────────────────────────────────────────
# QUERY CLASSIFIER
# ─────────────────────────────────────────
SCHEMA_PHRASES = {
    "what data", "what tables", "what is available",
    "show schema", "what can i query", "what labels",
    "what nodes", "database structure",
}

def _is_schema_query(question: str) -> bool:
    q = question.lower()
    return any(p in q for p in SCHEMA_PHRASES)


# ─────────────────────────────────────────
# NODES
# ─────────────────────────────────────────
def retrieve_node(state: StructuredState):
    question = state["question"]
    user_context = state.get("user_context")

    if _is_schema_query(question):
        context  = retriever.get_schema()
        strategy = "schema"
    else:
        spec = needs_structured_clarification(question)
        if spec:
            message = format_clarification_answer(
                spec["prompt"],
                spec["options"],
                footer="Reply with an option name or number (e.g. **By product**).",
            )
            context = {
                "mode": "needs_clarification",
                "clarification_kind": spec["kind"],
                "clarification_message": message,
                "clarification_options": spec["options"],
                "original_question": question,
                "chunks": [],
                "strategy": "clarification",
            }
            return {
                "retrieved_context": context,
                "sources": [],
                "strategy": "clarification",
                "cypher_generated": None,
                "keywords": [],
            }

        lim = analytics_result_limit(question, 5)
        context  = retriever.retrieve(question, limit=lim, user_context=user_context)
        strategy = context.get("strategy", "text2cypher")

    # Extract generated cypher for logging if available
    cypher = None
    if context["chunks"]:
        cypher = context["chunks"][0].get("cypher")

    return {
        "retrieved_context": context,
        "sources":           context["chunks"],
        "strategy":          strategy,
        "cypher_generated":  cypher,
        "keywords":          [],
    }


def generate_node(state: StructuredState):
    chunks   = state["retrieved_context"]["chunks"]
    strategy = state.get("strategy", "text2cypher")
    question = state["question"]
    retrieved = state["retrieved_context"]

    if strategy == "clarification" or retrieved.get("mode") == "needs_clarification":
        msg = retrieved.get("clarification_message") or (
            "I need one more detail before I can query the structured database. Please pick an option."
        )
        return {"answer": msg, "low_confidence": False}

    if not chunks:
        return {
            "answer":         "I couldn't find relevant data for that query.",
            "low_confidence": False,
        }

    has_error = any(c.get("id") == "error" for c in chunks)

    if STRUCTURED_FAST_ANSWER and not has_error and strategy == "text2cypher":
        fast = try_tabular_answer(chunks, question=question)
        if fast:
            return {"answer": fast, "low_confidence": False}

    # Build context with explicit metadata so LLM knows what it is looking at
    context_lines = []
    for i, c in enumerate(chunks, 1):
        title = c.get("title", "Result")
        text  = c.get("text", "")
        label = c.get("label", "")
        score = c.get("score")
        cypher = c.get("cypher", "")
        
        meta = f"#{i}"
        if label:
            meta += f" | Type: {label}"
        if score is not None:
            meta += f" | Score: {score:.3f}"
        if cypher:
            meta += f" | Query: {cypher}"
        
        context_lines.append(f"{meta}\nTitle: {title}\n{text}")
    
    context_text = "\n\n".join(context_lines)

    # ── IMPROVED SYSTEM PROMPTS ─────────────────────────────
    if has_error:
        system_prompt = load_prompt("structured_synthesis", context=context_text, question=question)
    elif strategy == "schema":
        system_prompt = load_prompt("structured_synthesis", context=context_text, question=question)
    elif strategy == "vector":
        system_prompt = load_prompt("structured_synthesis", context=context_text, question=question)
    else:
        system_prompt = load_prompt("structured_synthesis", context=context_text, question=question)

    response = provider.chat_completion(
        model=LLM_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": f"{question}"},
        ],
        max_tokens=600,
    )

    return {
        "answer":         response.choices[0].message.content.strip(),
        "low_confidence": has_error,
    }


def should_continue(state: StructuredState):
    # Always run generate_node — clarification and empty retrieval still need a reply.
    return "generate"


# ─────────────────────────────────────────
# GRAPH
# ─────────────────────────────────────────
workflow = StateGraph(StructuredState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_conditional_edges(
    "retrieve",
    should_continue,
    {"generate": "generate"},
)
workflow.add_edge("generate", END)

structured_agent = workflow.compile()