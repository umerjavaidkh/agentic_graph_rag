"""
retrieval/unstructured/graph.py — Neo4j Graph RAG agent.

Vector seed + graph expansion + LLM synthesis.
"""

from langgraph.graph import END, StateGraph

import math
import re

from ...interface.routing import document_agent_structured_guard, has_document_cue, structured_entity_summary
from .retriever import (
    DocumentRAGRetriever,
    is_page_question,
    is_synthesis_question,
    is_toc_question,
    is_visual_page_question,
)
from ...shared.config.prompts import load_prompt
from ...shared.config.settings import (
    CHAT_MODEL,
    DOCUMENT_SYNTHESIS_CONTEXT_MAX_CHARS,
    DOCUMENT_SYNTHESIS_LONG_MAX_TOKENS,
    DOCUMENT_SYNTHESIS_MAX_TOKENS,
    RETRIEVAL_FINAL_LIMIT,
)
from ...shared.model_providers.factory import get_chat_provider
from ...shared.telemetry import pipeline_step
from .state import ESGState
from .verification import compute_confidence

retriever = DocumentRAGRetriever()
provider = get_chat_provider()

_STRUCTURED_MISROUTE = re.compile(
    r"not in the document corpus|use structured data access",
    re.I,
)

# Retrieval modes where chunks are already the answer (TOC, page, box, subsection).
_STRUCTURAL_FAST_MODES = frozenset({
    "structural_toc",
    "structural_page",
    "structural_page_visual",
    "page_visual_list",
    "structural_box_list",
    "structural_box_content",
    "subsection_tree",
    "section_detail",
    "structural_filing_date",
    "needs_clarification",
})


def _build_fast_unstructured_answer(chunks: list[dict]) -> str:
    parts: list[str] = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        title = (chunk.get("title") or "").strip()
        if title and title.lower() not in text.lower()[:80]:
            parts.append(f"**{title}**\n{text}")
        else:
            parts.append(text)
    return "\n\n".join(parts).strip()


def _fix_misrouted_structured_answer(answer: str, question: str) -> str:
    """LLM sometimes mis-applies the structured-data redirect on document
    questions."""
    if not _STRUCTURED_MISROUTE.search(answer or ""):
        return (answer or "").strip()
    if not has_document_cue(question):
        return (answer or "").strip()
    return (
        # Names no dataset: this text reaches the user, and the structured
        # graph is whatever schema happens to be loaded (it was Northwind,
        # it is now an e-commerce dataset, it will be a customer's own).
        # Naming one made the reply wrong for every other deployment.
        "This is a document question (ingested PDF content), not the structured business database. "
        "I searched the ingested document sections but could not find the exact figure or detail you asked for. "
        "Try rephrasing with a section number or page reference if you have one."
    )


def retrieve_node(state: ESGState):
    question = state["question"]
    user_context = state.get("user_context")
    document_id_hint = state.get("document_id") or ""

    limit = max(RETRIEVAL_FINAL_LIMIT, 12) if is_synthesis_question(question) else RETRIEVAL_FINAL_LIMIT
    with pipeline_step("document.graph.retrieve", limit=limit):
        context = retriever.hybrid_retrieve(
            query=question,
            limit=limit,
            user_context=user_context,
            document_id_hint=document_id_hint,
        )
    strategy = context.get("strategy", "graph_rag")

    # When the question named no document clearly, carry the plausible ones
    # through so the caller can offer the choice. Guessing is measurably
    # worse at corpus scale: unscoped, a question about a decedent's return
    # was answered from an IRS medical-expenses publication, and answered
    # "not covered" -- which reads as the corpus lacking the answer rather
    # than as the wrong document having been searched.
    # Only when retrieval could NOT place the question. Building this list
    # walks every document's subtree applying a regex per content node --
    # 17.97s of a 20.09s query, measured on 998 documents -- and it was run
    # on every first-turn question, including the ones that had already
    # resolved a document and had nothing to disambiguate. Retrieval now
    # answers the same question far more cheaply, so ask it first and pay
    # for the scan only when it comes back empty.
    # The vector-scoped strategies already rank candidate documents as a
    # side effect of scoping, so take theirs when they have them -- it is
    # free, and it is the same question asked a cheaper way.
    candidates: list = list(context.get("document_candidates") or [])
    if not candidates and not document_id_hint and not context.get("document_id"):
        try:
            candidates = retriever.document_candidates(question, user_context=user_context)
        except Exception:  # never fail a working answer over a suggestion
            candidates = []

    return {
        "retrieved_context": context,
        "keywords": [],
        "sources": context.get("chunks", []),
        "query_type": strategy,
        "document_candidates": candidates,
    }


_CITATION_STOPWORDS = frozenset(
    "the a an and or of to in for on with is are was were be been this that these "
    "those it its as at by from not no which who whom whose what when where how "
    "section covers cover document report page pages".split()
)


# A digit before the period is a section number, not a sentence end:
# splitting on "2.4. Interoperability" produced a fragment ending at
# '"2.4.' and attributed the rest to a different page.
_SENTENCE_SPLIT = re.compile(r"(?<![0-9].)(?<=[.!?])\s+(?=[A-Z])")


def _claim_citations(answer: str, chunks: list[dict]) -> list[dict]:
    """Attribute each sentence of the answer to the chunk that supports it.

    A flat source list cannot say WHICH page supports which claim, so a reader
    checking a four-page answer has to read all four -- which removes the
    property that makes citation worth having. Per-claim attribution is also
    sharper to compute: matching one sentence against a chunk is a much
    narrower problem than matching a whole answer, where every chunk shares
    some vocabulary with something.

    Sentences with no confident support are returned with source_id None
    rather than dropped. Telling a reader which line is unverifiable is more
    useful than quietly omitting it, and it is the part a public deployment
    most needs to be honest about.
    """
    if not answer or not chunks:
        return []

    scored_chunks = [(c, _content_words(c.get("text") or c.get("title") or "")) for c in chunks]
    claims: list[dict] = []
    for sentence in _SENTENCE_SPLIT.split(answer.strip()):
        sentence = sentence.strip()
        if len(sentence) < 15:  # headings, "Yes.", list bullets
            continue
        words = _content_words(sentence)
        # Raw overlap counts grow with chunk length, so a big chunk wins on
        # size rather than on evidence: "What does Figure 1 show?" matched the
        # 12-word figure caption at 11 and the 536-word section CONTAINING it
        # at 12, and cited the section -- a page away from the figure.
        #
        # Length-normalising by sqrt (the same damping BM25 uses) ranks by how
        # concentrated the support is rather than how much text was searched:
        # caption 11/sqrt(12)=3.2, page 12/sqrt(69)=1.4, section 12/sqrt(536)
        # =0.5. sqrt rather than a plain ratio because a plain ratio hands the
        # win to any tiny fragment that happens to share its whole vocabulary.
        # The absolute floor below still applies, so normalisation only
        # decides between chunks that genuinely support the sentence.
        best, best_overlap, best_score = None, 0, 0.0
        for chunk, chunk_words in scored_chunks:
            overlap = len(words & chunk_words)
            if overlap == 0:
                continue
            score = overlap / math.sqrt(len(chunk_words) or 1)
            if score > best_score:
                best, best_overlap, best_score = chunk, overlap, score
        # Two matching content words is coincidence; a supported sentence
        # shares the terms it is reporting.
        supported = best is not None and best_overlap >= 3
        claims.append({
            "text": sentence,
            "source_id": best.get("id") if supported else None,
            "page": _chunk_page(best) if supported else None,
            "page_end": _chunk_page_end(best) if supported else None,
            "page_label": _chunk_page_label(best) if supported else None,
            "title": (best.get("title") or "")[:120] if supported else None,
            "overlap": best_overlap if supported else 0,
        })
    return claims


def _content_words(text: str) -> set[str]:
    return {
        w for w in re.findall(r"[a-z0-9]+", (text or "").lower())
        if len(w) > 3 and w not in _CITATION_STOPWORDS
    }


def _chunk_page(chunk: dict) -> object:
    """The page a chunk sits on, however the retriever happened to shape it.

    Retrieval paths do not agree on a name: the structural strategies emit
    `pdf_page`, the graph/vector path emits `page_start`, and some wrap the
    node under `raw`. Reading only `raw.page_start` -- which is what the
    claim builder did -- meant every citation from a structural strategy
    reported page None, so "What is Box 9 about?" cited no page at all.
    """
    return _chunk_field(chunk, ("pdf_page", "page_start", "page"))


def _chunk_page_label(chunk: dict) -> object:
    """The number PRINTED on the page, which is rarely the PDF index.

    A reader checking a citation looks for the number printed on the paper,
    while the viewer can only open the file by its index. The two differ by
    seven in the Go.Data report -- printed page 2 is PDF page 9 -- so citing
    one and navigating by the other lands the reader on the wrong page in
    both directions. Both travel with the claim: `page`/`page_end` drive
    navigation, `page_label` is what gets shown.
    """
    return _chunk_field(chunk, ("document_page", "page_label"))


def _chunk_page_end(chunk: dict) -> object:
    """The last page of a chunk that spans several, else its only page.

    20 of the 122 units in one 52-page report span more than one page, and a
    citation that names a single page for those sends the reader to where the
    content starts and silently drops the rest -- Box 9 runs across pages 30
    and 31, and only 30 was reported. Emitting the end alongside the start
    lets a citation read as a range, and collapses back to one page whenever
    start and end agree.
    """
    end = _chunk_field(chunk, ("page_end",))
    start = _chunk_page(chunk)
    return end if end is not None else start


def _chunk_field(chunk: dict, keys: tuple) -> object:
    for source in (chunk, chunk.get("raw") if isinstance(chunk.get("raw"), dict) else None):
        if not source:
            continue
        for key in keys:
            value = source.get(key)
            if value is not None:
                return value
    return None


def _verbatim_claims(chunks: list[dict]) -> list[dict]:
    """One citation per chunk, for answers assembled verbatim from chunks.

    The structural fast path (_STRUCTURAL_FAST_MODES: TOC, page, box,
    section and figure lookups) does not synthesise anything -- it
    concatenates chunk text as-is. So attribution needs no lexical matching
    and cannot be wrong: the chunk the text came from IS the source.

    Without this the fast path returned an answer with neither sources nor
    claims, which is why "What is Box 9 about?" and "What does Figure 1
    show?" came back with no page at all, while the TOC answer -- which
    formats a citation into its own text -- looked fine. Box 9 also spans
    two pages, so it yields two claims rather than one arbitrary page.
    """
    claims: list[dict] = []
    for chunk in chunks:
        text = (chunk.get("text") or "").strip()
        if not text:
            continue
        claims.append({
            "text": text[:400],
            "source_id": chunk.get("id"),
            "page": _chunk_page(chunk),
            "page_end": _chunk_page_end(chunk),
            "page_label": _chunk_page_label(chunk),
            "title": (chunk.get("title") or "")[:120],
            # Verbatim, so support is total rather than estimated -- the
            # lexical overlap score that _claim_citations reports has no
            # meaning here.
            "overlap": None,
        })
    return claims


def _grounded_sources(answer: str, chunks: list[dict]) -> list[dict]:
    """Keep the chunks the answer actually draws on.

    Every retrieved chunk was being cited, not just the ones used, so a reader
    was pointed at roughly four times the material that bore on the answer --
    measured at 0.26 mean page precision against 88% correct section naming.
    A citation nobody can practically check is close to no citation at all,
    which is the whole reason citation is tracked separately from answer text.

    Lexical overlap rather than a second LLM call: it costs nothing per query
    and is deterministic, so the same answer always cites the same sources.

    Fails OPEN. If nothing clears the bar the original list is returned
    unchanged -- dropping a source a reader wanted is worse than showing one
    they did not need, and the source viewer must never be left with nothing
    to open.
    """
    if not answer or len(chunks) <= 1:
        return chunks

    words = _content_words
    answer_words = words(answer)
    if not answer_words:
        return chunks

    scored = []
    for c in chunks:
        overlap = len(answer_words & words(c.get("text") or c.get("title") or ""))
        scored.append((overlap, c))

    best = max(o for o, _ in scored)
    if best == 0:
        return chunks
    # Half the best chunk's overlap: keeps genuine supporting passages while
    # dropping the long tail that merely came back from the same retrieval.
    keep = [c for o, c in scored if o >= max(1, best // 2)]
    return keep or chunks


def generate_node(state: ESGState):
    question = state["question"]
    retrieved = state.get("retrieved_context", {}) or {}
    chunks = retrieved.get("chunks", []) or []

    with pipeline_step(
        "document.graph.generate",
        mode=retrieved.get("mode"),
        chunks=len(chunks),
    ):
        return _generate_document_answer(
            question,
            retrieved,
            chunks,
            user_context=state.get("user_context"),
            skip_structured_guard=bool(state.get("skip_structured_guard")),
        )


def _generate_document_answer(
    question: str,
    retrieved: dict,
    chunks: list,
    *,
    user_context=None,
    skip_structured_guard: bool = False,
) -> dict:
    denied = next((c for c in chunks if c.get("id") == "access_denied"), None)
    if not chunks or denied:
        # Misroute guard: structured-graph question sent to document agent →
        # autofix or generic hint. Gated on retrieval having found NOTHING
        # real here (either no chunks at all, or the only "chunk" is the
        # synthetic access_denied marker) -- not on the question's wording
        # alone. Financial-document vocabulary ("sales", "revenue", "profit")
        # is also a Northwind-era structured-data cue, so a bare keyword
        # match used to redirect real 10-K questions away from chunks the
        # document agent had already found (see
        # [[repo_keyword_routing_scaling_risk]]). Only reaching for this
        # guard once retrieval itself came up empty (or denied) keeps it
        # doing its original job -- catching genuine misroutes -- without
        # discarding real content that happens to share vocabulary with the
        # other graph.
        #
        # The `denied` case matters just as much as the empty-chunks case:
        # a user who lacks document/"esg" access but DOES have structured
        # access, asking a structured-shaped question, used to get stuck on
        # the flat "access denied" message below without ever trying the
        # redirect -- because access_denied_response() returns a non-empty
        # chunks list (one marker chunk), so `if not chunks:` alone never
        # fired for this case. Skipped entirely when called as the
        # structured path's own low-confidence fallback
        # (routing.try_document_fallback) -- bouncing back to structured
        # there would just recreate the low-confidence answer we're trying
        # to improve on, an infinite ping-pong for questions phrased like
        # analytics but whose entity only exists in the ingested documents.
        if not skip_structured_guard:
            guard = document_agent_structured_guard(question, user_context)
            if guard is not None:
                return guard

        if denied:
            return {
                "answer": (denied.get("text") or "Access denied for document data.").strip(),
                "low_confidence": False,
            }

        answer = "I could not find relevant information in the ingested documents."
        low_confidence, confidence_note = compute_confidence(
            question, answer, chunks, "", provider=provider, model=CHAT_MODEL
        )
        return {
            "answer": answer,
            "low_confidence": low_confidence,
            "confidence_note": confidence_note,
        }

    mode = (retrieved.get("mode") or "").strip()
    if mode in _STRUCTURAL_FAST_MODES:
        answer = _build_fast_unstructured_answer(chunks)
        if answer:
            # Carry sources and claims explicitly: this path returns before
            # the synthesis path below that would otherwise attach them, and
            # a fast answer with no citation is exactly the case a reader
            # cannot check.
            return {
                "answer": answer,
                "low_confidence": False,
                "sources": [c for c in chunks if (c.get("text") or "").strip()],
                "claims": _verbatim_claims(chunks),
            }

    # Budget the total prompt context in characters, not just per-chunk --
    # a single chunk (e.g. a whole Chapter node's .text) can itself be huge
    # once chapter detection is accurate, so capping only the OUTPUT side
    # (DOCUMENT_SYNTHESIS_MAX_TOKENS) never bounded the INPUT side at all.
    # Truncates/drops the tail of context rather than the LLM call failing
    # outright on a context-window or rate-limit error.
    context_lines: list[str] = []
    used_chars = 0
    for i, c in enumerate(chunks, 1):
        title = c.get("title", "Result")
        text = (c.get("text") or "").strip()
        if not text:
            continue
        remaining = DOCUMENT_SYNTHESIS_CONTEXT_MAX_CHARS - used_chars
        if remaining <= 0:
            break
        if len(text) > remaining:
            text = text[:remaining].rstrip() + "\n[... truncated, content continues beyond this excerpt ...]"
        rel = c.get("related") or []
        rel_note = f" (graph: {', '.join(rel)})" if rel else ""
        line = f"[Chunk {i}] {title}{rel_note}\n{text}"
        context_lines.append(line)
        used_chars += len(line)
    context_text = "\n\n".join(context_lines)

    if is_toc_question(question):
        prompt_name = "document_toc"
    elif is_visual_page_question(question):
        prompt_name = "document_visual"
    elif is_page_question(question):
        prompt_name = "document_page"
    elif is_synthesis_question(question):
        prompt_name = "document_synthesis"
    else:
        prompt_name = "document_default"
    system_prompt = load_prompt(
        prompt_name,
        context=context_text,
        question=question,
        structured_entities=structured_entity_summary(),
    )
    response = provider.chat_completion(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": str(question)},
        ],
        temperature=0.1,
        max_tokens=(
            DOCUMENT_SYNTHESIS_LONG_MAX_TOKENS
            if (
                is_toc_question(question)
                or is_page_question(question)
                or is_visual_page_question(question)
            )
            else DOCUMENT_SYNTHESIS_MAX_TOKENS
        ),
    )
    answer = _fix_misrouted_structured_answer(
        response.choices[0].message.content.strip(), question
    )
    low_confidence, confidence_note = compute_confidence(
        question, answer, chunks, mode, provider=provider, model=CHAT_MODEL
    )
    return {
        "answer": answer,
        "low_confidence": low_confidence,
        "confidence_note": confidence_note,
        "sources": _grounded_sources(answer, chunks),
        "claims": _claim_citations(answer, chunks),
    }


def should_continue(state: ESGState):
    return "generate"


workflow = StateGraph(ESGState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_conditional_edges("retrieve", should_continue, {"generate": "generate"})
workflow.add_edge("generate", END)

esg_agent = workflow.compile()

