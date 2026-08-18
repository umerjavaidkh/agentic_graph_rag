"""Every answer path must cite a page, and cite the right one.

Measured on the Go.Data report before these fixes: of nine questions, two
carried a page citation. Four cited a section in PROSE ("as highlighted in
section 2.2") -- text the model wrote, which nothing verified and which a
reader cannot tell apart from a fabricated reference.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


def _stub_module(name: str) -> types.ModuleType:
    mod = types.ModuleType(name)
    sys.modules[name] = mod
    return mod


# graph.py does `from langgraph.graph import END, StateGraph` and builds a
# real StateGraph/compile() at module level -- stub both, permissively, so
# importing the real graph.py doesn't need the actual langgraph package
# (same pattern as test_structured_graph_confidence_unit.py).
for _n in ["langgraph", "langgraph.graph"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["langgraph.graph"].StateGraph = MagicMock()
sys.modules["langgraph.graph"].END = MagicMock()

if "neo4j" not in sys.modules:
    _stub_module("neo4j")
sys.modules["neo4j"].GraphDatabase = MagicMock()
sys.modules["neo4j"].Driver = MagicMock

for _n in ["src.auth", "src.auth.rbac_setup", "src.auth.roles"]:
    if _n not in sys.modules:
        _stub_module(_n)
sys.modules["src.auth.rbac_setup"].GraphRBAC = MagicMock()
sys.modules["src.auth.rbac_setup"].initialize_rbac_schema = MagicMock()
sys.modules["src.auth.roles"].UserContext = MagicMock
sys.modules["src.auth.roles"].DEFAULT_PUBLIC_CONTEXT = MagicMock(role=MagicMock(value="public"))

# graph.py does `retriever = DocumentRAGRetriever()` at module level, which
# would otherwise need a live Neo4j driver/RBAC/schema chain to construct.
# Not exercised by _generate_document_answer (only retrieve_node uses
# `retriever`) -- stub the class to a no-op constructor.
for _mod_name in ("src.retrieval.unstructured.graph", "src.retrieval.unstructured.retriever"):
    if _mod_name in sys.modules:
        del sys.modules[_mod_name]
_retriever_stub = _stub_module("src.retrieval.unstructured.retriever")
_retriever_stub.DocumentRAGRetriever = lambda *a, **k: MagicMock()
_retriever_stub.is_page_question = lambda q: False
_retriever_stub.is_synthesis_question = lambda q: False
_retriever_stub.is_toc_question = lambda q: False
_retriever_stub.is_visual_page_question = lambda q: False

import src.retrieval.unstructured.graph as graph_module
_chunk_page = graph_module._chunk_page
_chunk_page_end = graph_module._chunk_page_end
_claim_citations = graph_module._claim_citations
_verbatim_claims = graph_module._verbatim_claims


def test_page_is_read_whatever_the_retriever_called_it():
    """Structural strategies emit pdf_page, the graph path page_start, and
    some wrap the node under raw. Reading only raw.page_start meant every
    structural citation reported page None."""
    assert _chunk_page({"pdf_page": 30}) == 30
    assert _chunk_page({"page_start": 9}) == 9
    assert _chunk_page({"raw": {"page_start": 12}}) == 12
    assert _chunk_page({"title": "no page here"}) is None


def test_page_end_falls_back_to_the_start_for_single_page_chunks():
    assert _chunk_page_end({"page_start": 30, "page_end": 31}) == 31
    assert _chunk_page_end({"pdf_page": 30}) == 30
    assert _chunk_page_end({}) is None


def test_verbatim_claims_cite_the_chunk_the_text_came_from():
    """The structural fast path concatenates chunk text as-is, so attribution
    cannot be wrong -- but it used to return no sources and no claims at all,
    which is why "What is Box 9 about?" cited no page."""
    claims = _verbatim_claims([
        {"id": "n1", "title": "Box 9", "text": "Dashboard templates", "pdf_page": 30},
        {"id": "n2", "title": "empty", "text": "   "},
    ])
    assert len(claims) == 1
    assert claims[0]["source_id"] == "n1"
    assert claims[0]["page"] == 30


def test_denser_evidence_beats_a_longer_chunk_that_merely_contains_it():
    """A 536-word section CONTAINING the figure caption matched the sentence
    at overlap 12 and the 12-word caption at 11, so the section won on size
    and the citation pointed a page away from the figure."""
    sentence = (
        "Figure 1 shows screenshots of example outputs including summary "
        "charts, a contact dashboard, and a transmission chain."
    )
    caption = {
        "id": "caption",
        "pdf_page": 10,
        "text": "Screenshots of example outputs: summary charts, contact dashboard, transmission chain",
    }
    section = {
        "id": "section",
        "pdf_page": 9,
        # Same matching vocabulary, buried in far more unrelated text.
        "text": caption["text"] + " " + " ".join(f"unrelated{i}" for i in range(500)),
    }
    claims = _claim_citations(sentence, [section, caption])
    assert claims[0]["source_id"] == "caption"
    assert claims[0]["page"] == 10


def test_a_sentence_with_no_support_is_reported_not_dropped():
    """Telling a reader which line is unverifiable is the part a public
    deployment most needs to be honest about."""
    claims = _claim_citations(
        "Zebras migrate across the Serengeti every season without fail.",
        [{"id": "n1", "pdf_page": 3, "text": "Quarterly revenue rose in the fourth quarter."}],
    )
    assert len(claims) == 1
    assert claims[0]["source_id"] is None
    assert claims[0]["page"] is None


def test_query_response_declares_claims():
    """response_model strips any field the model does not name: the router
    set "claims" all along and /query dropped it on every request, so the UI
    could only ever show a document-level citation."""
    from pathlib import Path

    source = (Path(__file__).parent.parent / "src" / "api.py").read_text()
    model = source.split("class QueryResponse", 1)[1].split("\n@app.", 1)[0]
    assert "claims:" in model, "QueryResponse does not declare claims"
    assert "claims       = result.get(\"claims\"" in source, "claims never passed through"
