"""Pages are embedded, because sections do not cover the whole document.

A Section's embedding sees `text[:2000]`. Measured across 50 documents,
sections held 4,449,412 of 8,526,653 characters -- so roughly half the
corpus was in no embedded node, and a question whose answer sat past a
section's cut could not be answered from a document that contained it.

Pages are the only node type that tiles a document completely with no
truncation: `search_text = text` verbatim, one chunk per page.
"""
from src.unstructured.models import NodeType
from src.unstructured.semantic.axis2 import (
    SEMANTIC_NODE_TYPES,
    SIMILARITY_EDGE_TYPES,
)


def test_pages_are_embedded():
    """The change itself. Without this, page content past a section's
    truncation is unreachable by meaning."""
    assert NodeType.PAGE in SEMANTIC_NODE_TYPES


def test_sections_and_chapters_are_still_embedded():
    """Pages are added, not substituted: a section embedding still carries
    the summary-level match that a single page cannot."""
    assert NodeType.CHAPTER in SEMANTIC_NODE_TYPES
    assert NodeType.SECTION in SEMANTIC_NODE_TYPES


def test_pages_do_not_get_similarity_edges():
    """Embedding for retrieval and linking for structure are separate
    concerns. Page-to-page similarity edges add nothing a reader needs and
    would bury the chapter/section structure the graph view exists to show
    -- dozens of pages per document against a handful of sections."""
    assert NodeType.PAGE not in SIMILARITY_EDGE_TYPES


def test_similarity_edges_stay_structural():
    assert SIMILARITY_EDGE_TYPES == {NodeType.CHAPTER, NodeType.SECTION}


def test_embedding_targets_are_a_superset_of_edge_targets():
    """Anything linked by similarity must have a vector to be similar with.
    The reverse need not hold, and deliberately does not."""
    assert SIMILARITY_EDGE_TYPES <= SEMANTIC_NODE_TYPES
