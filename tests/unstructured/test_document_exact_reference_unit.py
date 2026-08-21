"""Naming a document outright must outrank the thread's current one."""
from src.unstructured.retrieval.services.document_resolver import DocumentResolver as D


class _Rows(list):
    pass


class _Session:
    def __init__(self, docs):
        self._docs = docs

    def run(self, *_a, **_k):
        return _Rows({"logical_id": i, "title": t} for i, t in self._docs)


def _resolver():
    return D.__new__(D)


def test_punctuation_does_not_change_which_document_is_meant():
    """Filenames use underscores, display titles spaces, filing ids hyphens."""
    n = _resolver()._normalise_reference
    assert n("rag_document_2") == n("rag document 2") == n("RAG-Document-2")


def test_a_named_document_is_found():
    s = _Session([("doc_rag_document_2", "rag_document_2"), ("cvx-10k", "CVX_10-K_2026-02-24")])
    got = _resolver().exact_document_reference(s, "What is Box 9 about in rag_document_2?")
    assert got and got[0] == "doc_rag_document_2"


def test_a_one_word_title_cannot_match_any_sentence_using_that_word():
    """Otherwise a document called "Policy" wins every question mentioning policy."""
    s = _Session([("doc_policy", "Policy")])
    assert _resolver().exact_document_reference(s, "what does the policy say?") is None


def test_ambiguity_declines_rather_than_guesses():
    """The same filing ingested twice: neither copy is a safe answer."""
    s = _Session([("a:r1", "CVX_10-K_2026-02-24"), ("b:r1", "CVX_10-K_2026-02-24")])
    assert _resolver().exact_document_reference(s, "contents of CVX_10-K_2026-02-24?") is None


def test_a_question_naming_nothing_falls_through():
    """So the thread's document still answers "what's on page 6?"."""
    s = _Session([("doc_rag_document_2", "rag_document_2")])
    assert _resolver().exact_document_reference(s, "what is on page 6?") is None


def test_the_named_document_is_consulted_before_the_thread_hint():
    import inspect
    src = inspect.getsource(D.resolve_document_for_query)
    assert src.index("exact_document_reference") < src.index("if document_id_hint:")
