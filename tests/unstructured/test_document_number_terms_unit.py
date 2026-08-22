"""A document's number is the most distinctive thing a query can name.

"IRS Publication 559" yielded ['irs', 'publication'] -- terms every IRS
publication in the corpus shares -- because the token scan required a
leading letter and dropped the number. The query was maximally ambiguous by
construction, and the resolver could only decline.

Even kept, the number would not have matched: `\\b559\\b` finds nothing in
"doc_irs_p559", since there is no word boundary between "p" and "559".
"""
import pytest

from src.unstructured.retrieval.services.document_resolver import (
    _DOC_NUMBER_RE,
    DocumentResolver,
)


@pytest.fixture
def resolver():
    return DocumentResolver(graph_seeds=None)


@pytest.mark.parametrize("query, number", [
    ("what does IRS Publication 559 say", "559"),
    ("IRS Pub. 502 medical expenses", "502"),
    ("NIST IR 8286 risk register", "8286"),
    ("see Form 1040 instructions", "1040"),
    ("NIST SP 800-53 controls", "800-53"),
    ("report no. 8425", "8425"),
])
def test_a_document_number_is_kept(resolver, query, number):
    assert number in resolver.doc_name_terms(query)


@pytest.mark.parametrize("query", [
    "the invoice was $559 in 2025",
    "revenue grew 12 percent",
    "as of December 31, 2025",
    "we had 400 employees",
])
def test_a_bare_number_is_not_taken_for_an_identifier(resolver, query):
    """Anchored on a document-type word, so an amount, a count or a year in
    the question is never mistaken for a publication number."""
    terms = resolver.doc_name_terms(query)
    assert not any(t.isdigit() for t in terms), f"{query!r} -> {terms}"


def test_the_pattern_requires_a_document_word():
    assert _DOC_NUMBER_RE.search("Publication 559")
    assert not _DOC_NUMBER_RE.search("559 dollars")


def test_ambiguity_lead_is_explicit():
    """The threshold that decides guess-versus-ask. It was a bare 1.5 inline;
    naming it is what lets the picker use the same rule as the resolver."""
    assert DocumentResolver.AMBIGUITY_LEAD > 1.0


def test_candidates_are_capped():
    """A picker long enough to scroll is a worse answer than the guess it
    replaces."""
    import inspect

    sig = inspect.signature(DocumentResolver.candidates_for_query)
    assert sig.parameters["limit"].default == 10
