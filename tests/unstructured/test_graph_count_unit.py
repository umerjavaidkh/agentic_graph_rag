"""tests/unstructured/test_graph_count_unit.py — counting is a graph read.

No sentence in NIST SP 800-161 says how many tables it contains, so asked
to count from prose the model read a "List of Tables" fragment and
answered 23 against an actual 88, and "3 main chapters" against 15. Both
confident, because a plausible list was in front of it.
"""
from __future__ import annotations

from src.unstructured.retrieval.services.structural import StructuralService


class _Session:
    def __init__(self, total=0, outline_rows=()):
        self.total = total
        self.outline_rows = list(outline_rows)
        self.queries: list[str] = []

    def run(self, q, **kw):
        self.queries.append(" ".join(q.split()))
        if "count(" in q:
            return _Single({"total": self.total})
        return list(self.outline_rows)


class _Single:
    def __init__(self, row):
        self._row = row

    def single(self):
        return self._row


def test_counts_tables_from_the_graph():
    svc, s = StructuralService(), _Session(total=88)

    out = svc.count_units(s, "How many tables are in NIST SP 800-161?", ["d1"])

    assert out[0]["count"] == 88
    assert "88 tables" in out[0]["text"]
    assert "n.region_kind = $region_kind" in " ".join(s.queries)


def test_titled_units_are_counted_by_distinct_title():
    """Matches outline(): a heading split across two nodes is one chapter."""
    svc, s = StructuralService(), _Session(total=15)

    svc.count_units(s, "How many chapters are in NIST SP 800-161?", ["d1"])

    assert "count(DISTINCT toLower(trim(n.title)))" in " ".join(s.queries)


def test_singular_is_not_pluralised():
    svc, s = StructuralService(), _Session(total=1)

    assert "1 figure." in svc.count_units(s, "How many figures?", ["d1"])[0]["text"]


def test_a_question_that_is_not_a_count_is_left_alone():
    svc, s = StructuralService(), _Session(total=88)

    assert svc.count_units(s, "What do the tables show?", ["d1"]) == []


def test_no_opinion_without_a_document_to_count_within():
    """Counting across the whole corpus would answer a question nobody asked."""
    svc, s = StructuralService(), _Session(total=88)

    assert svc.count_units(s, "How many tables are there?", []) == []


def test_count_is_returned_verbatim_rather_than_reworded():
    from src.unstructured.retrieval.graph import _generate_document_answer

    out = _generate_document_answer(
        "How many tables?", {},
        [{"id": "graph_count", "title": "Number of tables",
          "text": "This document contains 88 tables."}],
    )

    assert out["answer"] == "This document contains 88 tables."
    assert out["low_confidence"] is False
