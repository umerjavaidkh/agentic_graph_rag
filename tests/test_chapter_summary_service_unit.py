"""tests/test_chapter_summary_service_unit.py — ChapterSummaryService
(query-time fetch of Chapter.summary for a resolved document).

Run with:
    python -m pytest tests/test_chapter_summary_service_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest


from src.unstructured.retrieval.services.chapter_summary import ChapterSummaryService


class _FakeSession:
    def __init__(self, rows):
        self.rows = rows
        self.last_cypher = None
        self.last_kwargs = None

    def run(self, cypher, **kwargs):
        self.last_cypher = cypher
        self.last_kwargs = kwargs
        return self.rows


def test_returns_empty_without_document_id():
    service = ChapterSummaryService()
    assert service.fetch_for_document(_FakeSession([]), "", tenant_id="default") == []


def test_maps_rows_to_candidate_chunks():
    rows = [
        {"id": "chapter_1", "title": "Item 1. Business", "summary": "Overview of the business.", "order": 1},
        {"id": "chapter_2", "title": "Item 1A. Risk Factors", "summary": "Discusses credit and market risk.", "order": 2},
    ]
    session = _FakeSession(rows)
    items = ChapterSummaryService().fetch_for_document(session, "aapl-10k-2024", tenant_id="default")

    assert len(items) == 2
    assert items[0]["id"] == "chapter_1"
    assert items[0]["text"] == "Overview of the business."
    assert items[0]["title"] == "Item 1. Business"
    assert items[0]["related"] == ["via:chapter_summary"]
    assert session.last_kwargs["doc_id"] == "aapl-10k-2024"
    assert session.last_kwargs["tenant_id"] == "default"


def test_skips_rows_missing_id_or_summary():
    rows = [
        {"id": "", "title": "x", "summary": "y", "order": 1},
        {"id": "chapter_1", "title": "x", "summary": "", "order": 1},
        {"id": "chapter_2", "title": "x", "summary": "real summary", "order": 2},
    ]
    session = _FakeSession(rows)
    items = ChapterSummaryService().fetch_for_document(session, "doc-1")
    assert [i["id"] for i in items] == ["chapter_2"]


def test_falls_back_to_id_when_title_missing():
    rows = [{"id": "chapter_1", "title": "", "summary": "s", "order": 1}]
    session = _FakeSession(rows)
    items = ChapterSummaryService().fetch_for_document(session, "doc-1")
    assert items[0]["title"] == "chapter_1"
