"""tests/test_thread_memory_document_continuity_unit.py — document_id
capture and carry-forward across conversation turns.

Covers the other half of conversation continuity (see
test_document_resolver_hint_unit.py for the resolver-side priority
logic): extract_critical_from_result must capture document_id/title from
a retrieval result, and resolve_follow_up must carry it forward into the
next turn's hint unless something more specific overrides it.

Run with:
    python -m pytest tests/test_thread_memory_document_continuity_unit.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path


from src.shared.conversation.thread_memory import extract_critical_from_result, resolve_follow_up


def _result(document_id="stratec-compliance-policy-2025", document_title="STRATEC Policy", **rc_extra):
    rc = {"mode": "structural_page", "document_id": document_id, "document_title": document_title, **rc_extra}
    return {
        "agent": "unstructured",
        "query_type": "structural_page",
        "sources": [],
        "retrieved_context": rc,
    }


def test_extract_captures_document_id_and_title():
    snapshot = extract_critical_from_result("What's on page 6?", _result())
    assert snapshot is not None
    assert snapshot["document_id"] == "stratec-compliance-policy-2025"
    assert snapshot["document_title"] == "STRATEC Policy"


def test_extract_handles_missing_document_id():
    result = _result(document_id=None, document_title=None)
    snapshot = extract_critical_from_result("What's on page 6?", result)
    assert snapshot is not None
    assert snapshot["document_id"] is None


def test_resolve_follow_up_carries_document_id_forward_by_default():
    prior = {"mode": "structural_page", "document_id": "stratec-compliance-policy-2025"}
    resolved = resolve_follow_up("What is discussed on page 6 of this document?", prior)
    assert resolved["document_id"] == "stratec-compliance-policy-2025"


def test_resolve_follow_up_no_prior_means_no_hint():
    resolved = resolve_follow_up("What is discussed on page 6 of this document?", None)
    assert resolved["document_id"] is None


def test_clarification_choice_overrides_carried_forward_hint():
    prior = {
        "document_id": "stratec-compliance-policy-2025",
        "pending_clarification": {
            "kind": "document_choice",
            "options": [{"id": "jpm-10k-2017-02-28", "label": "JPM 10-K"}],
            "original_question": "what does this filing discuss",
        },
    }
    resolved = resolve_follow_up("JPM 10-K", prior)
    assert resolved["document_id"] == "jpm-10k-2017-02-28"
    assert resolved["follow_up_kind"] == "clarification_document"


def test_subsection_detail_resolution_carries_document_id_forward():
    prior = {
        "document_id": "stratec-compliance-policy-2025",
        "parent_id": "section_4",
        "parent_title": "Compliance Management System",
        "children": [{"id": "section_4_1", "title": "Corruption Prevention"}],
    }
    resolved = resolve_follow_up("1", prior)
    assert resolved["follow_up_kind"] == "subsection_detail"
    assert resolved["document_id"] == "stratec-compliance-policy-2025"


def test_page_follow_up_carries_document_id_forward():
    prior = {"document_id": "stratec-compliance-policy-2025", "pdf_page": 6}
    resolved = resolve_follow_up("what's on that page", prior)
    assert resolved["follow_up_kind"] == "page"
    assert resolved["document_id"] == "stratec-compliance-policy-2025"
