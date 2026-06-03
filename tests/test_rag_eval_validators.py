"""Unit tests for eval/validators.py (no live API)."""

from eval.validators import validate_response


def test_document_route_and_keywords():
    case = {
        "expect": {
            "route_tool": "search_documents",
            "agent": "unstructured",
            "min_answer_chars": 5,
            "any_of_keywords": [["GOARN"], ["goarn"]],
        }
    }
    ok = validate_response(
        case,
        {
            "answer": "GOARN collaborated with WHO.",
            "route_tool": "search_documents",
            "agent": "unstructured",
            "sources": [{}],
            "total_chunks": 1,
        },
    )
    assert ok.passed


def test_structured_chart_expectation():
    case = {
        "expect": {
            "route_tool": "query_data",
            "agent": "structured",
            "expect_chart_types": ["line"],
            "min_table_rows": 2,
        }
    }
    ok = validate_response(
        case,
        {
            "answer": "Monthly orders in 1997.",
            "route_tool": "query_data",
            "agent": "structured",
            "presentation": {
                "kind": "mixed",
                "blocks": [
                    {"type": "chart", "chartType": "line"},
                    {"type": "table", "rows": [{}, {}, {}]},
                ],
            },
        },
    )
    assert ok.passed


def test_min_keyword_groups_in_sources():
    case = {
        "expect": {
            "any_of_keywords": [["English"], ["Spanish"], ["Mongolian"]],
            "min_keyword_groups": 2,
            "keywords_in_sources": True,
        }
    }
    ok = validate_response(
        case,
        {
            "answer": "The course is multilingual.",
            "sources": [{"text": "Available in English and Spanish on OpenWHO."}],
        },
    )
    assert ok.passed


def test_forbid_chart():
    case = {"expect": {"forbid_chart": True}}
    bad = validate_response(
        case,
        {"answer": "x", "presentation": {"blocks": [{"type": "chart", "chartType": "bar"}]}},
    )
    assert not bad.passed
