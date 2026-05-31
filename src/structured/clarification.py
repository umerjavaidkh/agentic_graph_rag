"""Detect vague structured-graph queries that need one clarifying slot.

Schema-agnostic detection; after the user picks an option, rewrites use explicit
entity/metric wording so text2cypher and demo templates route correctly.
"""
from __future__ import annotations

import re
from typing import Any, Optional

_SPECIFIC = re.compile(
    r"\b(top\s+\d+|how many|count of|total\s+\w+\s+for|by\s+\w+|per\s+\w+|"
    r"best product|highest|lowest|schema|what labels|what data)\b",
    re.I,
)

# option_id -> concrete follow-up question (entity explicit, not vague "sales")
_CLARIFICATION_REWRITES: dict[str, str] = {
    "by_product": "Show top 10 products by total revenue and units sold",
    "by_country": "Show total sales revenue grouped by ship country, top 10 countries",
    "by_customer": "Show top 10 customers by total order revenue",
    "by_revenue": "Show top 10 customers by total revenue",
    "by_orders": "Show top 10 customers by number of orders",
    "top_products": "What are the top 10 best-selling products by revenue?",
    "all_products": "List 15 products with name, category, and unit price",
    "by_category": "Show product count and total sales revenue by category",
    "recent_orders": "Show the 10 most recent orders with customer and total",
    "orders_by_customer": "Show top 10 customers by order count",
    "high_value": "Show top 10 orders by order value",
}

_OPTION_ALIASES: dict[str, list[str]] = {
    "by_product": ["product", "by product", "item", "items", "top sellers"],
    "by_country": ["country", "by country", "region", "geographic", "geography"],
    "by_customer": ["customer", "customers", "account", "by customer", "client"],
    "by_revenue": ["revenue", "spend", "total spend"],
    "by_orders": ["orders", "order count", "transactions"],
    "top_products": ["top sellers", "best sellers", "top products"],
    "all_products": ["catalog", "all products", "list products"],
    "by_category": ["category", "categories"],
    "recent_orders": ["recent", "latest orders"],
    "orders_by_customer": ["orders by customer"],
    "high_value": ["highest value", "high value", "largest orders"],
}


def needs_structured_clarification(question: str) -> Optional[dict[str, Any]]:
    """Return clarification spec or None if the query is specific enough."""
    q = question.strip()
    if not q or _SPECIFIC.search(q):
        return None

    ql = q.lower()

    if re.search(r"\b(?:show|tell me about|what are|what is)\s+(?:the\s+)?sales\b", ql):
        if not re.search(r"\b(by|per|for|product|customer|country|region|category)\b", ql):
            return _spec(
                "structured_metric",
                "Your question about **sales** could mean several things. What do you want?",
                [
                    ("by_product", "By product / item", "Top items by revenue or units sold"),
                    ("by_country", "By country / region", "Sales grouped geographically"),
                    ("by_customer", "By customer / account", "Who bought the most"),
                ],
            )

    if re.search(r"\b(?:best|top)\s+customers?\b", ql) and not re.search(
        r"\b(by|order|spend|revenue|purchase|count)\b", ql
    ):
        return _spec(
            "structured_metric",
            "How should **best customers** be ranked?",
            [
                ("by_revenue", "By total spend", "Highest revenue or order value"),
                ("by_orders", "By order count", "Most transactions or orders"),
            ],
        )

    if re.search(r"\b(?:show|list|get)\s+(?:me\s+)?(?:all\s+)?products?\b", ql) and not re.search(
        r"\b(top|best|category|supplier|price|where|from)\b", ql
    ):
        return _spec(
            "structured_entity",
            "Which **products / items** view do you want?",
            [
                ("top_products", "Top sellers", "Best-selling items in the graph"),
                ("all_products", "Catalog sample", "Sample rows from the product/item nodes"),
                ("by_category", "By category", "Items grouped by category or type"),
            ],
        )

    if re.search(r"\b(?:show|list)\s+(?:me\s+)?orders?\b", ql) and not re.search(
        r"\b(for|by|customer|product|country|date|recent|last)\b", ql
    ):
        return _spec(
            "structured_entity",
            "Which **orders / transactions** view do you want?",
            [
                ("recent_orders", "Most recent", "Latest orders or transactions"),
                ("orders_by_customer", "By customer", "Grouped or filtered by customer"),
                ("high_value", "Highest value", "Sorted by amount or total"),
            ],
        )

    return None


def apply_structured_clarification(question: str, choice: dict[str, Any]) -> str:
    """Rewrite a vague question after the user picks an option."""
    cid = (choice.get("id") or "").strip()
    if cid in _CLARIFICATION_REWRITES:
        return _CLARIFICATION_REWRITES[cid]
    label = (choice.get("label") or "").strip()
    if label:
        return f"{question.strip()} grouped {label.lower()}"
    return question.strip()


def _spec(kind: str, prompt: str, options: list[tuple[str, str, str]]) -> dict[str, Any]:
    return {
        "kind": kind,
        "prompt": prompt,
        "options": [
            {
                "id": oid,
                "label": label,
                "detail": detail,
                "aliases": list(
                    dict.fromkeys(
                        [label.lower(), *_OPTION_ALIASES.get(oid, [])]
                    )
                ),
            }
            for oid, label, detail in options
        ],
    }
