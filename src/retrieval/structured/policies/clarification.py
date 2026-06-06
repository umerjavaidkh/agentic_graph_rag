"""Structured query clarification when metrics are ambiguous."""
from __future__ import annotations

from typing import Any, Optional

from ....conversation.clarification import format_clarification_answer


def needs_clarification(query: str) -> Optional[dict[str, Any]]:
    """
    Ask a follow-up when the metric is ambiguous.

    Example: "avg order price" could mean freight, computed order total, or unit price.
    """
    ql = (query or "").strip().lower()
    if not ql:
        return None

    if "order" in ql and "price" in ql and ("avg" in ql or "average" in ql):
        if any(k in ql for k in (
            "freight", "shipping", "ship cost", "order total", "total",
            "unitprice", "unit price", "line item",
        )):
            return None
        options = [
            {
                "id": "order_total",
                "label": "Order total (recommended)",
                "detail": "Sum of line items per order: unitPrice × quantity × (1 - discount)",
                "aliases": ["total", "order total", "line item total", "sum items", "items total"],
            },
            {
                "id": "freight",
                "label": "Freight / shipping cost",
                "detail": "Use Order.freight (shipping cost) per order",
                "aliases": ["freight", "shipping", "shipping cost", "ship cost"],
            },
            {
                "id": "unit_price",
                "label": "Average unit price",
                "detail": "Average of line-item unitPrice (not the order total)",
                "aliases": ["unit price", "unitprice", "item price"],
            },
        ]
        prompt = (
            "When you say **average order price**, which metric do you mean?\n\n"
            "Reply with 1, 2, or 3 (or the option name)."
        )
        answer = format_clarification_answer(prompt, options)
        return {
            "query": query,
            "strategy": "clarification",
            "mode": "needs_clarification",
            "original_question": query,
            "clarification_kind": "structured_order_price",
            "clarification_options": options,
            "chunks": [
                {
                    "id": "clarification",
                    "title": "Clarification",
                    "text": answer,
                    "score": 1.0,
                    "related": [],
                }
            ],
            "total_available": 1,
        }

    return None
