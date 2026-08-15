"""
Clarification must come from the connected graph, not a fixed metric list.

The version this replaced asked "order total, freight, or unit price?" and
defined them with unitPrice x quantity x (1 - discount). Against a graph
without those fields every option was uncomputable, so "what is the average
price of an order item?" -- a question with one obvious answer -- could not
be answered at all. These cases pin the two properties that matters: ask
only when the graph really is ambiguous, and offer only real fields.
"""
from src.retrieval.structured.policies.clarification import (
    needs_clarification,
    numeric_metric_candidates,
)

# Two properties a "price" question could plausibly mean, and several it
# could not -- the shape that should trigger a question.
AMBIGUOUS = {
    ("Order", "total_price"), ("OrderItem", "unit_price"),
    ("OrderItem", "freight"), ("Review", "score"),
}
# One price column: not ambiguous, whatever the phrasing.
UNAMBIGUOUS = {("OrderItem", "price"), ("Payment", "value"), ("Review", "score")}


def test_asks_when_two_properties_match():
    result = needs_clarification("What is the average price?", AMBIGUOUS)
    assert result is not None
    ids = {o["id"] for o in result["clarification_options"]}
    assert ids == {"Order.total_price", "OrderItem.unit_price"}


def test_offers_only_real_fields():
    """Every option must name a property that exists, so each one is answerable."""
    result = needs_clarification("average price", AMBIGUOUS)
    for option in result["clarification_options"]:
        label, prop = option["id"].split(".")
        assert (label, prop) in AMBIGUOUS


def test_single_match_is_answered_not_questioned():
    """The regression that motivated all of this."""
    assert needs_clarification(
        "What is the average price of an order item?", UNAMBIGUOUS
    ) is None


def test_property_named_value_is_matchable():
    """`value` reads like filler but is a real column name."""
    assert numeric_metric_candidates("average payment value", UNAMBIGUOUS) == [
        ("Payment", "value")
    ]


def test_no_aggregate_word_means_no_clarification():
    assert needs_clarification("show me the price list", AMBIGUOUS) is None


def test_no_schema_means_no_clarification():
    """With nothing introspected there is no basis to offer options."""
    assert needs_clarification("What is the average price?", set()) is None


def test_matches_camel_and_snake_case_alike():
    schema = {("Order", "totalPrice"), ("Item", "unit_price")}
    assert len(numeric_metric_candidates("average price", schema)) == 2


def test_unmatched_metric_word_offers_nothing():
    """A question naming no property should go to the generator, not a menu."""
    assert numeric_metric_candidates("average height", UNAMBIGUOUS) == []


def test_label_match_breaks_a_false_tie():
    """Two candidates are not ambiguity when one explains more of the question.

    "Total freight value across all order items" matches OrderItem.freight
    and Payment.value on property words alone. Only the first is also on the
    label the question names, so it wins outright -- asking here regressed a
    question that had been answering correctly.
    """
    schema = {("OrderItem", "freight"), ("Payment", "value"), ("OrderItem", "price")}
    assert needs_clarification(
        "What is the total freight value across all order items?", schema
    ) is None


def test_genuine_tie_still_asks():
    """Equal specificity on both sides is real ambiguity."""
    schema = {("Order", "total_price"), ("Invoice", "total_price")}
    result = needs_clarification("What is the average total price?", schema)
    assert result is not None
    assert len(result["clarification_options"]) == 2
