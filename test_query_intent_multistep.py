"""Heuristic gate for structured multistep planning."""
from src.retrieval.structured.query_intent import likely_needs_multistep_plan


def test_simple_queries_skip_planner():
    assert not likely_needs_multistep_plan("How many orders were placed in 1997?")
    assert not likely_needs_multistep_plan("Top 10 customers by revenue")
    assert not likely_needs_multistep_plan("Average order price")


def test_nested_queries_run_planner():
    assert likely_needs_multistep_plan(
        "Top 3 customers per country, then top 5 products for each of those customers"
    )
    assert likely_needs_multistep_plan("Top 3 products per category")
    assert likely_needs_multistep_plan(
        "Among the top 10 customers by revenue, show the top 2 products each ordered"
    )
