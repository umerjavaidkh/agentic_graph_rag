"""Unit tests for structured presentation chart type selection."""
from src.presentation.structured_planner import choose_chart_type


def test_choose_bar_for_top_n():
    assert choose_chart_type(
        "Top 5 products by sales",
        "productName",
        ["A", "B", "C", "D", "E"],
        [10.0, 8.0, 6.0, 4.0, 2.0],
    ) == "bar"


def test_choose_doughnut_for_share():
    assert choose_chart_type(
        "Revenue share by category",
        "categoryName",
        ["A", "B", "C"],
        [40.0, 35.0, 25.0],
    ) == "doughnut"


def test_choose_line_for_monthly():
    assert choose_chart_type(
        "Monthly order volume in 1997",
        "month",
        ["1997-01", "1997-02", "1997-03"],
        [10.0, 12.0, 9.0],
    ) == "line"


def test_choose_horizontal_for_many_rows():
    labels = [f"Item-{i}" for i in range(12)]
    values = [float(i) for i in range(12, 0, -1)]
    assert choose_chart_type(
        "Top 12 customers by revenue",
        "companyName",
        labels,
        values,
    ) == "bar-horizontal"
