"""
Guards for unknown_property_issue.

Both "rejects" cases below are real queries this system generated and
answered confidently and wrongly, which is what makes this validator worth
having: Neo4j returns null for a missing property rather than erroring, and
an aggregate over nulls still returns one row, so nothing downstream can
tell a wrong answer from a right one.
"""
from src.retrieval.structured.cypher.validator import unknown_property_issue

SCHEMA = {
    "OrderItem": {"price", "freight", "item_key", "line_no"},
    "Order": {"order_key", "purchased_at"},
    "CONTAINS": set(),
    "User": {"user_id", "department", "created_at"},
}


def test_rejects_node_property_read_off_the_relationship():
    # `li` is the CONTAINS relationship; `freight` is on the OrderItem node.
    # This returned "zero" against a true total of 2,251,910.
    issue = unknown_property_issue(
        "MATCH (:Order)-[li:CONTAINS]->(:OrderItem) "
        "RETURN sum(toFloat(li.freight)) AS totalFreightValue",
        SCHEMA,
    )
    assert issue and "li.freight" in issue


def test_rejects_property_no_type_has():
    # Left to the empty-result retry, this became avg(created_at) reported as
    # a salary of 1,786,771,191,163.
    issue = unknown_property_issue(
        "MATCH (u:User) WHERE u.salary IS NOT NULL RETURN avg(u.salary) AS averageSalary",
        SCHEMA,
    )
    assert issue and "u.salary" in issue


def test_accepts_correct_binding():
    assert unknown_property_issue(
        "MATCH (:Order)-[:CONTAINS]->(i:OrderItem) RETURN sum(i.freight) AS total",
        SCHEMA,
    ) is None


def test_names_the_available_properties():
    """The message has to be actionable -- the model is being asked to fix the
    query from it, and 'wrong property' alone does not say what to use."""
    issue = unknown_property_issue("MATCH (i:OrderItem) RETURN i.shipping", SCHEMA)
    assert "freight" in issue and "price" in issue


def test_ignores_variables_of_unknown_type():
    """A WITH alias is not bound to a label in the text, so its properties
    cannot be checked. Passing over it keeps the validator from rejecting
    valid queries it simply cannot see the type of."""
    assert unknown_property_issue(
        "MATCH (i:OrderItem) WITH i AS row RETURN row.freight, row.anything",
        SCHEMA,
    ) is None


def test_ignores_parameters_and_function_chains():
    """`$param.x` and `datetime().year` both look like var.prop to a regex."""
    assert unknown_property_issue(
        "MATCH (o:Order) WHERE o.purchased_at.year = date().year "
        "AND o.order_key = $filter.key RETURN count(o)",
        SCHEMA,
    ) is None


def test_multi_label_and_multi_reltype_bindings_union_their_properties():
    assert unknown_property_issue(
        "MATCH (n:Order:OrderItem) RETURN n.order_key, n.freight", SCHEMA
    ) is None


def test_empty_schema_disables_the_check():
    """No introspection means no basis to reject; staying quiet beats guessing."""
    assert unknown_property_issue("MATCH (i:OrderItem) RETURN i.whatever", {}) is None
