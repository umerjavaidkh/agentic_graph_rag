"""Neo4j connection, tenancy filtering and revision helpers.

Both axes write to and read from the same database -- the structured graph
and the document graph are separated by label, not by instance -- so the
driver and its tenancy/lifecycle predicates belong to neither side.
"""
