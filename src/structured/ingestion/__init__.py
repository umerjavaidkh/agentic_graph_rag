"""Loading tabular sources into the business graph.

Schema-agnostic on purpose: the loader infers labels and relationships from
the data rather than from a hardcoded model, and schema_docs records what
each field MEANS so the Cypher generator can pick the right one.
"""
