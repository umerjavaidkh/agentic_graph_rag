"""Everything that answers from the structured business graph.

Tabular sources -- CSV, Excel, SQLite -- loaded into Neo4j as a property
graph, and Text-to-Cypher retrieval over whatever schema happens to be
there. The counterpart of `unstructured/`, which answers from ingested
documents. Neither imports the other; both sit on `shared/`.
"""
