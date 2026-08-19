"""Infrastructure both retrieval axes depend on.

Nothing here knows whether a question is being answered from the structured
business graph or from ingested documents: configuration, auth, blob and
vector storage, model providers, telemetry, audit, feedback and the Neo4j
driver are all equally reachable from either side. Keeping them in one place
is what lets `structured/` and `unstructured/` sit beside each other without
either owning shared plumbing.
"""
