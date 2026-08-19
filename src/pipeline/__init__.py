"""Job orchestration shared by both ingestion paths.

The RQ queue, its job store, and the task entry point. Kept apart from
either axis because a document ingest and a tabular load are dispatched the
same way, and deliberately free of FastAPI imports so an RQ worker can load
tasks without pulling in the web app.
"""
