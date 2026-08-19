"""Document ingestion: parse, build both graph axes, load, validate.

The tabular loaders live under `structured/ingestion/`, and the job store
and queue that both share stay in the pipeline package -- this holds only
the document pipeline itself.
"""
