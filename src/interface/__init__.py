"""The entry layer: HTTP API, static UI, query routing, streaming.

Everything here depends on `structured/` and `unstructured/` -- never the
other way round. A core package importing from this one would mean the
retrieval side needed the web app to function.
"""
