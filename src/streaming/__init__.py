"""Streaming response orchestration.

Deliberately empty: importing any module here used to pull query_stream in
as a side effect, which closed an import cycle once document streaming moved
to unstructured/. Import the module you want directly.
"""
