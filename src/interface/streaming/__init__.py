"""Streaming response orchestration.

Deliberately does not re-export: importing any module here used to drag the
whole orchestrator in as a side effect, which closed an import cycle once
document streaming moved out. Import the module you want directly.
"""
