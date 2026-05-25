"""
bridge.py — Unified bridge for routing queries.

This module exposes a single `ask()` entry point used by the FastAPI
server. It delegates to the internal `unstructured` and `structured`
implementations but lives at the package root for a clearer API.
"""
from .unstructured.router import ask, MCP_TOOLS

__all__ = ["ask", "MCP_TOOLS"]
