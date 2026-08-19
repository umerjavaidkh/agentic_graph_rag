"""Per-request state shared between the router and the handlers.

Deliberately tiny and dependency-free. This lived in deps.py briefly, but
deps builds the ingestion manager and the thread pools, so importing it from
the router dragged the whole ingestion stack into a module that only needed
one ContextVar.
"""
import contextvars

# Set by `ask` when a retrieval mode was forced, read by the structured
# handler to decide whether a low-confidence answer may fall back to the other
# axis. Context-local, so concurrent requests do not see each other's setting.
_MODE_LOCKED: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "retrieval_mode_locked", default=False
)
