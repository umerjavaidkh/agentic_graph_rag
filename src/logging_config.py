"""Central logging configuration for the API and RAG pipeline."""
from __future__ import annotations

import logging
import os
import sys


def setup_logging() -> None:
    """Configure root logging once at process startup."""
    level_name = (os.environ.get("LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if root.handlers:
        root.setLevel(level)
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter(
            fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    root.addHandler(handler)
    root.setLevel(level)

    # Quiet noisy third-party loggers unless DEBUG.
    if level > logging.DEBUG:
        for name in ("neo4j", "httpx", "httpcore", "openai"):
            logging.getLogger(name).setLevel(logging.WARNING)
