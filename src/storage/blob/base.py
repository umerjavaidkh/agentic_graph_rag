"""
storage/blob/base.py — interface for raw text / visual_content storage.

Keeps large text blobs out of Neo4j node properties so the graph stays
structure + pointers. Keys are doc/revision/node-scoped (not content-hashed)
so an expired revision's blobs can be swept the same way its Neo4j nodes are
purged on supersede.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class BlobStore(ABC):
    @abstractmethod
    def put(self, key: str, content: str, *, content_type: str = "text/plain") -> str:
        """Write content under key, return the (possibly normalised) key."""

    @abstractmethod
    def get(self, key: str) -> Optional[str]:
        """Return content for key, or None if it doesn't exist."""

    @abstractmethod
    def put_bytes(self, key: str, content: bytes, *, content_type: str = "application/octet-stream") -> str:
        """Write binary content (e.g. an original source PDF) under key."""

    @abstractmethod
    def get_bytes(self, key: str) -> Optional[bytes]:
        """Return binary content for key, or None if it doesn't exist."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """Remove content for key. No-op if key doesn't exist."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Whether key currently has content stored."""

    @abstractmethod
    def delete_prefix(self, prefix: str) -> int:
        """Delete all keys under prefix (e.g. a superseded revision). Returns count deleted."""
