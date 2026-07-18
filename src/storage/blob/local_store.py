"""storage/blob/local_store.py — zero-dependency dev/test BlobStore backend."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from .base import BlobStore


class LocalFsBlobStore(BlobStore):
    """Writes each key to a file under root_dir. No Docker/network dependency."""

    def __init__(self, root_dir: str | Path):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, key: str) -> Path:
        # Keys are doc/revision/node-scoped path-like strings (e.g. "doc/rev/node/text").
        safe_key = key.strip("/")
        return self.root_dir / safe_key

    def put(self, key: str, content: str, *, content_type: str = "text/plain") -> str:
        path = self._path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return key

    def get(self, key: str) -> Optional[str]:
        path = self._path_for(key)
        if not path.exists():
            return None
        return path.read_text(encoding="utf-8")

    def delete(self, key: str) -> None:
        path = self._path_for(key)
        if path.exists():
            path.unlink()

    def exists(self, key: str) -> bool:
        return self._path_for(key).exists()

    def delete_prefix(self, prefix: str) -> int:
        base = self._path_for(prefix)
        if not base.exists():
            return 0
        count = 0
        if base.is_file():
            base.unlink()
            return 1
        for path in list(base.rglob("*")):
            if path.is_file():
                path.unlink()
                count += 1
        return count
