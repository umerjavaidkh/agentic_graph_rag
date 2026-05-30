"""Pluggable asset storage: local filesystem (demo) or MinIO (S3-compatible)."""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

from ..config.settings import (
    ASSETS_DIR,
    ASSETS_PUBLIC_PREFIX,
    MINIO_ACCESS_KEY,
    MINIO_BUCKET,
    MINIO_ENDPOINT,
    MINIO_SECRET_KEY,
    MINIO_SECURE,
)


class AssetStore(ABC):
    @abstractmethod
    def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
        """Store bytes; return the same key."""

    @abstractmethod
    def exists(self, key: str) -> bool:
        pass

    @abstractmethod
    def public_url(self, key: str) -> Optional[str]:
        """URL the browser can load, or None if not served."""

    def get_bytes(self, key: str) -> Optional[bytes]:
        return None

    def delete_prefix(self, prefix: str) -> int:
        """Remove all objects under a key prefix (e.g. book_abc/images/)."""
        raise NotImplementedError


class LocalAssetStore(AssetStore):
    def __init__(self, root: Path, public_prefix: str = "/assets"):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.public_prefix = public_prefix.rstrip("/")

    def _path(self, key: str) -> Path:
        safe = key.lstrip("/").replace("..", "")
        path = (self.root / safe).resolve()
        if not str(path).startswith(str(self.root)):
            raise ValueError("Invalid asset key")
        return path

    def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return key

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def public_url(self, key: str) -> Optional[str]:
        if not self.exists(key):
            return None
        return f"{self.public_prefix}/{key.lstrip('/')}"

    def get_bytes(self, key: str) -> Optional[bytes]:
        path = self._path(key)
        if path.is_file():
            return path.read_bytes()
        return None

    def delete_prefix(self, prefix: str) -> int:
        safe = prefix.lstrip("/").replace("..", "")
        if not safe:
            return self._delete_all_under_root()

        base = (self.root / safe).resolve()
        if not str(base).startswith(str(self.root)):
            raise ValueError("Invalid asset prefix")
        if not base.exists():
            return 0
        removed = 0
        if base.is_file():
            base.unlink()
            return 1
        for path in sorted(base.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
                removed += 1
            elif path.is_dir():
                try:
                    path.rmdir()
                except OSError:
                    pass
        try:
            base.rmdir()
        except OSError:
            pass
        return removed

    def _delete_all_under_root(self) -> int:
        """Wipe all book asset trees (admin DB reset)."""
        removed = 0
        if not self.root.exists():
            return 0
        for child in list(self.root.iterdir()):
            if child.name.startswith("."):
                continue
            if child.is_file():
                child.unlink()
                removed += 1
            elif child.is_dir():
                removed += self.delete_prefix(child.name + "/")
        return removed


class MinioAssetStore(AssetStore):
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
        public_prefix: str = "/assets",
    ):
        from minio import Minio

        self.client = Minio(
            endpoint,
            access_key=access_key,
            secret_key=secret_key,
            secure=secure,
        )
        self.bucket = bucket
        self.public_prefix = public_prefix.rstrip("/")
        if not self.client.bucket_exists(bucket):
            self.client.make_bucket(bucket)

    def put(self, key: str, data: bytes, content_type: str = "image/jpeg") -> str:
        self.client.put_object(
            self.bucket,
            key.lstrip("/"),
            io.BytesIO(data),
            length=len(data),
            content_type=content_type,
        )
        return key

    def exists(self, key: str) -> bool:
        from minio.error import S3Error

        try:
            self.client.stat_object(self.bucket, key.lstrip("/"))
            return True
        except S3Error:
            return False

    def public_url(self, key: str) -> Optional[str]:
        if not self.exists(key):
            return None
        return f"{self.public_prefix}/{key.lstrip('/')}"

    def get_bytes(self, key: str) -> Optional[bytes]:
        from minio.error import S3Error

        resp = None
        try:
            resp = self.client.get_object(self.bucket, key.lstrip("/"))
            return resp.read()
        except S3Error:
            return None
        finally:
            if resp is not None:
                try:
                    resp.close()
                    resp.release_conn()
                except Exception:
                    pass

    def delete_prefix(self, prefix: str) -> int:
        from minio.deleteobjects import DeleteObject

        safe = prefix.lstrip("/").replace("..", "")
        if not safe:
            removed = 0
            for obj in self.client.list_objects(self.bucket, recursive=True):
                self.client.remove_object(self.bucket, obj.object_name)
                removed += 1
            return removed

        to_delete = [
            DeleteObject(obj.object_name)
            for obj in self.client.list_objects(
                self.bucket, prefix=safe, recursive=True
            )
        ]
        if not to_delete:
            return 0
        errors = list(self.client.remove_objects(self.bucket, to_delete))
        if errors:
            raise RuntimeError(f"MinIO delete failed: {errors[:3]}")
        return len(to_delete)


def create_asset_store(backend: str) -> AssetStore:
    backend = (backend or "local").lower()
    if backend == "minio":
        return MinioAssetStore(
            MINIO_ENDPOINT,
            MINIO_ACCESS_KEY,
            MINIO_SECRET_KEY,
            MINIO_BUCKET,
            secure=MINIO_SECURE,
            public_prefix=ASSETS_PUBLIC_PREFIX,
        )
    return LocalAssetStore(Path(ASSETS_DIR), public_prefix=ASSETS_PUBLIC_PREFIX)
