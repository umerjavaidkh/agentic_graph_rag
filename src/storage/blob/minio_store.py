"""storage/blob/minio_store.py — production BlobStore backend (MinIO / S3-compatible)."""
from __future__ import annotations

import io
from typing import Optional

from .base import BlobStore


class MinioBlobStore(BlobStore):
    """Wraps the `minio` SDK. Requires the `minio` package (see requirements.txt)."""

    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        bucket: str,
        secure: bool = False,
    ):
        from minio import Minio  # optional dependency, imported lazily

        self._client = Minio(
            endpoint, access_key=access_key, secret_key=secret_key, secure=secure
        )
        self.bucket = bucket
        if not self._client.bucket_exists(self.bucket):
            self._client.make_bucket(self.bucket)

    def put(self, key: str, content: str, *, content_type: str = "text/plain") -> str:
        data = content.encode("utf-8")
        self._client.put_object(
            self.bucket, key, io.BytesIO(data), length=len(data), content_type=content_type
        )
        return key

    def get(self, key: str) -> Optional[str]:
        from minio.error import S3Error

        try:
            response = self._client.get_object(self.bucket, key)
            try:
                return response.read().decode("utf-8")
            finally:
                response.close()
                response.release_conn()
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return None
            raise

    def delete(self, key: str) -> None:
        self._client.remove_object(self.bucket, key)

    def exists(self, key: str) -> bool:
        from minio.error import S3Error

        try:
            self._client.stat_object(self.bucket, key)
            return True
        except S3Error as exc:
            if exc.code == "NoSuchKey":
                return False
            raise

    def delete_prefix(self, prefix: str) -> int:
        objects = list(self._client.list_objects(self.bucket, prefix=prefix, recursive=True))
        for obj in objects:
            self._client.remove_object(self.bucket, obj.object_name)
        return len(objects)
