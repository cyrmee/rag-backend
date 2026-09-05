import asyncio
import io

from minio import Minio

from app.config import settings

_client = Minio(
    settings.minio_endpoint,
    access_key=settings.minio_access_key,
    secret_key=settings.minio_secret_key,
    secure=settings.minio_secure,
)


def _ensure_bucket_sync() -> None:
    if not _client.bucket_exists(settings.minio_bucket):
        _client.make_bucket(settings.minio_bucket)


def _upload_sync(object_key: str, image_bytes: bytes, content_type: str) -> None:
    _ensure_bucket_sync()
    _client.put_object(
        settings.minio_bucket,
        object_key,
        io.BytesIO(image_bytes),
        length=len(image_bytes),
        content_type=content_type,
    )


def _download_sync(object_key: str) -> bytes:
    response = _client.get_object(settings.minio_bucket, object_key)
    try:
        return response.read()
    finally:
        response.close()
        response.release_conn()


def _list_keys_sync() -> list[str]:
    if not _client.bucket_exists(settings.minio_bucket):
        return []
    return [obj.object_name for obj in _client.list_objects(settings.minio_bucket, recursive=True)]


async def ensure_bucket() -> None:
    await asyncio.to_thread(_ensure_bucket_sync)


async def upload_image(object_key: str, image_bytes: bytes, content_type: str = "image/png") -> str:
    """Uploads to the configured bucket and returns the object key - the
    value stored in documents.source_image_path for later retrieval."""
    await asyncio.to_thread(_upload_sync, object_key, image_bytes, content_type)
    return object_key


async def get_image_bytes(object_key: str) -> bytes:
    return await asyncio.to_thread(_download_sync, object_key)


async def list_image_keys() -> list[str]:
    return await asyncio.to_thread(_list_keys_sync)
