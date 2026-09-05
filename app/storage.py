import asyncio
import io
from datetime import timedelta

from minio import Minio

from app.config import settings

DOCUMENT_PREFIX = "documents"

_CONTENT_TYPE_BY_SUFFIX = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".txt": "text/plain",
    ".md": "text/markdown",
}


def document_key_for(filename: str) -> str:
    """The MinIO object key an original uploaded file is stored/looked up
    under - keyed by filename so re-uploading the same filename overwrites
    it, consistent with the DB's per-filename upsert semantics."""
    return f"{DOCUMENT_PREFIX}/{filename}"

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


def _object_exists_sync(object_key: str) -> bool:
    try:
        _client.stat_object(settings.minio_bucket, object_key)
        return True
    except Exception:
        return False


def _presigned_url_sync(object_key: str, expires_seconds: int) -> str:
    return _client.presigned_get_object(
        settings.minio_bucket, object_key, expires=timedelta(seconds=expires_seconds)
    )


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


async def upload_document(filename: str, file_bytes: bytes) -> str:
    """Uploads the original file a user submitted to /upload, so /ask
    responses can link back to the source document. Returns the object key
    stored under documents.source_document_path is implicit (derivable via
    document_key_for(filename)) - callers don't need to persist it."""
    suffix = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    content_type = _CONTENT_TYPE_BY_SUFFIX.get(suffix, "application/octet-stream")
    key = document_key_for(filename)
    await asyncio.to_thread(_upload_sync, key, file_bytes, content_type)
    return key


async def get_document_url(filename: str, expires_seconds: int = 3600) -> str | None:
    """Presigned, time-limited URL to the original uploaded file, or None
    if it was never stored (e.g. ingested before this feature existed)."""
    key = document_key_for(filename)
    exists = await asyncio.to_thread(_object_exists_sync, key)
    if not exists:
        return None
    return await asyncio.to_thread(_presigned_url_sync, key, expires_seconds)
