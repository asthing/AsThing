import importlib
import sys
import time

from sqlalchemy import select

from app.database import SessionLocal
from app.document_models import (
    Document,
    DocumentProcessingJob,
    DocumentStatus,
    ProcessingJobState,
)

MAX_REDIS_DOWN_UPLOAD_SECONDS = 2.0


class RecordingStorage:
    def upload_file(self, _file, _object_key: str, _content_type: str) -> None:
        return None

    def delete_file(self, _object_key: str) -> None:
        return None


def test_unreachable_redis_keeps_upload_fast_and_durable(
    client, monkeypatch, caplog
) -> None:
    importlib.import_module("celery")
    monkeypatch.setenv("REDIS_URL", "redis://127.0.0.1:1/0")
    monkeypatch.setattr("app.documents.get_storage_client", RecordingStorage)
    sys.modules.pop("app.celery_app", None)

    knowledge_base_response = client.post(
        "/knowledge-bases", json={"name": "Redis unavailable"}
    )
    assert knowledge_base_response.status_code == 201
    knowledge_base_id = knowledge_base_response.json()["id"]

    started = time.perf_counter()
    try:
        response = client.post(
            f"/knowledge-bases/{knowledge_base_id}/documents",
            files={"file": ("queued.txt", b"durable queued content", "text/plain")},
        )
    finally:
        elapsed = time.perf_counter() - started
        sys.modules.pop("app.celery_app", None)
        importlib.invalidate_caches()

    assert response.status_code == 201
    assert elapsed < MAX_REDIS_DOWN_UPLOAD_SECONDS
    assert response.json()["status"] == "Uploaded"
    assert "127.0.0.1" not in caplog.text

    session = SessionLocal()
    try:
        document = session.scalar(select(Document))
        job = session.scalar(select(DocumentProcessingJob))
        assert document is not None
        assert job is not None
        assert document.status == DocumentStatus.UPLOADED
        assert job.document_id == document.id
        assert job.state == ProcessingJobState.QUEUED
        assert job.celery_task_id is None
    finally:
        session.close()
