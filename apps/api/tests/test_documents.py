from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from app.database import SessionLocal, get_session
from app.document_models import DocumentProcessingJob, ProcessingJobState
from app.documents import MAX_FILE_SIZE_BYTES, Document
from app.main import app
from app.storage import ObjectStorageError
from app.task_dispatch import get_processing_task_dispatcher


class RecordingDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, UUID]] = []
        self.fail_dispatch = False
        self.committed_records_seen = False

    def dispatch(self, job_id: UUID, document_id: UUID) -> str:
        verification_session = SessionLocal()
        try:
            document = verification_session.get(Document, document_id)
            processing_job = verification_session.get(DocumentProcessingJob, job_id)
            self.committed_records_seen = document is not None and processing_job is not None
        finally:
            verification_session.close()

        self.calls.append((job_id, document_id))
        if self.fail_dispatch:
            raise RuntimeError("Broker connection included secret infrastructure details")
        return f"celery-{job_id}"


@pytest.fixture(autouse=True)
def dispatcher() -> RecordingDispatcher:
    recording_dispatcher = RecordingDispatcher()
    app.dependency_overrides[get_processing_task_dispatcher] = lambda: recording_dispatcher
    yield recording_dispatcher
    app.dependency_overrides.pop(get_processing_task_dispatcher, None)


class RecordingStorage:
    def __init__(self) -> None:
        self.deleted_object_keys: list[str] = []
        self.uploaded_files: list[dict[str, object]] = []
        self.fail_upload = False

    def upload_file(self, file, object_key: str, content_type: str) -> None:
        if self.fail_upload:
            raise ObjectStorageError("Object storage upload failed.")
        self.uploaded_files.append(
            {
                "content": file.read(),
                "content_type": content_type,
                "object_key": object_key,
            }
        )

    def delete_file(self, object_key: str) -> None:
        self.deleted_object_keys.append(object_key)


@pytest.fixture
def storage(monkeypatch) -> RecordingStorage:
    recording_storage = RecordingStorage()
    monkeypatch.setattr("app.documents.get_storage_client", lambda: recording_storage)
    return recording_storage


def create_knowledge_base(client) -> str:
    response = client.post("/knowledge-bases", json={"name": "Upload tests"})

    assert response.status_code == 201
    return response.json()["id"]


def test_upload_document_stores_object_metadata_and_queued_job(
    client, storage, dispatcher
) -> None:
    knowledge_base_id = create_knowledge_base(client)

    response = client.post(
        f"/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("notes.TXT", b"A small source document.", "text/plain")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["knowledge_base_id"] == knowledge_base_id
    assert body["original_filename"] == "notes.TXT"
    assert body["file_size_bytes"] == len(b"A small source document.")
    assert body["file_type"] == "txt"
    assert body["status"] == "Uploaded"
    assert datetime.fromisoformat(body["uploaded_at"]).tzinfo is not None
    assert storage.uploaded_files[0]["content"] == b"A small source document."

    session = SessionLocal()
    try:
        document = session.scalar(select(Document))
        processing_job = session.scalar(select(DocumentProcessingJob))
        assert document is not None
        assert processing_job is not None
        assert str(document.knowledge_base_id) == body["knowledge_base_id"]
        assert document.status == "Uploaded"
        assert document.storage_object_key == storage.uploaded_files[0]["object_key"]
        assert processing_job.document_id == document.id
        assert processing_job.state == ProcessingJobState.QUEUED
        assert processing_job.celery_task_id == f"celery-{processing_job.id}"
    finally:
        session.close()

    assert dispatcher.calls == [(processing_job.id, document.id)]
    assert dispatcher.committed_records_seen is True


def test_upload_document_dispatch_failure_keeps_durable_queued_job(
    client, storage, dispatcher, caplog
) -> None:
    knowledge_base_id = create_knowledge_base(client)
    dispatcher.fail_dispatch = True

    response = client.post(
        f"/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("notes.txt", b"A small source document.", "text/plain")},
    )

    assert response.status_code == 201
    assert response.json()["status"] == "Uploaded"
    assert "secret infrastructure details" not in caplog.text

    session = SessionLocal()
    try:
        document = session.scalar(select(Document))
        processing_job = session.scalar(select(DocumentProcessingJob))
        assert document is not None
        assert processing_job is not None
        assert processing_job.document_id == document.id
        assert processing_job.state == ProcessingJobState.QUEUED
        assert processing_job.celery_task_id is None
    finally:
        session.close()

    assert dispatcher.calls == [(processing_job.id, document.id)]
    assert dispatcher.committed_records_seen is True


def test_list_documents_returns_an_empty_list_for_a_new_knowledge_base(client) -> None:
    knowledge_base_id = create_knowledge_base(client)

    response = client.get(f"/knowledge-bases/{knowledge_base_id}/documents")

    assert response.status_code == 200
    assert response.json() == []


def test_list_documents_returns_metadata_newest_first_and_for_one_knowledge_base(client) -> None:
    knowledge_base_id = create_knowledge_base(client)
    other_knowledge_base_id = client.post(
        "/knowledge-bases", json={"name": "Another knowledge base"}
    ).json()["id"]
    knowledge_base_uuid = UUID(knowledge_base_id)
    other_knowledge_base_uuid = UUID(other_knowledge_base_id)
    uploaded_at = datetime(2026, 9, 13, 8, 30, tzinfo=UTC)
    old_document = Document(
        knowledge_base_id=knowledge_base_uuid,
        original_filename="older.txt",
        file_size_bytes=123,
        file_type="txt",
        status="Uploaded",
        storage_object_key=f"documents/{uuid4()}",
        uploaded_at=uploaded_at,
    )
    newest_document = Document(
        knowledge_base_id=knowledge_base_uuid,
        original_filename="architecture.pdf",
        file_size_bytes=3_145_728,
        file_type="pdf",
        status="Uploaded",
        storage_object_key=f"documents/{uuid4()}",
        uploaded_at=uploaded_at + timedelta(minutes=1),
    )
    other_document = Document(
        knowledge_base_id=other_knowledge_base_uuid,
        original_filename="private.docx",
        file_size_bytes=456,
        file_type="docx",
        status="Uploaded",
        storage_object_key=f"documents/{uuid4()}",
        uploaded_at=uploaded_at + timedelta(minutes=2),
    )
    session = SessionLocal()
    try:
        session.add_all([old_document, newest_document, other_document])
        session.flush()
        old_document_id = str(old_document.id)
        newest_document_id = str(newest_document.id)
        session.commit()
    finally:
        session.close()

    response = client.get(f"/knowledge-bases/{knowledge_base_id}/documents")

    assert response.status_code == 200
    body = response.json()
    assert [document["id"] for document in body] == [
        newest_document_id,
        old_document_id,
    ]
    assert body[0] == {
        "id": newest_document_id,
        "knowledge_base_id": knowledge_base_id,
        "original_filename": "architecture.pdf",
        "file_size_bytes": 3_145_728,
        "file_type": "pdf",
        "status": "Uploaded",
        "uploaded_at": "2026-09-13T08:31:00Z",
        "processing_started_at": None,
        "processing_completed_at": None,
        "failure_code": None,
        "failure_message": None,
    }


def test_list_documents_handles_metadata_availability_failure(client) -> None:
    knowledge_base_id = create_knowledge_base(client)

    class UnavailableDocumentSession:
        def get(self, *_args):
            return object()

        def scalars(self, _statement):
            raise OperationalError("SELECT documents", {}, Exception("Database unavailable"))

        def close(self) -> None:
            pass

    app.dependency_overrides[get_session] = UnavailableDocumentSession
    try:
        response = client.get(f"/knowledge-bases/{knowledge_base_id}/documents")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 503
    assert response.json() == {"detail": "Document metadata is unavailable. Please try again."}


def test_upload_document_rejects_unsupported_file_before_storage(client, storage) -> None:
    knowledge_base_id = create_knowledge_base(client)

    response = client.post(
        f"/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("notes.csv", b"name,value", "text/csv")},
    )

    assert response.status_code == 415
    assert response.json() == {"detail": "Unsupported file type. Upload a PDF, TXT, or DOCX file."}
    assert storage.uploaded_files == []


def test_upload_document_rejects_oversized_file_before_storage(client, storage) -> None:
    knowledge_base_id = create_knowledge_base(client)
    oversized_file = b"x" * (MAX_FILE_SIZE_BYTES + 1)

    response = client.post(
        f"/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("large.txt", oversized_file, "text/plain")},
    )

    assert response.status_code == 413
    assert response.json() == {"detail": "File is too large. The maximum size is 25 MB."}
    assert storage.uploaded_files == []


def test_upload_document_rejects_multiple_files(client, storage) -> None:
    knowledge_base_id = create_knowledge_base(client)

    response = client.post(
        f"/knowledge-bases/{knowledge_base_id}/documents",
        files=[
            ("file", ("first.txt", b"First source document.", "text/plain")),
            ("file", ("second.txt", b"Second source document.", "text/plain")),
        ],
    )

    assert response.status_code == 422
    assert response.json() == {"detail": "Upload exactly one file at a time."}
    assert storage.uploaded_files == []


def test_upload_document_allows_duplicate_filenames(client, storage) -> None:
    knowledge_base_id = create_knowledge_base(client)
    endpoint = f"/knowledge-bases/{knowledge_base_id}/documents"

    first_response = client.post(
        endpoint,
        files={"file": ("notes.txt", b"First source document.", "text/plain")},
    )
    second_response = client.post(
        endpoint,
        files={"file": ("notes.txt", b"Second source document.", "text/plain")},
    )

    assert first_response.status_code == 201
    assert second_response.status_code == 201
    assert first_response.json()["id"] != second_response.json()["id"]
    assert storage.uploaded_files[0]["object_key"] != storage.uploaded_files[1]["object_key"]

    session = SessionLocal()
    try:
        documents = session.scalars(select(Document)).all()
        processing_jobs = session.scalars(select(DocumentProcessingJob)).all()
        assert len(documents) == 2
        assert len(processing_jobs) == 2
        assert {job.document_id for job in processing_jobs} == {
            document.id for document in documents
        }
    finally:
        session.close()


def test_upload_document_handles_storage_failure_without_document_record(client, storage) -> None:
    knowledge_base_id = create_knowledge_base(client)
    storage.fail_upload = True

    response = client.post(
        f"/knowledge-bases/{knowledge_base_id}/documents",
        files={"file": ("notes.txt", b"A small source document.", "text/plain")},
    )

    assert response.status_code == 503
    assert response.json() == {"detail": "Document storage is unavailable. Please try again."}

    session = SessionLocal()
    try:
        assert session.scalars(select(Document)).all() == []
        assert session.scalars(select(DocumentProcessingJob)).all() == []
    finally:
        session.close()


def test_upload_document_requires_an_existing_knowledge_base(client, storage) -> None:
    response = client.post(
        f"/knowledge-bases/{uuid4()}/documents",
        files={"file": ("notes.txt", b"A small source document.", "text/plain")},
    )

    assert response.status_code == 404
    assert storage.uploaded_files == []
