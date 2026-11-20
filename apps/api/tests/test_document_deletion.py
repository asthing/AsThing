import os
import subprocess
import sys
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import boto3
import pytest
from botocore.exceptions import ClientError, EndpointConnectionError
from botocore.stub import Stubber
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session

from app.answers import AnswerService
from app.database import SessionLocal, engine
from app.document_deletion import (
    DELETION_INCOMPLETE,
    PROCESSING_CONFLICT,
    DocumentDeletionService,
    DocumentDeletionUnavailable,
    get_document_deletion_service,
    recover_document_deletions,
)
from app.document_models import (
    Document,
    DocumentChunk,
    DocumentProcessingJob,
    utc_now,
)
from app.knowledge_bases import KnowledgeBase
from app.main import app
from app.processing import DocumentProcessingError, StoredDocumentProcessor, prepare_chunks
from app.processing_jobs import ClaimedProcessingJob, claim_processing_job, recover_processing_jobs
from app.retrieval import SourceRetrievalService
from app.storage import ObjectStorage, ObjectStorageError
from app.vector_store import (
    ChunkVector,
    QdrantVectorStore,
    VectorStoreError,
    VectorStoreErrorCategory,
    _QdrantClientAdapter,
)
from app.worker_execution import run_processing_task


def seed_document(status="Ready", *, knowledge_base_id=None, filename="same.txt", with_job=True):
    with SessionLocal() as session:
        if knowledge_base_id is None:
            kb = KnowledgeBase(name=str(uuid4()), normalized_name=str(uuid4()))
            session.add(kb)
            session.flush()
            knowledge_base_id = kb.id
        document = Document(
            knowledge_base_id=knowledge_base_id, original_filename=filename,
            file_size_bytes=100, file_type="txt", status=status,
            storage_object_key=f"test/{uuid4()}", processing_attempt_count=1,
        )
        session.add(document)
        session.flush()
        job = DocumentProcessingJob(
            document_id=document.id,
            state="Queued" if status == "Uploaded" else "Processing" if status == "Processing"
            else "Completed", attempt_count=1, lease_expires_at=utc_now() + timedelta(minutes=5),
        )
        if with_job:
            session.add(job)
        prepared = prepare_chunks(document.id, "A fictional source fact.")[0]
        session.add(DocumentChunk(
            id=prepared.id, document_id=document.id, ordinal=0, content=prepared.content,
            content_hash=prepared.content_hash, qdrant_point_id=prepared.qdrant_point_id,
        ))
        session.commit()
        return knowledge_base_id, document.id, document.storage_object_key, job.id


class Externals:
    def __init__(self):
        self.calls = []
        self.fail_vector = False
        self.fail_storage = False

    def delete_document(self, document_id):
        self.calls.append(("vectors", document_id))
        with SessionLocal() as session:
            assert session.get(Document, document_id).deletion_pending
        if self.fail_vector:
            raise VectorStoreError(VectorStoreErrorCategory.DELETE_FAILED)

    def delete_file_strict(self, key):
        self.calls.append(("object", key))
        if self.fail_storage:
            raise ObjectStorageError("PRIVATE raw endpoint credentials")


@pytest.fixture
def deletion(client):
    externals = Externals()
    service = DocumentDeletionService(
        storage_provider=lambda: externals, vector_store_provider=lambda: externals,
    )
    app.dependency_overrides[get_document_deletion_service] = lambda: service
    yield service, externals
    app.dependency_overrides.pop(get_document_deletion_service, None)


def remaining(document_id):
    with SessionLocal() as session:
        return (
            session.get(Document, document_id),
            list(session.scalars(
                select(DocumentChunk).where(DocumentChunk.document_id == document_id)
            )),
            list(session.scalars(
                select(DocumentProcessingJob)
                .where(DocumentProcessingJob.document_id == document_id)
            )),
        )


@pytest.mark.parametrize("status", ["Uploaded", "Ready", "Failed"])
def test_delete_removes_all_metadata_and_preserves_same_named_documents(client, deletion, status):
    _, externals = deletion
    kb, target, key, _ = seed_document(status)
    _, preserved, _, _ = seed_document(knowledge_base_id=kb)
    other_kb, other, _, _ = seed_document()
    response = client.delete(f"/knowledge-bases/{kb}/documents/{target}")
    assert response.status_code == 204
    assert response.content == b""
    assert externals.calls == [("vectors", target), ("object", key)]
    assert remaining(target) == (None, [], [])
    assert remaining(preserved)[0] is not None
    assert remaining(other)[0].knowledge_base_id == other_kb
    listed = client.get(f"/knowledge-bases/{kb}/documents").json()
    assert [item["id"] for item in listed] == [str(preserved)]
    assert listed[0]["deletion_pending"] is False


def test_processing_and_wrong_knowledge_base_are_rejected_without_external_calls(client, deletion):
    _, externals = deletion
    kb, document, _, _ = seed_document("Processing")
    response = client.delete(f"/knowledge-bases/{kb}/documents/{document}")
    assert response.status_code == 409
    assert response.json() == {"detail": PROCESSING_CONFLICT}
    other_kb, _, _, _ = seed_document()
    response = client.delete(f"/knowledge-bases/{other_kb}/documents/{document}")
    assert response.status_code == 404
    assert externals.calls == []
    assert not remaining(document)[0].deletion_pending


@pytest.mark.parametrize("failure", ["fail_vector", "fail_storage"])
def test_partial_failure_retains_identifiers_and_is_retryable(client, deletion, failure, caplog):
    _, externals = deletion
    kb, document, key, _ = seed_document()
    setattr(externals, failure, True)
    response = client.delete(f"/knowledge-bases/{kb}/documents/{document}")
    assert response.status_code == 503
    assert response.json() == {"detail": DELETION_INCOMPLETE}
    retained, chunks, jobs = remaining(document)
    assert retained.deletion_pending and retained.storage_object_key == key
    assert chunks and jobs
    assert client.get(f"/knowledge-bases/{kb}/documents").json()[0]["deletion_pending"]
    assert "PRIVATE" not in response.text + caplog.text
    setattr(externals, failure, False)
    assert client.delete(f"/knowledge-bases/{kb}/documents/{document}").status_code == 204
    assert client.delete(f"/knowledge-bases/{kb}/documents/{document}").status_code == 404
    assert remaining(document) == (None, [], [])


def test_external_success_then_commit_failure_repeats_cleanup_safely():
    kb, document, _, _ = seed_document()
    externals = Externals()

    class CommitFailure(Session):
        def commit(self):
            if any(isinstance(item, Document) for item in self.deleted):
                raise OperationalError("private sql", {}, Exception("private infrastructure"))
            return super().commit()

    service = DocumentDeletionService(
        session_factory=lambda: CommitFailure(bind=engine), storage_provider=lambda: externals,
        vector_store_provider=lambda: externals,
    )
    with pytest.raises(DocumentDeletionUnavailable):
        service.delete(kb, document)
    retained, chunks, jobs = remaining(document)
    assert retained.deletion_pending and chunks and jobs
    service.session_factory = SessionLocal
    assert service.resume(document)
    calls = len(externals.calls)
    assert service.resume(document)
    assert len(externals.calls) == calls
    assert remaining(document) == (None, [], [])


def test_no_job_legacy_document_and_empty_list(client, deletion):
    kb, document, _, _ = seed_document(with_job=False)
    assert client.delete(f"/knowledge-bases/{kb}/documents/{document}").status_code == 204
    assert client.get(f"/knowledge-bases/{kb}/documents").json() == []


def test_delete_never_reports_success_when_cleanup_did_not_complete(client, deletion, monkeypatch):
    service, externals = deletion
    kb, document, _, _ = seed_document()
    monkeypatch.setattr(service, "resume", lambda *_args, **_kwargs: False)
    response = client.delete(f"/knowledge-bases/{kb}/documents/{document}")
    assert response.status_code == 503
    assert remaining(document)[0].deletion_pending
    assert externals.calls == []


def test_recovery_is_bounded_publishes_outside_transaction_and_preserves_failed_intent():
    ids = []
    for _ in range(3):
        _, document, _, _ = seed_document()
        with SessionLocal() as session:
            session.get(Document, document).deletion_requested_at = utc_now()
            session.commit()
        ids.append(document)
    calls = []

    class Dispatcher:
        def dispatch(self, document_id):
            calls.append(document_id)
            raise RuntimeError("PRIVATE broker error")

    result = recover_document_deletions(dispatcher=Dispatcher(), batch_size=2)
    assert result == {"found": 2, "dispatched": 0, "dispatch_failed": 2}
    assert len(calls) == 2
    assert all(remaining(document)[0].deletion_pending for document in ids)


def test_pending_and_deleted_documents_cannot_process_or_supply_evidence(deletion):
    service, externals = deletion
    kb, document, _, _ = seed_document("Uploaded")
    externals.fail_vector = True
    with pytest.raises(DocumentDeletionUnavailable):
        service.delete(kb, document)
    assert claim_processing_job(document) is None
    assert not run_processing_task(document, processor_hook=lambda _: pytest.fail("must not run"))

    class Dispatcher:
        def dispatch(self, *_args):
            pytest.fail("pending deletion must not be dispatched for processing")

    assert recover_processing_jobs(Dispatcher()).found == 0
    with SessionLocal() as session:
        session.get(Document, document).status = "Ready"
        chunk_id = session.scalar(select(DocumentChunk.id))
        session.commit()
    retriever = SourceRetrievalService()
    assert retriever._hydrate_ready_chunks(kb, [chunk_id]) == {}
    assert not retriever._has_ready_documents(kb)
    answer = AnswerService(retriever=retriever).answer(kb, "Fictional fact?")
    assert answer.reason == "no_ready_documents" and answer.references == []
    externals.fail_vector = False
    service.resume(document)
    assert not run_processing_task(document, processor_hook=lambda _: pytest.fail("must not run"))


def test_stale_attempt_cannot_persist_chunks_or_vectors():
    _, document, _, job_id = seed_document()
    claim = ClaimedProcessingJob(job_id, document, 1, utc_now() + timedelta(minutes=5))
    vectors = SimpleNamespace(upsert=lambda _: pytest.fail("stale vector write"))
    processor = StoredDocumentProcessor(vector_store=vectors)
    with pytest.raises(DocumentProcessingError):
        processor.persist(claim, prepare_chunks(document, "stale content"))
    vector = ChunkVector(uuid4(), [0.1] * 1024, uuid4(), document, uuid4(), 0, "model", "revision")
    with pytest.raises(VectorStoreError):
        processor.upsert_vectors(claim, [vector])
    assert remaining(document)[1][0].content == "A fictional source fact."


@pytest.mark.parametrize("code", ["NoSuchKey", "NoSuchBucket", "AccessDenied", "InternalError"])
def test_strict_minio_delete_distinguishes_missing_from_infrastructure(monkeypatch, code):
    class Client:
        def close(self):
            pass

        def delete_object(self, **_kwargs):
            raise ClientError({"Error": {"Code": code, "Message": "PRIVATE"}}, "DeleteObject")

    options = []
    monkeypatch.setattr("app.storage.boto3.client", lambda *a, **k: options.append(k) or Client())
    for key in ("S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(key, "private-test-value")
    storage = ObjectStorage.__new__(ObjectStorage)
    storage.bucket = "test"
    if code in {"NoSuchKey", "NoSuchBucket"}:
        storage.delete_file_strict("test-key")
    else:
        with pytest.raises(ObjectStorageError, match="Object storage deletion failed"):
            storage.delete_file_strict("test-key")
    assert options[0]["config"].retries == {"total_max_attempts": 1}


def test_upload_compensation_still_swallows_storage_failure():
    def fail(**_kwargs):
        raise EndpointConnectionError(endpoint_url="PRIVATE")

    storage = ObjectStorage.__new__(ObjectStorage)
    storage.bucket = "test"
    storage.client = SimpleNamespace(delete_object=fail)
    storage.delete_file("test-key")


def test_strict_delete_uses_a_real_boto_client_with_stubbed_response(monkeypatch):
    client = boto3.client(
        "s3", region_name="us-east-1", aws_access_key_id="test", aws_secret_access_key="test",
    )
    stubber = Stubber(client)
    stubber.add_response("delete_object", {}, {"Bucket": "test", "Key": "test-key"})
    monkeypatch.setattr("app.storage.boto3.client", lambda *a, **k: client)
    for name in ("S3_ENDPOINT_URL", "S3_ACCESS_KEY_ID", "S3_SECRET_ACCESS_KEY"):
        monkeypatch.setenv(name, "test")
    storage = ObjectStorage.__new__(ObjectStorage)
    storage.bucket = "test"
    with stubber:
        storage.delete_file_strict("test-key")
        stubber.assert_no_pending_responses()


@pytest.mark.parametrize("status", ["completed", "acknowledged"])
def test_qdrant_deletion_requires_completion_and_scopes_filter_to_document(status):
    calls = []
    document_id = uuid4()
    adapter = _QdrantClientAdapter.__new__(_QdrantClientAdapter)
    adapter.client = SimpleNamespace(
        collection_exists=lambda _name: True,
        delete=lambda **kwargs: calls.append(kwargs) or SimpleNamespace(status=status),
    )
    store = QdrantVectorStore(adapter_factory=lambda: adapter)
    if status == "completed":
        store.delete_document(document_id)
    else:
        with pytest.raises(VectorStoreError):
            store.delete_document(document_id)
    assert calls[0]["wait"] is True
    assert calls[0]["points_selector"].filter.must[0].key == "document_id"
    assert calls[0]["points_selector"].filter.must[0].match.value == str(document_id)


def test_missing_vector_collection_is_already_cleaned_but_unavailable_is_not():
    adapter = _QdrantClientAdapter.__new__(_QdrantClientAdapter)
    adapter.client = SimpleNamespace(
        collection_exists=lambda _name: False,
        delete=lambda **_kwargs: pytest.fail("missing collection must not be mutated"),
    )
    store = QdrantVectorStore(adapter_factory=lambda: adapter)
    store.delete_document(uuid4())

    def unavailable(_name):
        raise RuntimeError("PRIVATE vector endpoint")

    adapter.client.collection_exists = unavailable
    with pytest.raises(VectorStoreError) as error:
        store.delete_document(uuid4())
    assert "PRIVATE" not in str(error.value)


def test_worker_only_deletion_import_registers_foreign_key_metadata():
    environment = {**os.environ, "DATABASE_URL": "sqlite+pysqlite://"}
    completed = subprocess.run(
        [sys.executable, "-c", (
            "from app.document_deletion import DocumentDeletionService; "
            "from app.database import Base, engine; "
            "assert 'knowledge_bases' in Base.metadata.tables; "
            "Base.metadata.create_all(engine)"
        )],
        env=environment, capture_output=True, text=True, timeout=20,
    )
    assert completed.returncode == 0, "worker-only deletion metadata import failed"
