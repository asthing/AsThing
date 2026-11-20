from datetime import timedelta
from uuid import uuid4

import pytest
from sqlalchemy import select
from test_document_deletion import remaining, seed_document

from app.database import SessionLocal
from app.document_deletion import DocumentDeletionService
from app.document_models import DocumentChunk, DocumentProcessingJob, utc_now
from app.knowledge_base_deletion import (
    DELETION_INCOMPLETE,
    KnowledgeBaseDeletionService,
)
from app.knowledge_bases import KnowledgeBase, get_knowledge_base_deletion_service
from app.main import app
from app.vector_store import VectorStoreError, VectorStoreErrorCategory


class Externals:
    def __init__(self):
        self.deleted_vectors = []
        self.deleted_objects = []
        self.failure_document = None

    def delete_document(self, document_id):
        if document_id == self.failure_document:
            raise VectorStoreError(VectorStoreErrorCategory.DELETE_FAILED)
        self.deleted_vectors.append(document_id)

    def delete_file_strict(self, key):
        self.deleted_objects.append(key)


@pytest.fixture
def deletion(client):
    externals = Externals()
    documents = DocumentDeletionService(
        storage_provider=lambda: externals, vector_store_provider=lambda: externals,
    )
    service = KnowledgeBaseDeletionService(documents=documents)
    app.dependency_overrides[get_knowledge_base_deletion_service] = lambda: service
    yield service, externals
    app.dependency_overrides.pop(get_knowledge_base_deletion_service, None)


def test_delete_empty_knowledge_base_and_missing_repeat(client, deletion):
    created = client.post("/knowledge-bases", json={"name": "Empty"}).json()
    response = client.delete(f"/knowledge-bases/{created['id']}")
    assert response.status_code == 204 and response.content == b""
    assert client.get("/knowledge-bases").json() == []
    assert client.delete(f"/knowledge-bases/{created['id']}").status_code == 404


def test_cascade_uploaded_ready_failed_and_preserve_another_kb(client, deletion):
    _, externals = deletion
    kb, uploaded, key1, _ = seed_document("Uploaded")
    _, ready, key2, _ = seed_document("Ready", knowledge_base_id=kb)
    _, failed, key3, _ = seed_document("Failed", knowledge_base_id=kb)
    other_kb, keeper, _, _ = seed_document("Ready")  # Same filename, independent KB.
    response = client.delete(f"/knowledge-bases/{kb}")
    assert response.status_code == 204
    assert set(externals.deleted_vectors) == {uploaded, ready, failed}
    assert set(externals.deleted_objects) == {key1, key2, key3}
    for doc in (uploaded, ready, failed):
        assert remaining(doc) == (None, [], [])
    assert remaining(keeper)[0].knowledge_base_id == other_kb
    assert [item["id"] for item in client.get("/knowledge-bases").json()] == [str(other_kb)]


def test_processing_blocks_entire_request_before_any_document_cleanup(client, deletion):
    _, externals = deletion
    kb, ready, _, _ = seed_document("Ready")
    _, processing, _, _ = seed_document("Processing", knowledge_base_id=kb)
    response = client.delete(f"/knowledge-bases/{kb}")
    assert response.status_code == 409
    assert "processing" in response.json()["detail"]
    assert externals.deleted_vectors == [] and externals.deleted_objects == []
    assert remaining(ready)[0] is not None and remaining(processing)[0] is not None


def test_partial_failure_keeps_parent_and_durable_intent_then_retry_finishes(client, deletion):
    _, externals = deletion
    kb, one, _, _ = seed_document("Ready")
    _, two, _, _ = seed_document("Ready", knowledge_base_id=kb)
    ordered = sorted([one, two])
    externals.failure_document = ordered[1]
    response = client.delete(f"/knowledge-bases/{kb}")
    assert response.status_code == 503
    assert response.json() == {"detail": DELETION_INCOMPLETE}
    assert remaining(ordered[0]) == (None, [], [])
    assert remaining(ordered[1])[0].deletion_pending
    with SessionLocal() as session:
        assert session.get(KnowledgeBase, kb) is not None
    externals.failure_document = None
    assert client.delete(f"/knowledge-bases/{kb}").status_code == 204
    with SessionLocal() as session:
        assert session.get(KnowledgeBase, kb) is None


def test_unresolved_write_never_discards_knowledge_base_or_identifiers(client, deletion):
    kb, target, _, job_id = seed_document("Failed")
    with SessionLocal() as session:
        job = session.get(DocumentProcessingJob, job_id)
        job.vector_write_token = uuid4()
        job.vector_write_started_at = utc_now() - timedelta(minutes=10)
        session.commit()
    for _ in range(2):
        response = client.delete(f"/knowledge-bases/{kb}")
        assert response.status_code == 503
        assert response.json() == {"detail": DELETION_INCOMPLETE}
        assert remaining(target)[0].deletion_pending
        with SessionLocal() as session:
            assert session.get(KnowledgeBase, kb) is not None
            assert session.scalar(select(DocumentChunk.id).where(
                DocumentChunk.document_id == target,
            )) is not None


def test_new_document_before_finalization_retains_parent(client, deletion):
    service, _ = deletion
    kb, target, _, _ = seed_document("Ready")
    original_delete = service.documents.delete
    new_document = None

    def delete_then_upload(knowledge_base_id, document_id):
        nonlocal new_document
        original_delete(knowledge_base_id, document_id)
        _, new_document, _, _ = seed_document("Uploaded", knowledge_base_id=knowledge_base_id)

    service.documents.delete = delete_then_upload
    response = client.delete(f"/knowledge-bases/{kb}")
    assert response.status_code == 409
    assert remaining(target) == (None, [], [])
    assert remaining(new_document)[0] is not None
    with SessionLocal() as session:
        assert session.get(KnowledgeBase, kb) is not None


def test_legacy_uploaded_document_without_job_can_be_removed(client, deletion):
    kb, target, _, _ = seed_document("Uploaded", with_job=False)
    assert client.delete(f"/knowledge-bases/{kb}").status_code == 204
    assert remaining(target) == (None, [], [])


def test_missing_kb_has_no_external_side_effects(client, deletion):
    _, externals = deletion
    assert client.delete(f"/knowledge-bases/{uuid4()}").json() == {
        "detail": "Knowledge base not found.",
    }
    assert externals.deleted_vectors == [] and externals.deleted_objects == []
