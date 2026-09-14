from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from app.database import SessionLocal
from app.document_models import (
    Document,
    DocumentChunk,
    DocumentProcessingJob,
    DocumentStatus,
    ProcessingJobState,
)
from app.knowledge_bases import KnowledgeBase


def create_document() -> Document:
    session = SessionLocal()
    try:
        knowledge_base = KnowledgeBase(
            name=f"Models {uuid4()}", normalized_name=f"models-{uuid4()}"
        )
        session.add(knowledge_base)
        session.flush()
        document = Document(
            knowledge_base_id=knowledge_base.id,
            original_filename="source.txt",
            file_size_bytes=12,
            file_type="txt",
            storage_object_key=f"documents/{uuid4()}.txt",
        )
        session.add(document)
        session.commit()
        session.refresh(document)
        session.expunge(document)
        return document
    finally:
        session.close()


def test_public_and_internal_status_values_are_separate() -> None:
    assert {status.value for status in DocumentStatus} == {
        "Uploaded",
        "Processing",
        "Ready",
        "Failed",
    }
    assert {state.value for state in ProcessingJobState} == {
        "Queued",
        "Processing",
        "Completed",
        "CleanupPending",
    }
    assert "CleanupPending" not in {status.value for status in DocumentStatus}


def test_processing_job_has_recovery_index() -> None:
    index_names = {index.name for index in DocumentProcessingJob.__table__.indexes}

    assert "ix_document_processing_jobs_recovery" in index_names


def test_document_defaults_to_uploaded_with_no_processing_attempts() -> None:
    document = create_document()

    assert document.status == DocumentStatus.UPLOADED
    assert document.processing_attempt_count == 0
    assert document.processing_started_at is None
    assert document.processing_completed_at is None
    assert document.failure_code is None
    assert document.failure_message is None


def test_document_rejects_non_public_status() -> None:
    document = create_document()
    session = SessionLocal()
    try:
        stored_document = session.get(Document, document.id)
        assert stored_document is not None
        stored_document.status = ProcessingJobState.CLEANUP_PENDING
        with pytest.raises(IntegrityError):
            session.commit()
    finally:
        session.rollback()
        session.close()


def test_processing_job_is_unique_per_document_and_has_recovery_defaults() -> None:
    document = create_document()
    session = SessionLocal()
    try:
        job = DocumentProcessingJob(document_id=document.id)
        session.add(job)
        session.commit()
        session.refresh(job)

        assert job.state == ProcessingJobState.QUEUED
        assert job.attempt_count == 0
        assert job.lease_expires_at is None
        assert job.last_heartbeat_at is None
        assert job.queued_at is not None
        assert job.updated_at is not None

        session.add(DocumentProcessingJob(document_id=document.id))
        with pytest.raises(IntegrityError):
            session.commit()
    finally:
        session.rollback()
        session.close()


def test_processing_job_rejects_public_document_status_as_internal_state() -> None:
    document = create_document()
    session = SessionLocal()
    try:
        session.add(DocumentProcessingJob(document_id=document.id, state=DocumentStatus.READY))
        with pytest.raises(IntegrityError):
            session.commit()
    finally:
        session.rollback()
        session.close()


def test_chunk_ordinal_and_hash_are_unique_within_a_document() -> None:
    document = create_document()
    session = SessionLocal()
    try:
        session.add(
            DocumentChunk(
                document_id=document.id,
                ordinal=0,
                content="First chunk",
                content_hash="a" * 64,
                qdrant_point_id=uuid4(),
            )
        )
        session.commit()
        session.add(
            DocumentChunk(
                document_id=document.id,
                ordinal=0,
                content="Duplicate ordinal",
                content_hash="b" * 64,
                qdrant_point_id=uuid4(),
            )
        )
        with pytest.raises(IntegrityError):
            session.commit()
    finally:
        session.rollback()
        session.close()
