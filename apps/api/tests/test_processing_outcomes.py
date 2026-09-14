from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import event, select
from sqlalchemy.exc import OperationalError

from app.database import SessionLocal, engine
from app.document_models import (
    Document,
    DocumentChunk,
    DocumentProcessingJob,
    DocumentStatus,
    ProcessingJobState,
)
from app.embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_IDENTIFIER,
    EMBEDDING_MODEL_VERSION,
    EmbeddingBatch,
    EmbeddingErrorCategory,
    EmbeddingModelMetadata,
    EmbeddingServiceError,
)
from app.knowledge_bases import KnowledgeBase
from app.processing import (
    DocumentProcessingError,
    ProcessingErrorCategory,
    StoredDocumentProcessor,
    prepare_chunks,
)
from app.processing_jobs import claim_processing_job, recover_processing_jobs
from app.processing_outcomes import (
    FAILURE_MESSAGES,
    PendingProcessingCleanup,
    ProcessingFailureCode,
    ProcessingOutcomeError,
    SafeProcessingFailure,
    begin_processing_cleanup,
    complete_processing_cleanup,
    complete_processing_success,
    map_processing_failure,
)
from app.vector_store import VectorStoreError, VectorStoreErrorCategory
from app.worker_execution import run_processing_task

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"
START_TIME = datetime(2026, 9, 14, 10, 0, tzinfo=UTC)


class FakeStorage:
    def __init__(self, content: bytes) -> None:
        self.content = content

    def download_file(self, _object_key: str) -> bytes:
        return self.content


class FakeEmbeddingService:
    metadata = EmbeddingModelMetadata(
        identifier=EMBEDDING_MODEL_IDENTIFIER,
        version=EMBEDDING_MODEL_VERSION,
        dimension=EMBEDDING_DIMENSION,
    )

    def embed(self, texts) -> EmbeddingBatch:
        return EmbeddingBatch(
            vectors=[[0.1] * EMBEDDING_DIMENSION for _ in texts],
            model=self.metadata,
        )


class PartialFailureVectorStore:
    def __init__(self, document_id: UUID, job_id: UUID) -> None:
        self.document_id = document_id
        self.job_id = job_id
        self.points = set()
        self.deleted_point_ids = []
        self.deleted_document_ids = []
        self.cleanup_state_observations = []

    def upsert(self, vectors) -> None:
        self.points.add(vectors[0].point_id)
        raise VectorStoreError(VectorStoreErrorCategory.UPSERT_FAILED)

    def delete_points(self, point_ids) -> None:
        self._record_cleanup_state()
        self.deleted_point_ids.append(list(point_ids))
        self.points.difference_update(point_ids)

    def delete_document(self, document_id: UUID) -> None:
        self._record_cleanup_state()
        self.deleted_document_ids.append(document_id)
        self.points.clear()

    def _record_cleanup_state(self) -> None:
        document, job, chunks = load_state(self.document_id, self.job_id)
        self.cleanup_state_observations.append(
            (document.status, job.state, len(chunks))
        )


class RecordingDispatcher:
    def __init__(self) -> None:
        self.calls = []

    def dispatch(self, job_id: UUID, document_id: UUID) -> str:
        self.calls.append((job_id, document_id))
        return str(job_id)


class CleanupOnlyProcessor:
    def __init__(self) -> None:
        self.calls = []

    def delete_vectors(self, document_id: UUID, point_ids) -> None:
        self.calls.append((document_id, list(point_ids)))


def create_document_and_job() -> tuple[UUID, UUID]:
    session = SessionLocal()
    try:
        knowledge_base = KnowledgeBase(
            name=f"Outcome {uuid4()}", normalized_name=f"outcome-{uuid4()}"
        )
        session.add(knowledge_base)
        session.flush()
        document = Document(
            knowledge_base_id=knowledge_base.id,
            original_filename="outcome.txt",
            file_size_bytes=100,
            file_type="txt",
            storage_object_key=f"documents/{uuid4()}.txt",
        )
        session.add(document)
        session.flush()
        job = DocumentProcessingJob(document_id=document.id)
        session.add(job)
        session.commit()
        return document.id, job.id
    finally:
        session.close()


def claim_document(document_id: UUID):
    claim = claim_processing_job(document_id, now=START_TIME)
    assert claim is not None
    return claim


def persist_test_chunks(document_id: UUID) -> list[DocumentChunk]:
    prepared = prepare_chunks(document_id, "retrievable content " * 60)
    session = SessionLocal()
    try:
        chunks = [
            DocumentChunk(
                id=chunk.id,
                document_id=document_id,
                ordinal=chunk.ordinal,
                content=chunk.content,
                content_hash=chunk.content_hash,
                qdrant_point_id=chunk.qdrant_point_id,
            )
            for chunk in prepared
        ]
        session.add_all(chunks)
        session.commit()
        for chunk in chunks:
            _ = (chunk.id, chunk.qdrant_point_id)
            session.expunge(chunk)
        return chunks
    finally:
        session.close()


def load_state(
    document_id: UUID, job_id: UUID
) -> tuple[Document, DocumentProcessingJob, list[DocumentChunk]]:
    session = SessionLocal()
    try:
        document = session.get(Document, document_id)
        job = session.get(DocumentProcessingJob, job_id)
        chunks = list(
            session.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.ordinal)
            )
        )
        assert document is not None
        assert job is not None
        session.expunge(document)
        session.expunge(job)
        for chunk in chunks:
            session.expunge(chunk)
        return document, job, chunks
    finally:
        session.close()


def safe_failure(code: ProcessingFailureCode) -> SafeProcessingFailure:
    return SafeProcessingFailure(code=code, message=FAILURE_MESSAGES[code])


def test_success_transition_is_guarded_atomic_and_idempotent() -> None:
    document_id, job_id = create_document_and_job()
    claim = claim_document(document_id)
    persist_test_chunks(document_id)
    completion_time = START_TIME + timedelta(minutes=1)

    first = complete_processing_success(claim, now=completion_time)
    repeated = complete_processing_success(
        claim, now=completion_time + timedelta(minutes=1)
    )

    document, job, chunks = load_state(document_id, job_id)
    assert first is True
    assert repeated is True
    assert chunks
    assert document.status == DocumentStatus.READY
    assert document.processing_completed_at == completion_time.replace(tzinfo=None)
    assert document.failure_code is None
    assert document.failure_message is None
    assert job.state == ProcessingJobState.COMPLETED
    assert job.completed_at == completion_time.replace(tzinfo=None)
    assert job.failure_code is None
    assert job.failure_message is None


def test_success_transition_rejects_a_document_without_persisted_chunks() -> None:
    document_id, job_id = create_document_and_job()
    claim = claim_document(document_id)

    completed = complete_processing_success(
        claim, now=START_TIME + timedelta(minutes=1)
    )

    document, job, chunks = load_state(document_id, job_id)
    assert completed is False
    assert chunks == []
    assert document.status == DocumentStatus.PROCESSING
    assert job.state == ProcessingJobState.PROCESSING


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (
            DocumentProcessingError(ProcessingErrorCategory.SOURCE_OBJECT_UNAVAILABLE),
            ProcessingFailureCode.SOURCE_UNAVAILABLE,
        ),
        (
            DocumentProcessingError(
                ProcessingErrorCategory.UNSUPPORTED_OR_INVALID_SOURCE
            ),
            ProcessingFailureCode.UNSUPPORTED_DOCUMENT,
        ),
        (
            DocumentProcessingError(ProcessingErrorCategory.TEXT_EXTRACTION_FAILED),
            ProcessingFailureCode.TEXT_EXTRACTION_FAILED,
        ),
        (
            DocumentProcessingError(ProcessingErrorCategory.NO_USABLE_TEXT),
            ProcessingFailureCode.NO_USABLE_TEXT,
        ),
        (
            DocumentProcessingError(ProcessingErrorCategory.CHUNK_PERSISTENCE_FAILED),
            ProcessingFailureCode.CHUNK_PERSISTENCE_FAILED,
        ),
        (
            EmbeddingServiceError(EmbeddingErrorCategory.GENERATION_FAILED),
            ProcessingFailureCode.EMBEDDING_FAILED,
        ),
        (
            VectorStoreError(VectorStoreErrorCategory.UPSERT_FAILED),
            ProcessingFailureCode.VECTOR_PERSISTENCE_FAILED,
        ),
        (
            VectorStoreError(VectorStoreErrorCategory.DELETE_FAILED),
            ProcessingFailureCode.CLEANUP_FAILED,
        ),
    ],
)
def test_each_categorized_failure_starts_compensation(error, expected_code) -> None:
    document_id, job_id = create_document_and_job()
    claim = claim_document(document_id)
    failure = map_processing_failure(error)

    cleanup = begin_processing_cleanup(
        claim,
        failure,
        now=START_TIME + timedelta(seconds=1),
    )

    document, job, _ = load_state(document_id, job_id)
    assert cleanup is not None
    assert failure.code == expected_code
    assert failure.message == FAILURE_MESSAGES[expected_code]
    assert document.status == DocumentStatus.PROCESSING
    assert job.state == ProcessingJobState.CLEANUP_PENDING
    assert job.failure_code == expected_code
    assert job.failure_message == FAILURE_MESSAGES[expected_code]


def test_partial_vector_failure_is_compensated_before_failed_status() -> None:
    document_id, job_id = create_document_and_job()
    vector_store = PartialFailureVectorStore(document_id, job_id)
    processor = StoredDocumentProcessor(
        storage_provider=lambda: FakeStorage(
            (FIXTURE_DIRECTORY / "sample.txt").read_bytes()
        ),
        embedding_service=FakeEmbeddingService(),
        vector_store=vector_store,
    )

    result = run_processing_task(document_id, processor=processor)

    document, job, chunks = load_state(document_id, job_id)
    assert result is False
    assert vector_store.points == set()
    assert len(vector_store.deleted_point_ids) == 1
    assert vector_store.deleted_document_ids == [document_id]
    assert all(
        status == DocumentStatus.PROCESSING
        and job_state == ProcessingJobState.CLEANUP_PENDING
        and chunk_count > 0
        for status, job_state, chunk_count in vector_store.cleanup_state_observations
    )
    assert chunks == []
    assert document.status == DocumentStatus.FAILED
    assert document.failure_code == ProcessingFailureCode.VECTOR_PERSISTENCE_FAILED
    assert job.state == ProcessingJobState.COMPLETED
    assert job.failure_code == ProcessingFailureCode.VECTOR_PERSISTENCE_FAILED


def test_qdrant_cleanup_failure_leaves_cleanup_pending_and_chunks_intact() -> None:
    document_id, job_id = create_document_and_job()
    claim = claim_document(document_id)
    chunks = persist_test_chunks(document_id)
    failure = safe_failure(ProcessingFailureCode.EMBEDDING_FAILED)
    cleanup = begin_processing_cleanup(
        claim, failure, now=START_TIME + timedelta(seconds=1)
    )
    assert cleanup is not None

    def fail_vector_cleanup(_document_id, _point_ids) -> None:
        raise RuntimeError("http://qdrant:6333 private response")

    with pytest.raises(ProcessingOutcomeError) as error:
        complete_processing_cleanup(cleanup, delete_vectors=fail_vector_cleanup)

    document, job, persisted_chunks = load_state(document_id, job_id)
    assert error.value.code == ProcessingFailureCode.CLEANUP_FAILED
    assert str(error.value) == ProcessingFailureCode.CLEANUP_FAILED.value
    assert error.value.__cause__ is None
    assert [chunk.id for chunk in persisted_chunks] == [chunk.id for chunk in chunks]
    assert document.status == DocumentStatus.PROCESSING
    assert document.failure_code is None
    assert job.state == ProcessingJobState.CLEANUP_PENDING
    assert job.failure_code == ProcessingFailureCode.EMBEDDING_FAILED


def test_database_cleanup_failure_rolls_back_chunks_and_terminal_states() -> None:
    document_id, job_id = create_document_and_job()
    claim = claim_document(document_id)
    chunks = persist_test_chunks(document_id)
    failure = safe_failure(ProcessingFailureCode.CHUNK_PERSISTENCE_FAILED)
    cleanup = begin_processing_cleanup(
        claim, failure, now=START_TIME + timedelta(seconds=1)
    )
    assert cleanup is not None
    sessions = [SessionLocal(), SessionLocal(), SessionLocal()]

    def fail_commit() -> None:
        raise OperationalError("private database URL", {}, Exception("private"))

    sessions[-1].commit = fail_commit

    def session_factory():
        return sessions.pop(0)

    with pytest.raises(ProcessingOutcomeError) as error:
        complete_processing_cleanup(
            cleanup,
            delete_vectors=lambda _document_id, _point_ids: None,
            session_factory=session_factory,
        )

    document, job, persisted_chunks = load_state(document_id, job_id)
    assert error.value.code == ProcessingFailureCode.CLEANUP_FAILED
    assert [chunk.id for chunk in persisted_chunks] == [chunk.id for chunk in chunks]
    assert document.status == DocumentStatus.PROCESSING
    assert job.state == ProcessingJobState.CLEANUP_PENDING


def test_repeated_cleanup_is_idempotent_when_vectors_and_chunks_are_absent() -> None:
    document_id, job_id = create_document_and_job()
    claim = claim_document(document_id)
    failure = safe_failure(ProcessingFailureCode.NO_USABLE_TEXT)
    cleanup = begin_processing_cleanup(
        claim, failure, now=START_TIME + timedelta(seconds=1)
    )
    assert cleanup is not None
    cleanup_calls = []

    def record_cleanup(cleanup_document_id, point_ids) -> None:
        cleanup_calls.append((cleanup_document_id, list(point_ids)))

    first = complete_processing_cleanup(cleanup, delete_vectors=record_cleanup)
    repeated = complete_processing_cleanup(cleanup, delete_vectors=record_cleanup)

    document, job, chunks = load_state(document_id, job_id)
    assert first is True
    assert repeated is True
    assert cleanup_calls == [(document_id, [])]
    assert chunks == []
    assert document.status == DocumentStatus.FAILED
    assert job.state == ProcessingJobState.COMPLETED


def test_postgresql_chunks_are_deleted_before_failed_status_is_written() -> None:
    document_id, _ = create_document_and_job()
    claim = claim_document(document_id)
    persist_test_chunks(document_id)
    cleanup = begin_processing_cleanup(
        claim,
        safe_failure(ProcessingFailureCode.TEXT_EXTRACTION_FAILED),
        now=START_TIME + timedelta(seconds=1),
    )
    assert cleanup is not None
    statements = []

    def capture_statement(
        _connection, _cursor, statement, _parameters, _context, _executemany
    ) -> None:
        statements.append(" ".join(statement.upper().split()))

    event.listen(engine, "before_cursor_execute", capture_statement)
    try:
        completed = complete_processing_cleanup(
            cleanup, delete_vectors=lambda _document_id, _point_ids: None
        )
    finally:
        event.remove(engine, "before_cursor_execute", capture_statement)

    chunk_delete_index = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("DELETE FROM DOCUMENT_CHUNKS")
    )
    document_update_index = next(
        index
        for index, statement in enumerate(statements)
        if statement.startswith("UPDATE DOCUMENTS SET STATUS=")
    )
    assert completed is True
    assert chunk_delete_index < document_update_index


def test_stale_attempt_cannot_mark_a_newer_attempt_ready_or_failed() -> None:
    document_id, job_id = create_document_and_job()
    first_claim = claim_document(document_id)
    reclaim_time = START_TIME + timedelta(minutes=6)
    current_claim = claim_processing_job(document_id, now=reclaim_time)
    assert current_claim is not None
    persist_test_chunks(document_id)
    stale_cleanup = PendingProcessingCleanup(
        job_id=first_claim.job_id,
        document_id=first_claim.document_id,
        attempt_count=first_claim.attempt_count,
        failure=safe_failure(ProcessingFailureCode.TEXT_EXTRACTION_FAILED),
    )
    cleanup_calls = []

    ready = complete_processing_success(
        first_claim, now=reclaim_time + timedelta(seconds=1)
    )
    cleanup_started = begin_processing_cleanup(
        first_claim,
        stale_cleanup.failure,
        now=reclaim_time + timedelta(seconds=1),
    )
    failed = complete_processing_cleanup(
        stale_cleanup,
        delete_vectors=lambda *_args: cleanup_calls.append(True),
    )

    document, job, chunks = load_state(document_id, job_id)
    assert ready is False
    assert cleanup_started is None
    assert failed is False
    assert cleanup_calls == []
    assert chunks
    assert document.status == DocumentStatus.PROCESSING
    assert document.processing_attempt_count == current_claim.attempt_count
    assert job.state == ProcessingJobState.PROCESSING
    assert job.attempt_count == current_claim.attempt_count


def test_cleanup_pending_job_is_redispatched_and_resumed_by_worker_entry() -> None:
    document_id, job_id = create_document_and_job()
    claim = claim_document(document_id)
    chunks = persist_test_chunks(document_id)
    cleanup = begin_processing_cleanup(
        claim,
        safe_failure(ProcessingFailureCode.EMBEDDING_FAILED),
        now=START_TIME + timedelta(seconds=1),
    )
    assert cleanup is not None
    dispatcher = RecordingDispatcher()

    recovery = recover_processing_jobs(dispatcher, now=START_TIME + timedelta(minutes=1))
    processor = CleanupOnlyProcessor()
    resumed = run_processing_task(document_id, processor=processor)

    document, job, persisted_chunks = load_state(document_id, job_id)
    assert recovery.found == 1
    assert recovery.dispatched == 1
    assert dispatcher.calls == [(job_id, document_id)]
    assert resumed is True
    assert processor.calls == [
        (document_id, [chunk.qdrant_point_id for chunk in chunks])
    ]
    assert persisted_chunks == []
    assert document.status == DocumentStatus.FAILED
    assert job.state == ProcessingJobState.COMPLETED
