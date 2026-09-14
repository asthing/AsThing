from datetime import UTC, datetime, timedelta
from uuid import UUID, uuid4

from sqlalchemy import func, select

from app.database import SessionLocal
from app.document_models import (
    Document,
    DocumentProcessingJob,
    DocumentStatus,
    ProcessingJobState,
)
from app.knowledge_bases import KnowledgeBase
from app.processing_jobs import claim_processing_job, recover_processing_jobs


class RecordingDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, UUID]] = []
        self.fail = False

    def dispatch(self, job_id: UUID, document_id: UUID) -> str:
        self.calls.append((job_id, document_id))
        if self.fail:
            raise RuntimeError("Broker error containing private connection details")
        return str(job_id)


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def create_document_and_job(
    *,
    job_state: ProcessingJobState = ProcessingJobState.QUEUED,
    celery_task_id: str | None = None,
    lease_expires_at: datetime | None = None,
) -> tuple[UUID, UUID]:
    session = SessionLocal()
    try:
        knowledge_base = KnowledgeBase(
            name=f"Processing {uuid4()}", normalized_name=f"processing-{uuid4()}"
        )
        session.add(knowledge_base)
        session.flush()
        document = Document(
            knowledge_base_id=knowledge_base.id,
            original_filename="source.txt",
            file_size_bytes=12,
            file_type="txt",
            storage_object_key=f"documents/{uuid4()}.txt",
            status=(
                DocumentStatus.PROCESSING
                if job_state == ProcessingJobState.PROCESSING
                else DocumentStatus.UPLOADED
            ),
        )
        session.add(document)
        session.flush()
        job = DocumentProcessingJob(
            document_id=document.id,
            state=job_state,
            celery_task_id=celery_task_id,
            lease_expires_at=lease_expires_at,
        )
        session.add(job)
        session.commit()
        return document.id, job.id
    finally:
        session.close()


def load_document_and_job(
    document_id: UUID, job_id: UUID
) -> tuple[Document, DocumentProcessingJob]:
    session = SessionLocal()
    try:
        document = session.get(Document, document_id)
        job = session.get(DocumentProcessingJob, job_id)
        assert document is not None
        assert job is not None
        session.expunge(document)
        session.expunge(job)
        return document, job
    finally:
        session.close()


def test_queued_job_with_recorded_dispatch_is_redispatched_safely() -> None:
    document_id, job_id = create_document_and_job(celery_task_id="earlier-task-id")
    dispatcher = RecordingDispatcher()

    result = recover_processing_jobs(dispatcher)

    document, job = load_document_and_job(document_id, job_id)
    assert result.found == 1
    assert result.dispatched == 1
    assert dispatcher.calls == [(job_id, document_id)]
    assert document.status == DocumentStatus.UPLOADED
    assert job.state == ProcessingJobState.QUEUED
    assert job.celery_task_id == str(job_id)


def test_dispatch_failure_leaves_queued_job_recoverable(caplog) -> None:
    document_id, job_id = create_document_and_job()
    dispatcher = RecordingDispatcher()
    dispatcher.fail = True

    result = recover_processing_jobs(dispatcher)

    document, job = load_document_and_job(document_id, job_id)
    assert result.dispatch_failed == 1
    assert result.task_ids_recorded == 0
    assert document.status == DocumentStatus.UPLOADED
    assert job.state == ProcessingJobState.QUEUED
    assert job.celery_task_id is None
    assert "private connection details" not in caplog.text


def test_successful_redispatch_records_task_id() -> None:
    document_id, job_id = create_document_and_job()

    result = recover_processing_jobs(RecordingDispatcher())

    _, job = load_document_and_job(document_id, job_id)
    assert result.dispatched == 1
    assert result.task_ids_recorded == 1
    assert job.celery_task_id == str(job_id)


def test_first_claim_transitions_document_and_job_to_processing() -> None:
    document_id, job_id = create_document_and_job()
    claim_time = datetime(2026, 9, 13, 10, 0, tzinfo=UTC)

    claim = claim_processing_job(document_id, now=claim_time)

    document, job = load_document_and_job(document_id, job_id)
    assert claim is not None
    assert claim.job_id == job_id
    assert claim.attempt_count == 1
    assert claim.lease_expires_at == claim_time + timedelta(minutes=5)
    assert job.state == ProcessingJobState.PROCESSING
    assert job.attempt_count == 1
    assert as_utc(job.last_heartbeat_at) == claim_time
    assert as_utc(job.lease_expires_at) == claim_time + timedelta(minutes=5)
    assert document.status == DocumentStatus.PROCESSING
    assert document.processing_attempt_count == 1
    assert as_utc(document.processing_started_at) == claim_time


def test_active_lease_prevents_duplicate_claim() -> None:
    document_id, job_id = create_document_and_job()
    first_claim_time = datetime(2026, 9, 13, 10, 0, tzinfo=UTC)
    first_claim = claim_processing_job(document_id, now=first_claim_time)

    duplicate_claim = claim_processing_job(
        document_id,
        now=first_claim_time + timedelta(minutes=1),
    )

    _, job = load_document_and_job(document_id, job_id)
    assert first_claim is not None
    assert duplicate_claim is None
    assert job.attempt_count == 1
    assert as_utc(job.lease_expires_at) == first_claim_time + timedelta(minutes=5)


def test_expired_lease_allows_reclaim_and_increments_attempt_count() -> None:
    document_id, job_id = create_document_and_job()
    first_claim_time = datetime(2026, 9, 13, 10, 0, tzinfo=UTC)
    first_claim = claim_processing_job(document_id, now=first_claim_time)
    reclaim_time = first_claim_time + timedelta(minutes=6)

    reclaimed = claim_processing_job(document_id, now=reclaim_time)

    document, job = load_document_and_job(document_id, job_id)
    assert first_claim is not None
    assert reclaimed is not None
    assert reclaimed.attempt_count == 2
    assert job.attempt_count == 2
    assert as_utc(job.last_heartbeat_at) == reclaim_time
    assert as_utc(job.lease_expires_at) == reclaim_time + timedelta(minutes=5)
    assert document.status == DocumentStatus.PROCESSING
    assert document.processing_attempt_count == 2
    assert as_utc(document.processing_started_at) == first_claim_time


def test_recovery_dispatches_stale_processing_job_and_skips_active_lease() -> None:
    recovery_time = datetime(2026, 9, 13, 10, 0, tzinfo=UTC)
    stale_document_id, stale_job_id = create_document_and_job(
        job_state=ProcessingJobState.PROCESSING,
        lease_expires_at=recovery_time - timedelta(seconds=1),
    )
    _, active_job_id = create_document_and_job(
        job_state=ProcessingJobState.PROCESSING,
        lease_expires_at=recovery_time + timedelta(minutes=1),
    )
    dispatcher = RecordingDispatcher()

    result = recover_processing_jobs(dispatcher, now=recovery_time)

    assert result.found == 1
    assert dispatcher.calls == [(stale_job_id, stale_document_id)]
    assert active_job_id != stale_job_id


def test_recovery_can_repeat_without_creating_duplicate_jobs() -> None:
    document_id, job_id = create_document_and_job()
    dispatcher = RecordingDispatcher()

    first_result = recover_processing_jobs(dispatcher)
    second_result = recover_processing_jobs(dispatcher)

    session = SessionLocal()
    try:
        job_count = session.scalar(select(func.count()).select_from(DocumentProcessingJob))
    finally:
        session.close()
    assert first_result.found == 1
    assert second_result.found == 1
    assert job_count == 1
    assert dispatcher.calls == [(job_id, document_id), (job_id, document_id)]
