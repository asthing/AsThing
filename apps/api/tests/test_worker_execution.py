import importlib
import sys
from datetime import UTC, datetime, timedelta
from types import ModuleType
from uuid import UUID, uuid4

from sqlalchemy import func, select, update
from sqlalchemy.exc import OperationalError

from app import worker_execution
from app.database import SessionLocal
from app.document_models import (
    Document,
    DocumentProcessingJob,
    DocumentStatus,
    ProcessingJobState,
)
from app.knowledge_bases import KnowledgeBase
from app.processing_jobs import (
    PROCESSING_LEASE_SECONDS_ENV,
    RecoveryResult,
    claim_processing_job,
    get_processing_lease_duration,
    heartbeat_processing_job,
)
from app.worker_execution import (
    MAX_WORKER_RECOVERY_BATCH_SIZE,
    run_processing_task,
    run_recovery_batch,
)


class RecordingDispatcher:
    def __init__(self) -> None:
        self.calls: list[tuple[UUID, UUID]] = []

    def dispatch(self, job_id: UUID, document_id: UUID) -> str:
        self.calls.append((job_id, document_id))
        return str(job_id)


class UnavailableSession:
    def execute(self, _statement):
        raise OperationalError(
            "database operation containing postgresql://user:secret@host",
            {},
            Exception("private database details"),
        )

    def commit(self) -> None:
        raise AssertionError("An unavailable session must not commit.")

    def rollback(self) -> None:
        pass

    def close(self) -> None:
        pass


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def create_document_and_job(
    *, job_state: ProcessingJobState = ProcessingJobState.QUEUED
) -> tuple[UUID, UUID]:
    session = SessionLocal()
    try:
        knowledge_base = KnowledgeBase(
            name=f"Worker {uuid4()}", normalized_name=f"worker-{uuid4()}"
        )
        session.add(knowledge_base)
        session.flush()
        document = Document(
            knowledge_base_id=knowledge_base.id,
            original_filename="worker-source.txt",
            file_size_bytes=20,
            file_type="txt",
            storage_object_key=f"documents/{uuid4()}.txt",
        )
        session.add(document)
        session.flush()
        job = DocumentProcessingJob(document_id=document.id, state=job_state)
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


def test_worker_task_claims_before_hook_and_heartbeats_before_and_after() -> None:
    document_id, job_id = create_document_and_job()
    claim_time = datetime(2026, 9, 13, 10, 0, tzinfo=UTC)
    before_hook_time = claim_time + timedelta(minutes=1)
    after_hook_time = claim_time + timedelta(minutes=2)
    clock_values = iter([claim_time, before_hook_time, after_hook_time])
    hook_calls = []

    def processor_hook(claim) -> None:
        document, job = load_document_and_job(document_id, job_id)
        assert claim.job_id == job_id
        assert document.status == DocumentStatus.PROCESSING
        assert job.state == ProcessingJobState.PROCESSING
        assert as_utc(job.last_heartbeat_at) == before_hook_time
        hook_calls.append(claim)

    result = run_processing_task(
        document_id,
        processor_hook=processor_hook,
        clock=lambda: next(clock_values),
        lease_duration=timedelta(minutes=5),
    )

    document, job = load_document_and_job(document_id, job_id)
    assert result is True
    assert len(hook_calls) == 1
    assert job.attempt_count == 1
    assert as_utc(job.last_heartbeat_at) == after_hook_time
    assert as_utc(job.lease_expires_at) == after_hook_time + timedelta(minutes=5)
    assert document.status == DocumentStatus.PROCESSING


def test_worker_heartbeats_around_retrieval_extraction_and_persistence(monkeypatch) -> None:
    document_id, _ = create_document_and_job()
    events = []

    class RecordingProcessor:
        def retrieve(self, _claim):
            events.append("retrieve")
            return "source"

        def extract_and_chunk(self, _claim, source):
            assert source == "source"
            events.append("extract")
            return ["chunk"]

        def persist(self, _claim, chunks) -> None:
            assert chunks == ["chunk"]
            events.append("persist")

        def embed(self, _claim):
            events.append("embed")
            return ["vector"]

        def upsert_vectors(self, _claim, vectors) -> None:
            assert vectors == ["vector"]
            events.append("upsert")

    def record_heartbeat(*_args, **_kwargs) -> bool:
        events.append("heartbeat")
        return True

    monkeypatch.setattr(worker_execution, "heartbeat_processing_job", record_heartbeat)
    def record_completion(*_args, **_kwargs) -> bool:
        events.append("complete")
        return True

    monkeypatch.setattr(worker_execution, "complete_processing_success", record_completion)

    result = run_processing_task(document_id, processor=RecordingProcessor())

    assert result is True
    assert events == [
        "heartbeat",
        "retrieve",
        "heartbeat",
        "heartbeat",
        "extract",
        "heartbeat",
        "heartbeat",
        "persist",
        "heartbeat",
        "heartbeat",
        "embed",
        "heartbeat",
        "heartbeat",
        "upsert",
        "heartbeat",
        "complete",
    ]


def test_worker_task_exits_quietly_when_active_lease_prevents_claim() -> None:
    document_id, job_id = create_document_and_job()
    claim_time = datetime(2026, 9, 13, 10, 0, tzinfo=UTC)
    first_claim = claim_processing_job(document_id, now=claim_time)
    hook_calls = []

    result = run_processing_task(
        document_id,
        processor_hook=hook_calls.append,
        clock=lambda: claim_time + timedelta(minutes=1),
    )

    document, job = load_document_and_job(document_id, job_id)
    assert first_claim is not None
    assert result is False
    assert hook_calls == []
    assert job.attempt_count == 1
    assert document.status == DocumentStatus.PROCESSING


def test_worker_task_does_not_claim_completed_job_or_call_hook() -> None:
    document_id, job_id = create_document_and_job(job_state=ProcessingJobState.COMPLETED)
    hook_calls = []

    result = run_processing_task(document_id, processor_hook=hook_calls.append)

    document, job = load_document_and_job(document_id, job_id)
    assert result is False
    assert hook_calls == []
    assert job.state == ProcessingJobState.COMPLETED
    assert job.attempt_count == 0
    assert document.status == DocumentStatus.UPLOADED


def test_heartbeat_extends_only_the_current_active_attempt() -> None:
    document_id, job_id = create_document_and_job()
    first_claim_time = datetime(2026, 9, 13, 10, 0, tzinfo=UTC)
    first_claim = claim_processing_job(document_id, now=first_claim_time)
    reclaim_time = first_claim_time + timedelta(minutes=6)
    reclaimed = claim_processing_job(document_id, now=reclaim_time)

    stale_heartbeat = heartbeat_processing_job(
        job_id,
        1,
        now=reclaim_time + timedelta(seconds=10),
    )
    current_heartbeat_time = reclaim_time + timedelta(seconds=20)
    current_heartbeat = heartbeat_processing_job(
        job_id,
        2,
        now=current_heartbeat_time,
        lease_duration=timedelta(minutes=10),
    )

    _, job = load_document_and_job(document_id, job_id)
    assert first_claim is not None
    assert reclaimed is not None
    assert stale_heartbeat is False
    assert current_heartbeat is True
    assert job.attempt_count == 2
    assert as_utc(job.last_heartbeat_at) == current_heartbeat_time
    assert as_utc(job.lease_expires_at) == current_heartbeat_time + timedelta(minutes=10)


def test_heartbeat_cannot_revive_completed_job() -> None:
    document_id, job_id = create_document_and_job()
    claim_time = datetime(2026, 9, 13, 10, 0, tzinfo=UTC)
    claim = claim_processing_job(document_id, now=claim_time)
    session = SessionLocal()
    try:
        session.execute(
            update(DocumentProcessingJob)
            .where(DocumentProcessingJob.id == job_id)
            .values(state=ProcessingJobState.COMPLETED)
        )
        session.commit()
    finally:
        session.close()

    heartbeat = heartbeat_processing_job(
        job_id,
        1,
        now=claim_time + timedelta(minutes=1),
    )

    _, job = load_document_and_job(document_id, job_id)
    assert claim is not None
    assert heartbeat is False
    assert job.state == ProcessingJobState.COMPLETED
    assert as_utc(job.last_heartbeat_at) == claim_time


def test_heartbeat_cannot_revive_an_expired_lease() -> None:
    document_id, job_id = create_document_and_job()
    claim_time = datetime(2026, 9, 13, 10, 0, tzinfo=UTC)
    claim = claim_processing_job(document_id, now=claim_time)

    heartbeat = heartbeat_processing_job(
        job_id,
        1,
        now=claim_time + timedelta(minutes=6),
    )

    _, job = load_document_and_job(document_id, job_id)
    assert claim is not None
    assert heartbeat is False
    assert as_utc(job.last_heartbeat_at) == claim_time
    assert as_utc(job.lease_expires_at) == claim_time + timedelta(minutes=5)


def test_heartbeat_database_failure_logs_no_internal_details(caplog) -> None:
    job_id = uuid4()

    heartbeat = heartbeat_processing_job(
        job_id,
        1,
        session_factory=UnavailableSession,
    )

    assert heartbeat is False
    assert caplog.records[-1].processing_job_id == str(job_id)
    assert "postgresql://" not in caplog.text
    assert "private database details" not in caplog.text


def test_processing_lease_duration_is_configurable(monkeypatch) -> None:
    monkeypatch.setenv(PROCESSING_LEASE_SECONDS_ENV, "420")

    assert get_processing_lease_duration() == timedelta(minutes=7)


def test_worker_recovery_entry_point_limits_each_batch() -> None:
    created = [create_document_and_job() for _ in range(3)]
    dispatcher = RecordingDispatcher()

    result = run_recovery_batch(dispatcher=dispatcher, batch_size=1)

    session = SessionLocal()
    try:
        job_count = session.scalar(select(func.count()).select_from(DocumentProcessingJob))
    finally:
        session.close()
    assert result.found == 1
    assert result.dispatched == 1
    assert len(dispatcher.calls) == 1
    assert job_count == 3
    assert dispatcher.calls[0] in [(job_id, document_id) for document_id, job_id in created]


def test_worker_recovery_entry_point_caps_large_batch_requests(monkeypatch) -> None:
    captured_batch_sizes = []

    def record_recovery_call(_dispatcher, **options) -> RecoveryResult:
        captured_batch_sizes.append(options["batch_size"])
        return RecoveryResult(0, 0, 0, 0)

    monkeypatch.setattr(worker_execution, "recover_processing_jobs", record_recovery_call)

    run_recovery_batch(
        dispatcher=RecordingDispatcher(),
        batch_size=MAX_WORKER_RECOVERY_BATCH_SIZE + 500,
    )

    assert captured_batch_sizes == [MAX_WORKER_RECOVERY_BATCH_SIZE]


def test_worker_recovery_database_failure_is_safe(caplog) -> None:
    result = run_recovery_batch(
        dispatcher=RecordingDispatcher(),
        session_factory=UnavailableSession,
    )

    assert result.found == 0
    assert result.dispatched == 0
    assert "postgresql://" not in caplog.text
    assert "private database details" not in caplog.text


def test_celery_task_passes_document_identifier_to_worker_runner(monkeypatch) -> None:
    class FakeCelery:
        def task(self, **_options):
            return lambda function: function

    fake_celery_module = ModuleType("app.celery_app")
    fake_celery_module.celery_app = FakeCelery()
    monkeypatch.setitem(sys.modules, "app.celery_app", fake_celery_module)
    monkeypatch.delitem(sys.modules, "app.tasks", raising=False)
    tasks_module = importlib.import_module("app.tasks")
    calls = []
    monkeypatch.setattr(tasks_module, "run_processing_task", calls.append)
    document_id = uuid4()

    tasks_module.process_document(str(document_id))

    assert calls == [document_id]
    assert not hasattr(tasks_module, "process")
