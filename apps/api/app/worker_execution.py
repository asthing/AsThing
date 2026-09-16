import logging
from collections.abc import Callable
from datetime import datetime, timedelta
from threading import Event, Thread
from uuid import UUID

from sqlalchemy.exc import SQLAlchemyError

from app.database import SessionLocal
from app.document_models import utc_now
from app.embeddings import EmbeddingServiceError
from app.processing import (
    DocumentProcessingError,
    DocumentProcessingPhases,
    StoredDocumentProcessor,
)
from app.processing_jobs import (
    DEFAULT_RECOVERY_BATCH_SIZE,
    ClaimedProcessingJob,
    RecoveryResult,
    SessionFactory,
    claim_processing_job,
    get_processing_lease_duration,
    heartbeat_processing_job,
    recover_processing_jobs,
)
from app.processing_outcomes import (
    PendingProcessingCleanup,
    ProcessingFailureCode,
    ProcessingOutcomeError,
    begin_processing_cleanup,
    complete_processing_cleanup,
    complete_processing_success,
    get_pending_processing_cleanup,
    map_processing_failure,
)
from app.task_dispatch import ProcessingTaskDispatcher, get_processing_task_dispatcher
from app.vector_store import VectorStoreError

MAX_WORKER_RECOVERY_BATCH_SIZE = DEFAULT_RECOVERY_BATCH_SIZE
Clock = Callable[[], datetime]
ProcessingHook = Callable[[ClaimedProcessingJob], None]
logger = logging.getLogger(__name__)


def pending_processing_hook(_claim: ClaimedProcessingJob) -> None:
    """Placeholder until document processing is implemented in a later checkpoint."""


def run_processing_task(
    document_id: UUID,
    *,
    processor_hook: ProcessingHook | None = None,
    processor: DocumentProcessingPhases | None = None,
    session_factory: SessionFactory = SessionLocal,
    clock: Clock = utc_now,
    lease_duration: timedelta | None = None,
) -> bool:
    """Claim one document and bracket its processing hook with lease heartbeats."""
    resolved_lease_duration = lease_duration or get_processing_lease_duration()
    try:
        claim = claim_processing_job(
            document_id,
            session_factory=session_factory,
            now=clock(),
            lease_duration=resolved_lease_duration,
        )
    except SQLAlchemyError:
        logger.warning(
            "Document processing claim failed; the durable job remains recoverable.",
            extra={"document_id": str(document_id)},
        )
        return False

    if claim is None:
        return _resume_pending_cleanup(
            document_id,
            processor=processor,
            session_factory=session_factory,
            clock=clock,
        )

    resolved_processor = processor
    try:
        if processor_hook is not None:
            phase_succeeded, _ = _run_processing_phase(
                claim,
                action=lambda: processor_hook(claim),
                phase="processor_hook",
                session_factory=session_factory,
                clock=clock,
                lease_duration=resolved_lease_duration,
            )
            return phase_succeeded

        resolved_processor = resolved_processor or StoredDocumentProcessor(
            session_factory=session_factory
        )
        retrieval_succeeded, source = _run_processing_phase(
            claim,
            action=lambda: resolved_processor.retrieve(claim),
            phase="retrieval",
            session_factory=session_factory,
            clock=clock,
            lease_duration=resolved_lease_duration,
        )
        if not retrieval_succeeded or source is None:
            return False

        extraction_succeeded, chunks = _run_processing_phase(
            claim,
            action=lambda: resolved_processor.extract_and_chunk(claim, source),
            phase="extraction",
            session_factory=session_factory,
            clock=clock,
            lease_duration=resolved_lease_duration,
        )
        if not extraction_succeeded or chunks is None:
            return False

        persistence_succeeded, _ = _run_processing_phase(
            claim,
            action=lambda: resolved_processor.persist(claim, chunks),
            phase="chunk_persistence",
            session_factory=session_factory,
            clock=clock,
            lease_duration=resolved_lease_duration,
        )
        if not persistence_succeeded:
            return False

        embedding_succeeded, vectors = _run_processing_phase(
            claim,
            action=lambda: resolved_processor.embed(claim),
            phase="embedding_generation",
            session_factory=session_factory,
            clock=clock,
            lease_duration=resolved_lease_duration,
        )
        if not embedding_succeeded or vectors is None:
            return False

        upsert_succeeded, _ = _run_processing_phase(
            claim,
            action=lambda: resolved_processor.upsert_vectors(claim, vectors),
            phase="vector_upsert",
            session_factory=session_factory,
            clock=clock,
            lease_duration=resolved_lease_duration,
        )
        if not upsert_succeeded:
            return False

        try:
            completed = complete_processing_success(
                claim,
                session_factory=session_factory,
                now=clock(),
            )
        except ProcessingOutcomeError as error:
            _log_outcome_error(claim, error)
            return False
        if not completed:
            logger.warning(
                "Document processing completion was rejected for a non-current attempt.",
                extra={
                    "document_id": str(claim.document_id),
                    "processing_job_id": str(claim.job_id),
                    "processing_attempt": claim.attempt_count,
                },
            )
        return completed
    except (DocumentProcessingError, EmbeddingServiceError, VectorStoreError) as error:
        logger.warning(
            "Document processing phase stopped with a categorized error.",
            extra={
                "document_id": str(claim.document_id),
                "processing_job_id": str(claim.job_id),
                "processing_attempt": claim.attempt_count,
                "processing_error_category": error.category.value,
            },
        )
        failure = map_processing_failure(error)
        try:
            cleanup = begin_processing_cleanup(
                claim,
                failure,
                session_factory=session_factory,
                now=clock(),
            )
        except ProcessingOutcomeError as outcome_error:
            _log_outcome_error(claim, outcome_error)
            return False
        if cleanup is None:
            logger.warning(
                "Document processing cleanup was rejected for a non-current attempt.",
                extra={
                    "document_id": str(claim.document_id),
                    "processing_job_id": str(claim.job_id),
                    "processing_attempt": claim.attempt_count,
                    "processing_failure_code": failure.code.value,
                },
            )
            return False

        if resolved_processor is None:
            try:
                resolved_processor = StoredDocumentProcessor(session_factory=session_factory)
            except VectorStoreError:
                _log_outcome_error(
                    claim,
                    ProcessingOutcomeError(ProcessingFailureCode.CLEANUP_FAILED),
                )
                return False
        _complete_cleanup_safely(
            cleanup,
            processor=resolved_processor,
            session_factory=session_factory,
            clock=clock,
        )
        return False


def run_recovery_batch(
    *,
    dispatcher: ProcessingTaskDispatcher | None = None,
    session_factory: SessionFactory = SessionLocal,
    now: datetime | None = None,
    batch_size: int = DEFAULT_RECOVERY_BATCH_SIZE,
) -> RecoveryResult:
    """Run one bounded recovery pass without scheduling another process."""
    bounded_batch_size = min(max(batch_size, 0), MAX_WORKER_RECOVERY_BATCH_SIZE)
    if bounded_batch_size == 0:
        return RecoveryResult(0, 0, 0, 0)

    resolved_dispatcher = dispatcher or get_processing_task_dispatcher()
    try:
        return recover_processing_jobs(
            resolved_dispatcher,
            session_factory=session_factory,
            now=now,
            batch_size=bounded_batch_size,
        )
    except SQLAlchemyError:
        logger.warning("Document processing recovery could not read the durable job queue.")
        return RecoveryResult(0, 0, 0, 0)


def _heartbeat_claim(
    claim: ClaimedProcessingJob,
    *,
    session_factory: SessionFactory,
    now: datetime,
    lease_duration: timedelta,
    position: str,
) -> bool:
    heartbeat_succeeded = heartbeat_processing_job(
        claim.job_id,
        claim.attempt_count,
        session_factory=session_factory,
        now=now,
        lease_duration=lease_duration,
    )
    if not heartbeat_succeeded:
        logger.warning(
            "Document processing lease heartbeat was rejected.",
            extra={
                "document_id": str(claim.document_id),
                "processing_job_id": str(claim.job_id),
                "processing_attempt": claim.attempt_count,
                "heartbeat_position": position,
            },
        )
    return heartbeat_succeeded


def _run_processing_phase[PhaseResult](
    claim: ClaimedProcessingJob,
    *,
    action: Callable[[], PhaseResult],
    phase: str,
    session_factory: SessionFactory,
    clock: Clock,
    lease_duration: timedelta,
) -> tuple[bool, PhaseResult | None]:
    if not _heartbeat_claim(
        claim,
        session_factory=session_factory,
        now=clock(),
        lease_duration=lease_duration,
        position=f"before_{phase}",
    ):
        return False, None

    stop_renewal = Event()
    lease_lost = Event()

    def renew_lease() -> None:
        interval = min(60.0, lease_duration.total_seconds() / 3)
        while not stop_renewal.wait(interval):
            if not _heartbeat_claim(
                claim,
                session_factory=session_factory,
                now=clock(),
                lease_duration=lease_duration,
                position=f"during_{phase}",
            ):
                lease_lost.set()
                return

    renewal_thread = Thread(target=renew_lease, daemon=True)
    renewal_thread.start()
    try:
        result = action()
    except Exception:
        stop_renewal.set()
        renewal_thread.join()
        _heartbeat_claim(
            claim,
            session_factory=session_factory,
            now=clock(),
            lease_duration=lease_duration,
            position=f"after_{phase}",
        )
        raise

    stop_renewal.set()
    renewal_thread.join()
    if lease_lost.is_set():
        return False, None
    if not _heartbeat_claim(
        claim,
        session_factory=session_factory,
        now=clock(),
        lease_duration=lease_duration,
        position=f"after_{phase}",
    ):
        return False, None
    return True, result


def _resume_pending_cleanup(
    document_id: UUID,
    *,
    processor: DocumentProcessingPhases | None,
    session_factory: SessionFactory,
    clock: Clock,
) -> bool:
    try:
        cleanup = get_pending_processing_cleanup(
            document_id,
            session_factory=session_factory,
        )
    except ProcessingOutcomeError as error:
        logger.warning(
            "Pending document cleanup could not be loaded.",
            extra={
                "document_id": str(document_id),
                "processing_failure_code": error.code.value,
            },
        )
        return False
    if cleanup is None:
        return False

    resolved_processor = processor
    if resolved_processor is None:
        try:
            resolved_processor = StoredDocumentProcessor(session_factory=session_factory)
        except VectorStoreError:
            logger.warning(
                "Pending document cleanup could not initialize vector storage.",
                extra={
                    "document_id": str(cleanup.document_id),
                    "processing_job_id": str(cleanup.job_id),
                    "processing_attempt": cleanup.attempt_count,
                    "processing_failure_code": ProcessingFailureCode.CLEANUP_FAILED.value,
                },
            )
            return False
    return _complete_cleanup_safely(
        cleanup,
        processor=resolved_processor,
        session_factory=session_factory,
        clock=clock,
    )


def _complete_cleanup_safely(
    cleanup: PendingProcessingCleanup,
    *,
    processor: DocumentProcessingPhases,
    session_factory: SessionFactory,
    clock: Clock,
) -> bool:
    try:
        return complete_processing_cleanup(
            cleanup,
            delete_vectors=processor.delete_vectors,
            session_factory=session_factory,
            now=clock(),
        )
    except ProcessingOutcomeError as error:
        logger.warning(
            "Document processing cleanup remains recoverable.",
            extra={
                "document_id": str(cleanup.document_id),
                "processing_job_id": str(cleanup.job_id),
                "processing_attempt": cleanup.attempt_count,
                "processing_failure_code": error.code.value,
            },
        )
        return False


def _log_outcome_error(
    claim: ClaimedProcessingJob, error: ProcessingOutcomeError
) -> None:
    logger.warning(
        "Document processing state could not be finalized.",
        extra={
            "document_id": str(claim.document_id),
            "processing_job_id": str(claim.job_id),
            "processing_attempt": claim.attempt_count,
            "processing_failure_code": error.code.value,
        },
    )
