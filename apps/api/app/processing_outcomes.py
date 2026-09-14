from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from uuid import UUID

from sqlalchemy import delete, func, select, update
from sqlalchemy.exc import SQLAlchemyError

from app.database import SessionLocal
from app.document_models import (
    Document,
    DocumentChunk,
    DocumentProcessingJob,
    DocumentStatus,
    ProcessingJobState,
    utc_now,
)
from app.embeddings import EmbeddingServiceError
from app.processing import DocumentProcessingError, ProcessingErrorCategory
from app.processing_jobs import ClaimedProcessingJob, SessionFactory
from app.vector_store import VectorStoreError, VectorStoreErrorCategory


class ProcessingFailureCode(StrEnum):
    SOURCE_UNAVAILABLE = "SOURCE_UNAVAILABLE"
    UNSUPPORTED_DOCUMENT = "UNSUPPORTED_DOCUMENT"
    TEXT_EXTRACTION_FAILED = "TEXT_EXTRACTION_FAILED"
    NO_USABLE_TEXT = "NO_USABLE_TEXT"
    CHUNK_PERSISTENCE_FAILED = "CHUNK_PERSISTENCE_FAILED"
    EMBEDDING_FAILED = "EMBEDDING_FAILED"
    VECTOR_PERSISTENCE_FAILED = "VECTOR_PERSISTENCE_FAILED"
    CLEANUP_FAILED = "CLEANUP_FAILED"
    TERMINAL_UPDATE_FAILED = "TERMINAL_UPDATE_FAILED"


FAILURE_MESSAGES = {
    ProcessingFailureCode.SOURCE_UNAVAILABLE: "The source document could not be retrieved.",
    ProcessingFailureCode.UNSUPPORTED_DOCUMENT: "The source document is unsupported or invalid.",
    ProcessingFailureCode.TEXT_EXTRACTION_FAILED: "Text could not be extracted from the document.",
    ProcessingFailureCode.NO_USABLE_TEXT: "The document contains no usable text.",
    ProcessingFailureCode.CHUNK_PERSISTENCE_FAILED: "Document chunks could not be saved.",
    ProcessingFailureCode.EMBEDDING_FAILED: "Document embeddings could not be generated.",
    ProcessingFailureCode.VECTOR_PERSISTENCE_FAILED: "Document vectors could not be saved.",
    ProcessingFailureCode.CLEANUP_FAILED: "Processing cleanup could not be completed.",
    ProcessingFailureCode.TERMINAL_UPDATE_FAILED: "Processing status could not be finalized.",
}


@dataclass(frozen=True)
class SafeProcessingFailure:
    code: ProcessingFailureCode
    message: str


@dataclass(frozen=True)
class PendingProcessingCleanup:
    job_id: UUID
    document_id: UUID
    attempt_count: int
    failure: SafeProcessingFailure


class ProcessingOutcomeError(Exception):
    def __init__(self, code: ProcessingFailureCode) -> None:
        self.code = code
        super().__init__(code.value)


KnownProcessingError = DocumentProcessingError | EmbeddingServiceError | VectorStoreError
VectorCleanup = Callable[[UUID, Sequence[UUID]], None]


def map_processing_failure(error: KnownProcessingError) -> SafeProcessingFailure:
    if isinstance(error, DocumentProcessingError):
        code = {
            ProcessingErrorCategory.SOURCE_OBJECT_UNAVAILABLE: (
                ProcessingFailureCode.SOURCE_UNAVAILABLE
            ),
            ProcessingErrorCategory.UNSUPPORTED_OR_INVALID_SOURCE: (
                ProcessingFailureCode.UNSUPPORTED_DOCUMENT
            ),
            ProcessingErrorCategory.TEXT_EXTRACTION_FAILED: (
                ProcessingFailureCode.TEXT_EXTRACTION_FAILED
            ),
            ProcessingErrorCategory.NO_USABLE_TEXT: ProcessingFailureCode.NO_USABLE_TEXT,
            ProcessingErrorCategory.CHUNK_PERSISTENCE_FAILED: (
                ProcessingFailureCode.CHUNK_PERSISTENCE_FAILED
            ),
            ProcessingErrorCategory.PERSISTED_CHUNKS_UNAVAILABLE: (
                ProcessingFailureCode.CHUNK_PERSISTENCE_FAILED
            ),
        }[error.category]
    elif isinstance(error, EmbeddingServiceError):
        code = ProcessingFailureCode.EMBEDDING_FAILED
    elif error.category == VectorStoreErrorCategory.DELETE_FAILED:
        code = ProcessingFailureCode.CLEANUP_FAILED
    else:
        code = ProcessingFailureCode.VECTOR_PERSISTENCE_FAILED
    return SafeProcessingFailure(code=code, message=FAILURE_MESSAGES[code])


def complete_processing_success(
    claim: ClaimedProcessingJob,
    *,
    session_factory: SessionFactory = SessionLocal,
    now: datetime | None = None,
) -> bool:
    """Atomically make only the current, leased processing attempt Ready."""
    completion_time = now if now is not None else utc_now()
    session = session_factory()
    try:
        if _is_terminal_success(session, claim):
            session.rollback()
            return True

        chunk_count = session.scalar(
            select(func.count(DocumentChunk.id)).where(
                DocumentChunk.document_id == claim.document_id
            )
        )
        if not chunk_count:
            session.rollback()
            return False

        job_result = session.execute(
            update(DocumentProcessingJob)
            .where(
                DocumentProcessingJob.id == claim.job_id,
                DocumentProcessingJob.document_id == claim.document_id,
                DocumentProcessingJob.state == ProcessingJobState.PROCESSING,
                DocumentProcessingJob.attempt_count == claim.attempt_count,
                DocumentProcessingJob.lease_expires_at.is_not(None),
                DocumentProcessingJob.lease_expires_at > completion_time,
            )
            .values(
                state=ProcessingJobState.COMPLETED,
                completed_at=completion_time,
                lease_expires_at=None,
                failure_code=None,
                failure_message=None,
                updated_at=completion_time,
            )
        )
        if job_result.rowcount != 1:
            session.rollback()
            return _terminal_success_matches(claim, session_factory)

        document_result = session.execute(
            update(Document)
            .where(
                Document.id == claim.document_id,
                Document.status == DocumentStatus.PROCESSING,
                Document.processing_attempt_count == claim.attempt_count,
            )
            .values(
                status=DocumentStatus.READY,
                processing_completed_at=completion_time,
                failure_code=None,
                failure_message=None,
            )
        )
        if document_result.rowcount != 1:
            session.rollback()
            return False

        session.commit()
        return True
    except SQLAlchemyError:
        session.rollback()
        raise ProcessingOutcomeError(
            ProcessingFailureCode.TERMINAL_UPDATE_FAILED
        ) from None
    finally:
        session.close()


def begin_processing_cleanup(
    claim: ClaimedProcessingJob,
    failure: SafeProcessingFailure,
    *,
    session_factory: SessionFactory = SessionLocal,
    now: datetime | None = None,
) -> PendingProcessingCleanup | None:
    """Move the current leased attempt to CleanupPending before deleting data."""
    transition_time = now if now is not None else utc_now()
    session = session_factory()
    try:
        existing = _pending_cleanup_for_claim(session, claim)
        if existing is not None:
            session.rollback()
            return existing

        job_result = session.execute(
            update(DocumentProcessingJob)
            .where(
                DocumentProcessingJob.id == claim.job_id,
                DocumentProcessingJob.document_id == claim.document_id,
                DocumentProcessingJob.state == ProcessingJobState.PROCESSING,
                DocumentProcessingJob.attempt_count == claim.attempt_count,
                DocumentProcessingJob.lease_expires_at.is_not(None),
                DocumentProcessingJob.lease_expires_at > transition_time,
            )
            .values(
                state=ProcessingJobState.CLEANUP_PENDING,
                failure_code=failure.code,
                failure_message=failure.message,
                lease_expires_at=None,
                updated_at=transition_time,
            )
        )
        if job_result.rowcount != 1:
            session.rollback()
            return None

        current_document = session.scalar(
            select(Document.id).where(
                Document.id == claim.document_id,
                Document.status == DocumentStatus.PROCESSING,
                Document.processing_attempt_count == claim.attempt_count,
            )
        )
        if current_document is None:
            session.rollback()
            return None

        session.commit()
        return PendingProcessingCleanup(
            job_id=claim.job_id,
            document_id=claim.document_id,
            attempt_count=claim.attempt_count,
            failure=failure,
        )
    except SQLAlchemyError:
        session.rollback()
        raise ProcessingOutcomeError(ProcessingFailureCode.CLEANUP_FAILED) from None
    finally:
        session.close()


def get_pending_processing_cleanup(
    document_id: UUID,
    *,
    session_factory: SessionFactory = SessionLocal,
) -> PendingProcessingCleanup | None:
    session = session_factory()
    try:
        row = session.execute(
            select(
                DocumentProcessingJob.id,
                DocumentProcessingJob.document_id,
                DocumentProcessingJob.attempt_count,
                DocumentProcessingJob.failure_code,
                DocumentProcessingJob.failure_message,
            )
            .join(Document, Document.id == DocumentProcessingJob.document_id)
            .where(
                DocumentProcessingJob.document_id == document_id,
                DocumentProcessingJob.state == ProcessingJobState.CLEANUP_PENDING,
                Document.status == DocumentStatus.PROCESSING,
                Document.processing_attempt_count == DocumentProcessingJob.attempt_count,
            )
        ).one_or_none()
        if row is None:
            return None
        return PendingProcessingCleanup(
            job_id=row.id,
            document_id=row.document_id,
            attempt_count=row.attempt_count,
            failure=_stored_failure(row.failure_code, row.failure_message),
        )
    except SQLAlchemyError:
        raise ProcessingOutcomeError(ProcessingFailureCode.CLEANUP_FAILED) from None
    finally:
        session.close()


def complete_processing_cleanup(
    cleanup: PendingProcessingCleanup,
    *,
    delete_vectors: VectorCleanup,
    session_factory: SessionFactory = SessionLocal,
    now: datetime | None = None,
) -> bool:
    """Delete vectors, then atomically delete chunks and finalize Failed state."""
    if _terminal_failure_matches(cleanup, session_factory):
        return True

    point_ids = _load_cleanup_point_ids(cleanup, session_factory)
    if point_ids is None:
        return False

    try:
        delete_vectors(cleanup.document_id, point_ids)
    except Exception:
        raise ProcessingOutcomeError(ProcessingFailureCode.CLEANUP_FAILED) from None

    completion_time = now if now is not None else utc_now()
    session = session_factory()
    try:
        guard = session.execute(
            select(DocumentProcessingJob.id)
            .join(Document, Document.id == DocumentProcessingJob.document_id)
            .where(
                DocumentProcessingJob.id == cleanup.job_id,
                DocumentProcessingJob.document_id == cleanup.document_id,
                DocumentProcessingJob.state == ProcessingJobState.CLEANUP_PENDING,
                DocumentProcessingJob.attempt_count == cleanup.attempt_count,
                Document.status == DocumentStatus.PROCESSING,
                Document.processing_attempt_count == cleanup.attempt_count,
            )
            .with_for_update()
        ).one_or_none()
        if guard is None:
            session.rollback()
            return _terminal_failure_matches(cleanup, session_factory)

        session.execute(
            delete(DocumentChunk).where(DocumentChunk.document_id == cleanup.document_id)
        )
        session.flush()

        document_result = session.execute(
            update(Document)
            .where(
                Document.id == cleanup.document_id,
                Document.status == DocumentStatus.PROCESSING,
                Document.processing_attempt_count == cleanup.attempt_count,
            )
            .values(
                status=DocumentStatus.FAILED,
                processing_completed_at=completion_time,
                failure_code=cleanup.failure.code,
                failure_message=cleanup.failure.message,
            )
        )
        job_result = session.execute(
            update(DocumentProcessingJob)
            .where(
                DocumentProcessingJob.id == cleanup.job_id,
                DocumentProcessingJob.state == ProcessingJobState.CLEANUP_PENDING,
                DocumentProcessingJob.attempt_count == cleanup.attempt_count,
            )
            .values(
                state=ProcessingJobState.COMPLETED,
                completed_at=completion_time,
                lease_expires_at=None,
                failure_code=cleanup.failure.code,
                failure_message=cleanup.failure.message,
                updated_at=completion_time,
            )
        )
        if document_result.rowcount != 1 or job_result.rowcount != 1:
            session.rollback()
            return _terminal_failure_matches(cleanup, session_factory)

        session.commit()
        return True
    except SQLAlchemyError:
        session.rollback()
        raise ProcessingOutcomeError(ProcessingFailureCode.CLEANUP_FAILED) from None
    finally:
        session.close()


def _load_cleanup_point_ids(
    cleanup: PendingProcessingCleanup, session_factory: SessionFactory
) -> list[UUID] | None:
    session = session_factory()
    try:
        active_cleanup = session.scalar(
            select(DocumentProcessingJob.id)
            .join(Document, Document.id == DocumentProcessingJob.document_id)
            .where(
                DocumentProcessingJob.id == cleanup.job_id,
                DocumentProcessingJob.document_id == cleanup.document_id,
                DocumentProcessingJob.state == ProcessingJobState.CLEANUP_PENDING,
                DocumentProcessingJob.attempt_count == cleanup.attempt_count,
                Document.status == DocumentStatus.PROCESSING,
                Document.processing_attempt_count == cleanup.attempt_count,
            )
        )
        if active_cleanup is None:
            return None
        return list(
            session.scalars(
                select(DocumentChunk.qdrant_point_id)
                .where(DocumentChunk.document_id == cleanup.document_id)
                .order_by(DocumentChunk.ordinal)
            )
        )
    except SQLAlchemyError:
        raise ProcessingOutcomeError(ProcessingFailureCode.CLEANUP_FAILED) from None
    finally:
        session.close()


def _pending_cleanup_for_claim(
    session, claim: ClaimedProcessingJob
) -> PendingProcessingCleanup | None:
    row = session.execute(
        select(
            DocumentProcessingJob.id,
            DocumentProcessingJob.document_id,
            DocumentProcessingJob.attempt_count,
            DocumentProcessingJob.failure_code,
            DocumentProcessingJob.failure_message,
        )
        .join(Document, Document.id == DocumentProcessingJob.document_id)
        .where(
            DocumentProcessingJob.id == claim.job_id,
            DocumentProcessingJob.document_id == claim.document_id,
            DocumentProcessingJob.state == ProcessingJobState.CLEANUP_PENDING,
            DocumentProcessingJob.attempt_count == claim.attempt_count,
            Document.status == DocumentStatus.PROCESSING,
            Document.processing_attempt_count == claim.attempt_count,
        )
    ).one_or_none()
    if row is None:
        return None
    return PendingProcessingCleanup(
        job_id=row.id,
        document_id=row.document_id,
        attempt_count=row.attempt_count,
        failure=_stored_failure(row.failure_code, row.failure_message),
    )


def _stored_failure(code: str | None, message: str | None) -> SafeProcessingFailure:
    try:
        safe_code = ProcessingFailureCode(code)
    except (TypeError, ValueError):
        safe_code = ProcessingFailureCode.CLEANUP_FAILED
    return SafeProcessingFailure(
        code=safe_code,
        message=message or FAILURE_MESSAGES[safe_code],
    )


def _is_terminal_success(session, claim: ClaimedProcessingJob) -> bool:
    return (
        session.execute(
            select(DocumentProcessingJob.id)
            .join(Document, Document.id == DocumentProcessingJob.document_id)
            .where(
                DocumentProcessingJob.id == claim.job_id,
                DocumentProcessingJob.document_id == claim.document_id,
                DocumentProcessingJob.state == ProcessingJobState.COMPLETED,
                DocumentProcessingJob.attempt_count == claim.attempt_count,
                Document.status == DocumentStatus.READY,
                Document.processing_attempt_count == claim.attempt_count,
            )
        ).one_or_none()
        is not None
    )


def _terminal_success_matches(
    claim: ClaimedProcessingJob, session_factory: SessionFactory
) -> bool:
    session = session_factory()
    try:
        return _is_terminal_success(session, claim)
    except SQLAlchemyError:
        raise ProcessingOutcomeError(
            ProcessingFailureCode.TERMINAL_UPDATE_FAILED
        ) from None
    finally:
        session.close()


def _terminal_failure_matches(
    cleanup: PendingProcessingCleanup, session_factory: SessionFactory
) -> bool:
    session = session_factory()
    try:
        return (
            session.execute(
                select(DocumentProcessingJob.id)
                .join(Document, Document.id == DocumentProcessingJob.document_id)
                .where(
                    DocumentProcessingJob.id == cleanup.job_id,
                    DocumentProcessingJob.document_id == cleanup.document_id,
                    DocumentProcessingJob.state == ProcessingJobState.COMPLETED,
                    DocumentProcessingJob.attempt_count == cleanup.attempt_count,
                    DocumentProcessingJob.failure_code == cleanup.failure.code,
                    Document.status == DocumentStatus.FAILED,
                    Document.processing_attempt_count == cleanup.attempt_count,
                    Document.failure_code == cleanup.failure.code,
                )
            ).one_or_none()
            is not None
        )
    except SQLAlchemyError:
        raise ProcessingOutcomeError(ProcessingFailureCode.CLEANUP_FAILED) from None
    finally:
        session.close()
