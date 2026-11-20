"""Shared lock order for worker writes and permanent deletion (no runtime clients)."""

from uuid import UUID

from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.document_models import Document, DocumentProcessingJob, DocumentStatus, ProcessingJobState


def bound_lock_wait(session: Session) -> None:
    if session.get_bind().dialect.name == "postgresql":
        session.execute(text("SET LOCAL lock_timeout = '5s'"))


def lock_document(session: Session, document_id: UUID):
    """Locks must stay held until the caller's writes/external operation have completed."""
    job = session.scalar(
        select(DocumentProcessingJob)
        .where(DocumentProcessingJob.document_id == document_id)
        .with_for_update()
        .execution_options(populate_existing=True)
    )
    document = session.scalar(
        select(Document).where(Document.id == document_id)
        .with_for_update().execution_options(populate_existing=True)
    )
    return job, document


def lock_current_processing_document(session: Session, claim, *, vector_write_token=None) -> bool:
    job, document = lock_document(session, claim.document_id)
    return bool(
        job is not None and document is not None
        and job.id == claim.job_id
        and job.state == ProcessingJobState.PROCESSING
        and job.attempt_count == claim.attempt_count
        and job.vector_write_token == vector_write_token
        and document.processing_attempt_count == claim.attempt_count
        and document.status == DocumentStatus.PROCESSING
        and document.deletion_requested_at is None
    )
