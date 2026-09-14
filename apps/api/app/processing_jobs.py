import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import UUID

from sqlalchemy import and_, func, or_, select, update
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.document_models import (
    Document,
    DocumentProcessingJob,
    DocumentStatus,
    ProcessingJobState,
    utc_now,
)
from app.task_dispatch import ProcessingTaskDispatcher

DEFAULT_PROCESSING_LEASE_DURATION = timedelta(minutes=5)
DEFAULT_LEASE_DURATION = DEFAULT_PROCESSING_LEASE_DURATION
DEFAULT_RECOVERY_BATCH_SIZE = 100
PROCESSING_LEASE_SECONDS_ENV = "PROCESSING_LEASE_SECONDS"
SessionFactory = Callable[[], Session]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ClaimedProcessingJob:
    job_id: UUID
    document_id: UUID
    attempt_count: int
    lease_expires_at: datetime


@dataclass(frozen=True)
class RecoveryResult:
    found: int
    dispatched: int
    dispatch_failed: int
    task_ids_recorded: int


def get_processing_lease_duration() -> timedelta:
    configured_seconds = os.getenv(PROCESSING_LEASE_SECONDS_ENV)
    if configured_seconds is None:
        return DEFAULT_PROCESSING_LEASE_DURATION

    try:
        lease_seconds = int(configured_seconds)
    except ValueError:
        lease_seconds = 0
    if lease_seconds <= 0:
        logger.warning("Invalid processing lease duration; using the safe default.")
        return DEFAULT_PROCESSING_LEASE_DURATION
    return timedelta(seconds=lease_seconds)


def recover_processing_jobs(
    dispatcher: ProcessingTaskDispatcher,
    *,
    session_factory: SessionFactory = SessionLocal,
    now: datetime | None = None,
    batch_size: int = DEFAULT_RECOVERY_BATCH_SIZE,
) -> RecoveryResult:
    """Publish recoverable jobs after closing the database transaction used to find them."""
    recovery_time = now if now is not None else utc_now()
    session = session_factory()
    try:
        statement = (
            select(DocumentProcessingJob.id, DocumentProcessingJob.document_id)
            .where(
                or_(
                    DocumentProcessingJob.state == ProcessingJobState.QUEUED,
                    DocumentProcessingJob.state == ProcessingJobState.CLEANUP_PENDING,
                    and_(
                        DocumentProcessingJob.state == ProcessingJobState.PROCESSING,
                        or_(
                            DocumentProcessingJob.lease_expires_at.is_(None),
                            DocumentProcessingJob.lease_expires_at <= recovery_time,
                        ),
                    ),
                )
            )
            .order_by(DocumentProcessingJob.queued_at, DocumentProcessingJob.id)
            .limit(batch_size)
        )
        candidates = list(session.execute(statement))
        session.commit()
    finally:
        session.close()

    dispatched = 0
    dispatch_failed = 0
    task_ids_recorded = 0
    for job_id, document_id in candidates:
        try:
            celery_task_id = dispatcher.dispatch(job_id, document_id)
        except Exception:
            dispatch_failed += 1
            logger.warning(
                "Document processing recovery dispatch failed; the job remains recoverable.",
                extra={
                    "document_id": str(document_id),
                    "processing_job_id": str(job_id),
                },
            )
            continue

        dispatched += 1
        if _record_dispatched_task_id(
            job_id,
            celery_task_id,
            session_factory=session_factory,
            updated_at=recovery_time,
        ):
            task_ids_recorded += 1

    return RecoveryResult(
        found=len(candidates),
        dispatched=dispatched,
        dispatch_failed=dispatch_failed,
        task_ids_recorded=task_ids_recorded,
    )


def claim_processing_job(
    document_id: UUID,
    *,
    session_factory: SessionFactory = SessionLocal,
    now: datetime | None = None,
    lease_duration: timedelta = DEFAULT_LEASE_DURATION,
) -> ClaimedProcessingJob | None:
    """Atomically claim a queued or stale job for one document."""
    claim_time = now if now is not None else utc_now()
    lease_expires_at = claim_time + lease_duration
    session = session_factory()
    try:
        eligible_job = or_(
            DocumentProcessingJob.state == ProcessingJobState.QUEUED,
            and_(
                DocumentProcessingJob.state == ProcessingJobState.PROCESSING,
                or_(
                    DocumentProcessingJob.lease_expires_at.is_(None),
                    DocumentProcessingJob.lease_expires_at <= claim_time,
                ),
            ),
        )
        claim_statement = (
            update(DocumentProcessingJob)
            .where(DocumentProcessingJob.document_id == document_id, eligible_job)
            .values(
                state=ProcessingJobState.PROCESSING,
                attempt_count=DocumentProcessingJob.attempt_count + 1,
                lease_expires_at=lease_expires_at,
                last_heartbeat_at=claim_time,
                started_at=func.coalesce(DocumentProcessingJob.started_at, claim_time),
                completed_at=None,
                failure_code=None,
                failure_message=None,
                updated_at=claim_time,
            )
            .returning(
                DocumentProcessingJob.id,
                DocumentProcessingJob.document_id,
                DocumentProcessingJob.attempt_count,
            )
        )
        claimed = session.execute(claim_statement).one_or_none()
        if claimed is None:
            session.rollback()
            return None

        document_update = session.execute(
            update(Document)
            .where(
                Document.id == claimed.document_id,
                Document.status.in_([DocumentStatus.UPLOADED, DocumentStatus.PROCESSING]),
            )
            .values(
                status=DocumentStatus.PROCESSING,
                processing_attempt_count=claimed.attempt_count,
                processing_started_at=func.coalesce(Document.processing_started_at, claim_time),
                processing_completed_at=None,
                failure_code=None,
                failure_message=None,
            )
        )
        if document_update.rowcount != 1:
            session.rollback()
            return None

        session.commit()
        return ClaimedProcessingJob(
            job_id=claimed.id,
            document_id=claimed.document_id,
            attempt_count=claimed.attempt_count,
            lease_expires_at=lease_expires_at,
        )
    except SQLAlchemyError:
        session.rollback()
        raise
    finally:
        session.close()


def heartbeat_processing_job(
    job_id: UUID,
    attempt_count: int,
    *,
    session_factory: SessionFactory = SessionLocal,
    now: datetime | None = None,
    lease_duration: timedelta = DEFAULT_PROCESSING_LEASE_DURATION,
) -> bool:
    """Extend the lease only for the active attempt that still owns it."""
    heartbeat_time = now if now is not None else utc_now()
    lease_expires_at = heartbeat_time + lease_duration
    session = session_factory()
    try:
        result = session.execute(
            update(DocumentProcessingJob)
            .where(
                DocumentProcessingJob.id == job_id,
                DocumentProcessingJob.state == ProcessingJobState.PROCESSING,
                DocumentProcessingJob.attempt_count == attempt_count,
                DocumentProcessingJob.lease_expires_at.is_not(None),
                DocumentProcessingJob.lease_expires_at > heartbeat_time,
            )
            .values(
                last_heartbeat_at=heartbeat_time,
                lease_expires_at=lease_expires_at,
                updated_at=heartbeat_time,
            )
        )
        if result.rowcount != 1:
            session.rollback()
            return False
        session.commit()
        return True
    except SQLAlchemyError:
        session.rollback()
        logger.warning(
            "Document processing heartbeat failed; the active lease was not extended.",
            extra={
                "processing_job_id": str(job_id),
                "processing_attempt": attempt_count,
            },
        )
        return False
    finally:
        session.close()


def _record_dispatched_task_id(
    job_id: UUID,
    celery_task_id: str,
    *,
    session_factory: SessionFactory,
    updated_at: datetime,
) -> bool:
    session = session_factory()
    try:
        result = session.execute(
            update(DocumentProcessingJob)
            .where(DocumentProcessingJob.id == job_id)
            .values(celery_task_id=celery_task_id, updated_at=updated_at)
        )
        if result.rowcount != 1:
            session.rollback()
            return False
        session.commit()
        return True
    except SQLAlchemyError:
        session.rollback()
        logger.warning(
            "Recovered processing task ID could not be recorded; the job remains recoverable.",
            extra={"processing_job_id": str(job_id)},
        )
        return False
    finally:
        session.close()
