import logging
from uuid import UUID

from app.celery_app import celery_app
from app.document_deletion import (
    DocumentDeletionUnavailable,
    get_document_deletion_service,
    recover_document_deletions,
)
from app.processing_jobs import DEFAULT_RECOVERY_BATCH_SIZE
from app.worker_execution import run_processing_task, run_recovery_batch

logger = logging.getLogger(__name__)


@celery_app.task(name="documents.process")
def process_document(document_id: str) -> None:
    try:
        parsed_document_id = UUID(document_id)
    except ValueError:
        logger.warning("Document processing task received an invalid identifier.")
        return
    run_processing_task(parsed_document_id)


@celery_app.task(name="documents.recover")
def recover_document_processing_jobs(
    batch_size: int = DEFAULT_RECOVERY_BATCH_SIZE,
) -> dict[str, int]:
    result = run_recovery_batch(batch_size=batch_size)
    deletions = recover_document_deletions(batch_size=batch_size)
    return {
        "found": result.found,
        "dispatched": result.dispatched,
        "dispatch_failed": result.dispatch_failed,
        "task_ids_recorded": result.task_ids_recorded,
        "deletions_found": deletions["found"],
        "deletions_dispatched": deletions["dispatched"],
        "deletions_dispatch_failed": deletions["dispatch_failed"],
    }


@celery_app.task(name="documents.delete")
def delete_document(document_id: str) -> None:
    try:
        parsed_document_id = UUID(document_id)
    except (TypeError, ValueError):
        logger.warning("Document deletion task received an invalid identifier.")
        return
    try:
        get_document_deletion_service().resume(parsed_document_id)
    except DocumentDeletionUnavailable:
        # Durable intent is retained; the next explicit recovery pass can republish it.
        return
