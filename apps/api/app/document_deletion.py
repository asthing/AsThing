"""Durable, synchronous permanent deletion and bounded worker recovery."""

import logging
from uuid import UUID

from sqlalchemy import delete, select
from sqlalchemy.exc import InterfaceError, OperationalError, SQLAlchemyError
from sqlalchemy.exc import TimeoutError as DatabaseTimeoutError

from app.database import SessionLocal
from app.document_coordination import bound_lock_wait, lock_document
from app.document_models import (
    Document,
    DocumentChunk,
    DocumentProcessingJob,
    DocumentStatus,
    utc_now,
)

# Worker-only imports must register the target of documents.knowledge_base_id before ORM deletion.
from app.knowledge_bases import KnowledgeBase  # noqa: F401
from app.storage import ObjectStorageError, get_storage_client
from app.vector_store import QdrantVectorStore, VectorStoreError

logger = logging.getLogger(__name__)
DELETION_UNAVAILABLE = "Document deletion is unavailable. Please try again."
DELETION_INCOMPLETE = (
    "Deletion is incomplete. The document is excluded from searches. Retry deletion."
)
PROCESSING_CONFLICT = (
    "This document is processing and cannot be deleted. Wait for processing to finish."
)


class DeletionDocumentNotFound(Exception):
    pass


class DeletionProcessingConflict(Exception):
    pass


class DocumentDeletionUnavailable(Exception):
    def __init__(self, *, pending: bool):
        self.pending = pending
        super().__init__(DELETION_INCOMPLETE if pending else DELETION_UNAVAILABLE)


class DocumentDeletionService:
    def __init__(
        self, *, session_factory=SessionLocal, storage_provider=get_storage_client,
        vector_store_provider=QdrantVectorStore,
    ):
        self.session_factory = session_factory
        self.storage_provider = storage_provider
        self.vector_store_provider = vector_store_provider

    def delete(self, knowledge_base_id: UUID, document_id: UUID) -> None:
        session = self.session_factory()
        try:
            bound_lock_wait(session)
            _, document = lock_document(session, document_id)
            if document is None or document.knowledge_base_id != knowledge_base_id:
                raise DeletionDocumentNotFound
            if document.status == DocumentStatus.PROCESSING:
                raise DeletionProcessingConflict
            if document.deletion_requested_at is None:
                document.deletion_requested_at = utc_now()
            session.commit()
        except (InterfaceError, OperationalError, DatabaseTimeoutError):
            session.rollback()
            raise DocumentDeletionUnavailable(pending=False) from None
        finally:
            session.close()
        if not self.resume(document_id, knowledge_base_id=knowledge_base_id):
            raise DocumentDeletionUnavailable(pending=False)

    def resume(self, document_id: UUID, *, knowledge_base_id: UUID | None = None) -> bool:
        session = self.session_factory()
        try:
            bound_lock_wait(session)
            job, document = lock_document(session, document_id)
            if document is None:
                return True  # Another deletion already completed; never use cached external keys.
            if knowledge_base_id is not None and document.knowledge_base_id != knowledge_base_id:
                raise DeletionDocumentNotFound
            if document.deletion_requested_at is None:
                return False  # Recovery must never initiate an unrequested deletion.

            if job is not None and job.vector_write_token is not None:
                # Even an acknowledged delete cannot fence an earlier unacknowledged
                # upsert. Keep the identifiers and PostgreSQL retrieval exclusion intact.
                raise DocumentDeletionUnavailable(pending=True)

            # Keep job/document locks across acknowledged external cleanup AND the final commit.
            # All worker chunk/vector writes take these same locks and recheck current ownership.
            self.vector_store_provider().delete_document(document.id)
            self.storage_provider().delete_file_strict(document.storage_object_key)
            session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document.id))
            session.execute(
                delete(DocumentProcessingJob)
                .where(DocumentProcessingJob.document_id == document.id)
            )
            session.delete(document)
            session.commit()
            return True
        except (
            ObjectStorageError, VectorStoreError,
            InterfaceError, OperationalError, DatabaseTimeoutError,
        ):
            session.rollback()
            logger.warning(
                "Document deletion is incomplete and remains recoverable.",
                extra={"document_id": str(document_id)},
            )
            raise DocumentDeletionUnavailable(pending=True) from None
        finally:
            session.close()


def get_document_deletion_service() -> DocumentDeletionService:
    return DocumentDeletionService()


def recover_document_deletions(*, dispatcher=None, session_factory=SessionLocal, batch_size=100):
    from app.task_dispatch import CeleryDocumentDeletionDispatcher

    batch_size = min(max(batch_size, 0), 100)
    if not batch_size:
        return {"found": 0, "dispatched": 0, "dispatch_failed": 0}
    session = session_factory()
    try:
        ids = list(session.scalars(
            select(Document.id).where(Document.deletion_requested_at.is_not(None))
            .order_by(Document.deletion_requested_at, Document.id).limit(batch_size)
        ))
        session.commit()
    except SQLAlchemyError:
        logger.warning("Document deletion recovery is temporarily unavailable.")
        return {"found": 0, "dispatched": 0, "dispatch_failed": 0}
    finally:
        session.close()
    dispatcher = dispatcher or CeleryDocumentDeletionDispatcher()
    dispatched = 0
    for document_id in ids:
        try:
            dispatcher.dispatch(document_id)  # Always outside the discovery transaction.
            dispatched += 1
        except Exception:
            logger.warning(
                "Document deletion recovery dispatch failed; intent remains durable.",
                extra={"document_id": str(document_id)},
            )
    return {"found": len(ids), "dispatched": dispatched, "dispatch_failed": len(ids) - dispatched}
