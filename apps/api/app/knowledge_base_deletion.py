"""Delete a knowledge base only after its documents have been safely removed."""

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, SQLAlchemyError

from app.database import SessionLocal
from app.document_coordination import bound_lock_wait
from app.document_deletion import (
    DeletionDocumentNotFound,
    DeletionProcessingConflict,
    DocumentDeletionService,
    DocumentDeletionUnavailable,
)
from app.document_models import Document, DocumentStatus
from app.knowledge_bases import KnowledgeBase

PROCESSING_CONFLICT = (
    "Knowledge base deletion cannot finish while a document is processing. "
    "Wait for processing to finish and try again."
)
DELETION_INCOMPLETE = (
    "Knowledge base deletion is incomplete. Some documents may already be removed. "
    "Please try again."
)
DELETION_CHANGED = "Knowledge base changed during deletion. Please try again."


class DeletionKnowledgeBaseNotFound(Exception):
    pass


class KnowledgeBaseDeletionConflict(Exception):
    def __init__(self, message=PROCESSING_CONFLICT):
        super().__init__(message)


class KnowledgeBaseDeletionUnavailable(Exception):
    pass


class KnowledgeBaseDeletionService:
    def __init__(self, *, session_factory=SessionLocal, documents=None):
        self.session_factory = session_factory
        self.documents = documents or DocumentDeletionService(session_factory=session_factory)

    def delete(self, knowledge_base_id: UUID) -> None:
        session = self.session_factory()
        try:
            if session.get(KnowledgeBase, knowledge_base_id) is None:
                raise DeletionKnowledgeBaseNotFound
            documents = session.execute(
                select(Document.id, Document.status)
                .where(Document.knowledge_base_id == knowledge_base_id)
                .order_by(Document.id)
            ).all()
            if any(row.status == DocumentStatus.PROCESSING for row in documents):
                raise KnowledgeBaseDeletionConflict
        except SQLAlchemyError:
            raise KnowledgeBaseDeletionUnavailable from None
        finally:
            # Never hold a parent lock while child cleanup takes job/document locks.
            session.close()

        for document_id, _ in documents:
            try:
                self.documents.delete(knowledge_base_id, document_id)
            except DeletionDocumentNotFound:
                continue  # Concurrent document deletion has already completed cleanup.
            except DeletionProcessingConflict:
                raise KnowledgeBaseDeletionConflict from None
            except DocumentDeletionUnavailable:
                raise KnowledgeBaseDeletionUnavailable from None

        session = self.session_factory()
        try:
            # No child locks remain. FOR UPDATE also conflicts with FK key-share locks
            # taken by concurrent inserts; the FK is the final guard against orphan rows.
            bound_lock_wait(session)
            knowledge_base = session.scalar(
                select(KnowledgeBase).where(KnowledgeBase.id == knowledge_base_id)
                .with_for_update()
            )
            if knowledge_base is None:
                return  # Another knowledge-base deletion completed the same request.
            remaining_document = session.scalar(
                select(Document.id).where(Document.knowledge_base_id == knowledge_base_id).limit(1)
            )
            if remaining_document is not None:
                raise KnowledgeBaseDeletionConflict(DELETION_CHANGED)
            session.delete(knowledge_base)
            session.commit()
        except IntegrityError:
            session.rollback()
            raise KnowledgeBaseDeletionConflict(DELETION_CHANGED) from None
        except SQLAlchemyError:
            session.rollback()
            raise KnowledgeBaseDeletionUnavailable from None
        finally:
            session.close()
