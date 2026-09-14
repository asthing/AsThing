import logging
from datetime import UTC, datetime
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import BinaryIO
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from pydantic import BaseModel, ConfigDict, field_serializer
from sqlalchemy import desc, select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import get_session
from app.document_models import (
    Document,
    DocumentProcessingJob,
    DocumentStatus,
    ProcessingJobState,
)
from app.knowledge_bases import KnowledgeBase, KnowledgeBaseResponse
from app.storage import ObjectStorageError, get_storage_client
from app.task_dispatch import ProcessingTaskDispatcher, get_processing_task_dispatcher

MAX_FILE_SIZE_BYTES = 25 * 1024 * 1024
READ_CHUNK_SIZE_BYTES = 1024 * 1024
SUPPORTED_FILE_TYPES = {
    ".pdf": ("pdf", "application/pdf"),
    ".txt": ("txt", "text/plain"),
    ".docx": (
        "docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ),
}
UPLOADED_STATUS = DocumentStatus.UPLOADED
logger = logging.getLogger(__name__)


class DocumentResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    knowledge_base_id: UUID
    original_filename: str
    file_size_bytes: int
    file_type: str
    status: DocumentStatus
    uploaded_at: datetime
    processing_started_at: datetime | None = None
    processing_completed_at: datetime | None = None
    failure_code: str | None = None
    failure_message: str | None = None

    @field_serializer("uploaded_at")
    def serialize_uploaded_at(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


router = APIRouter(prefix="/knowledge-bases", tags=["documents"])


def get_knowledge_base_or_404(knowledge_base_id: UUID, session: Session) -> KnowledgeBase:
    try:
        knowledge_base = session.get(KnowledgeBase, knowledge_base_id)
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Knowledge base storage is unavailable.",
        ) from None

    if knowledge_base is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found.",
        )
    return knowledge_base


@router.get("/{knowledge_base_id}", response_model=KnowledgeBaseResponse)
def get_knowledge_base(
    knowledge_base_id: UUID, session: Session = Depends(get_session)
) -> KnowledgeBase:
    return get_knowledge_base_or_404(knowledge_base_id, session)


@router.get("/{knowledge_base_id}/documents", response_model=list[DocumentResponse])
def list_documents(
    knowledge_base_id: UUID, session: Session = Depends(get_session)
) -> list[Document]:
    get_knowledge_base_or_404(knowledge_base_id, session)

    try:
        statement = (
            select(Document)
            .where(Document.knowledge_base_id == knowledge_base_id)
            .order_by(desc(Document.uploaded_at), desc(Document.id))
        )
        return list(session.scalars(statement))
    except SQLAlchemyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document metadata is unavailable. Please try again.",
        ) from None


@router.post(
    "/{knowledge_base_id}/documents",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    knowledge_base_id: UUID,
    files: list[UploadFile] = File(..., alias="file"),
    session: Session = Depends(get_session),
    dispatcher: ProcessingTaskDispatcher = Depends(get_processing_task_dispatcher),
) -> DocumentResponse:
    knowledge_base = get_knowledge_base_or_404(knowledge_base_id, session)
    if len(files) != 1:
        for uploaded_file in files:
            await uploaded_file.close()
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Upload exactly one file at a time.",
        )

    file = files[0]
    original_filename, file_type, content_type = validate_file_name(file)
    staged_file, file_size_bytes = await stage_file(file)
    object_key = f"knowledge-bases/{knowledge_base.id}/documents/{uuid4()}.{file_type}"

    try:
        get_storage_client().upload_file(staged_file, object_key, content_type)
    except ObjectStorageError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document storage is unavailable. Please try again.",
        ) from None
    finally:
        staged_file.close()
        await file.close()

    document_id = uuid4()
    processing_job_id = uuid4()
    document = Document(
        id=document_id,
        knowledge_base_id=knowledge_base.id,
        original_filename=original_filename,
        file_size_bytes=file_size_bytes,
        file_type=file_type,
        status=UPLOADED_STATUS,
        storage_object_key=object_key,
    )
    processing_job = DocumentProcessingJob(
        id=processing_job_id,
        document_id=document_id,
        state=ProcessingJobState.QUEUED,
    )
    session.add_all([document, processing_job])
    try:
        session.flush()
        response = DocumentResponse.model_validate(document)
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        get_storage_client().delete_file(object_key)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Document metadata could not be saved. Please try again.",
        ) from None

    try:
        celery_task_id = dispatcher.dispatch(processing_job_id, document_id)
    except Exception:
        logger.warning(
            "Document processing dispatch failed; the queued job remains recoverable.",
            extra={
                "document_id": str(document_id),
                "processing_job_id": str(processing_job_id),
            },
        )
        return response

    try:
        processing_job.celery_task_id = celery_task_id
        session.commit()
    except SQLAlchemyError:
        session.rollback()
        logger.warning(
            "Document processing task ID could not be recorded; "
            "the queued job remains recoverable.",
            extra={
                "document_id": str(document_id),
                "processing_job_id": str(processing_job_id),
            },
        )

    return response


def validate_file_name(file: UploadFile) -> tuple[str, str, str]:
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail="Choose a file to upload.",
        )

    extension = Path(file.filename).suffix.lower()
    if extension not in SUPPORTED_FILE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported file type. Upload a PDF, TXT, or DOCX file.",
        )

    file_type, content_type = SUPPORTED_FILE_TYPES[extension]
    return file.filename, file_type, content_type


async def stage_file(file: UploadFile) -> tuple[BinaryIO, int]:
    staged_file = SpooledTemporaryFile(max_size=READ_CHUNK_SIZE_BYTES, mode="w+b")
    file_size_bytes = 0

    try:
        while chunk := await file.read(READ_CHUNK_SIZE_BYTES):
            file_size_bytes += len(chunk)
            if file_size_bytes > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                    detail="File is too large. The maximum size is 25 MB.",
                )
            staged_file.write(chunk)
        staged_file.seek(0)
        return staged_file, file_size_bytes
    except Exception:
        staged_file.close()
        await file.close()
        raise
