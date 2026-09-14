import hashlib
import logging
import re
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from io import BytesIO
from typing import Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from docx import Document as DocxDocument
from pypdf import PdfReader
from sqlalchemy import delete, select
from sqlalchemy.exc import SQLAlchemyError

from app.database import SessionLocal
from app.document_models import Document, DocumentChunk, DocumentProcessingJob, ProcessingJobState
from app.embeddings import LocalEmbeddingService
from app.processing_jobs import ClaimedProcessingJob, SessionFactory
from app.storage import ObjectStorage, ObjectStorageError, get_storage_client
from app.vector_store import (
    ChunkVector,
    QdrantVectorStore,
    VectorStoreError,
    VectorStoreErrorCategory,
)

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
CHUNK_STEP = CHUNK_SIZE - CHUNK_OVERLAP
StorageProvider = Callable[[], ObjectStorage]
PYPDF_LOGGER = "pypdf"
PYTHON_DOCX_LOGGER = "docx"


class ProcessingErrorCategory(StrEnum):
    SOURCE_OBJECT_UNAVAILABLE = "SourceObjectUnavailable"
    UNSUPPORTED_OR_INVALID_SOURCE = "UnsupportedOrInvalidSource"
    TEXT_EXTRACTION_FAILED = "TextExtractionFailed"
    NO_USABLE_TEXT = "NoUsableText"
    CHUNK_PERSISTENCE_FAILED = "ChunkPersistenceFailed"
    PERSISTED_CHUNKS_UNAVAILABLE = "PersistedChunksUnavailable"


class DocumentProcessingError(Exception):
    def __init__(self, category: ProcessingErrorCategory) -> None:
        self.category = category
        super().__init__(category.value)


@dataclass(frozen=True)
class RetrievedDocument:
    document_id: UUID
    file_type: str
    content: bytes


@dataclass(frozen=True)
class PreparedChunk:
    id: UUID
    ordinal: int
    content: str
    content_hash: str
    qdrant_point_id: UUID


class DocumentProcessingPhases(Protocol):
    def retrieve(self, claim: ClaimedProcessingJob) -> RetrievedDocument: ...

    def extract_and_chunk(
        self, claim: ClaimedProcessingJob, source: RetrievedDocument
    ) -> list[PreparedChunk]: ...

    def persist(self, claim: ClaimedProcessingJob, chunks: list[PreparedChunk]) -> None: ...

    def embed(self, claim: ClaimedProcessingJob) -> list[ChunkVector]: ...

    def upsert_vectors(
        self, claim: ClaimedProcessingJob, vectors: list[ChunkVector]
    ) -> None: ...

    def delete_vectors(self, document_id: UUID, point_ids: Sequence[UUID]) -> None: ...


class StoredDocumentProcessor:
    def __init__(
        self,
        *,
        session_factory: SessionFactory = SessionLocal,
        storage_provider: StorageProvider = get_storage_client,
        embedding_service: LocalEmbeddingService | None = None,
        vector_store: QdrantVectorStore | None = None,
    ) -> None:
        self.session_factory = session_factory
        self.storage_provider = storage_provider
        self.embedding_service = embedding_service or LocalEmbeddingService()
        self.vector_store = vector_store or QdrantVectorStore()

    def retrieve(self, claim: ClaimedProcessingJob) -> RetrievedDocument:
        session = self.session_factory()
        try:
            source_metadata = session.execute(
                select(Document.storage_object_key, Document.file_type)
                .join(
                    DocumentProcessingJob,
                    DocumentProcessingJob.document_id == Document.id,
                )
                .where(
                    Document.id == claim.document_id,
                    DocumentProcessingJob.id == claim.job_id,
                    DocumentProcessingJob.state == ProcessingJobState.PROCESSING,
                    DocumentProcessingJob.attempt_count == claim.attempt_count,
                )
            ).one_or_none()
        except SQLAlchemyError:
            raise DocumentProcessingError(
                ProcessingErrorCategory.SOURCE_OBJECT_UNAVAILABLE
            ) from None
        finally:
            session.close()

        if source_metadata is None or source_metadata.file_type not in {"txt", "pdf", "docx"}:
            raise DocumentProcessingError(
                ProcessingErrorCategory.UNSUPPORTED_OR_INVALID_SOURCE
            )

        try:
            content = self.storage_provider().download_file(source_metadata.storage_object_key)
        except ObjectStorageError:
            raise DocumentProcessingError(
                ProcessingErrorCategory.SOURCE_OBJECT_UNAVAILABLE
            ) from None

        if not isinstance(content, bytes):
            raise DocumentProcessingError(
                ProcessingErrorCategory.UNSUPPORTED_OR_INVALID_SOURCE
            )
        return RetrievedDocument(
            document_id=claim.document_id,
            file_type=source_metadata.file_type,
            content=content,
        )

    def extract_and_chunk(
        self, claim: ClaimedProcessingJob, source: RetrievedDocument
    ) -> list[PreparedChunk]:
        if source.document_id != claim.document_id:
            raise DocumentProcessingError(
                ProcessingErrorCategory.UNSUPPORTED_OR_INVALID_SOURCE
            )
        extracted_text = extract_text(source.file_type, source.content)
        normalized_text = normalize_text(extracted_text)
        if not normalized_text:
            raise DocumentProcessingError(ProcessingErrorCategory.NO_USABLE_TEXT)
        return prepare_chunks(claim.document_id, normalized_text)

    def persist(self, claim: ClaimedProcessingJob, chunks: list[PreparedChunk]) -> None:
        if not chunks:
            raise DocumentProcessingError(ProcessingErrorCategory.NO_USABLE_TEXT)

        session = self.session_factory()
        try:
            active_job_id = session.scalar(
                select(DocumentProcessingJob.id).where(
                    DocumentProcessingJob.id == claim.job_id,
                    DocumentProcessingJob.document_id == claim.document_id,
                    DocumentProcessingJob.state == ProcessingJobState.PROCESSING,
                    DocumentProcessingJob.attempt_count == claim.attempt_count,
                )
            )
            if active_job_id is None:
                raise DocumentProcessingError(
                    ProcessingErrorCategory.CHUNK_PERSISTENCE_FAILED
                )

            session.execute(
                delete(DocumentChunk).where(DocumentChunk.document_id == claim.document_id)
            )
            session.add_all(
                [
                    DocumentChunk(
                        id=chunk.id,
                        document_id=claim.document_id,
                        ordinal=chunk.ordinal,
                        content=chunk.content,
                        content_hash=chunk.content_hash,
                        qdrant_point_id=chunk.qdrant_point_id,
                    )
                    for chunk in chunks
                ]
            )
            session.flush()
            session.commit()
        except DocumentProcessingError:
            session.rollback()
            raise
        except SQLAlchemyError:
            session.rollback()
            raise DocumentProcessingError(
                ProcessingErrorCategory.CHUNK_PERSISTENCE_FAILED
            ) from None
        finally:
            session.close()

    def embed(self, claim: ClaimedProcessingJob) -> list[ChunkVector]:
        session = self.session_factory()
        try:
            rows = session.execute(
                select(
                    DocumentChunk.id.label("chunk_id"),
                    DocumentChunk.qdrant_point_id,
                    DocumentChunk.ordinal,
                    DocumentChunk.content,
                    Document.knowledge_base_id,
                )
                .join(Document, Document.id == DocumentChunk.document_id)
                .join(
                    DocumentProcessingJob,
                    DocumentProcessingJob.document_id == Document.id,
                )
                .where(
                    Document.id == claim.document_id,
                    DocumentProcessingJob.id == claim.job_id,
                    DocumentProcessingJob.state == ProcessingJobState.PROCESSING,
                    DocumentProcessingJob.attempt_count == claim.attempt_count,
                )
                .order_by(DocumentChunk.ordinal)
            ).all()
        except SQLAlchemyError:
            raise DocumentProcessingError(
                ProcessingErrorCategory.PERSISTED_CHUNKS_UNAVAILABLE
            ) from None
        finally:
            session.close()

        if not rows:
            raise DocumentProcessingError(
                ProcessingErrorCategory.PERSISTED_CHUNKS_UNAVAILABLE
            )

        embedding_batch = self.embedding_service.embed([row.content for row in rows])
        return [
            ChunkVector(
                point_id=row.qdrant_point_id,
                vector=vector,
                chunk_id=row.chunk_id,
                document_id=claim.document_id,
                knowledge_base_id=row.knowledge_base_id,
                ordinal=row.ordinal,
                embedding_model=embedding_batch.model.identifier,
                embedding_version=embedding_batch.model.version,
            )
            for row, vector in zip(rows, embedding_batch.vectors, strict=True)
        ]

    def upsert_vectors(
        self, claim: ClaimedProcessingJob, vectors: list[ChunkVector]
    ) -> None:
        if not vectors or any(vector.document_id != claim.document_id for vector in vectors):
            raise VectorStoreError(VectorStoreErrorCategory.VECTOR_INVALID)
        self.vector_store.upsert(vectors)

    def delete_vectors(self, document_id: UUID, point_ids: Sequence[UUID]) -> None:
        # Point deletion handles known persisted chunks; the payload filter also catches a
        # partially written batch whose PostgreSQL row set was not available to this attempt.
        self.vector_store.delete_points(point_ids)
        self.vector_store.delete_document(document_id)


def extract_text(file_type: str, content: bytes) -> str:
    if file_type == "txt":
        try:
            return content.decode("utf-8-sig")
        except UnicodeDecodeError:
            raise DocumentProcessingError(
                ProcessingErrorCategory.TEXT_EXTRACTION_FAILED
            ) from None

    if file_type == "pdf":
        try:
            with suppress_parser_logs(PYPDF_LOGGER):
                reader = PdfReader(BytesIO(content))
                if reader.is_encrypted:
                    raise DocumentProcessingError(
                        ProcessingErrorCategory.UNSUPPORTED_OR_INVALID_SOURCE
                    )
                return "\n".join(page.extract_text() or "" for page in reader.pages)
        except DocumentProcessingError:
            raise
        except Exception:
            raise DocumentProcessingError(
                ProcessingErrorCategory.TEXT_EXTRACTION_FAILED
            ) from None

    if file_type == "docx":
        try:
            with suppress_parser_logs(PYTHON_DOCX_LOGGER):
                document = DocxDocument(BytesIO(content))
                return "\n".join(paragraph.text for paragraph in document.paragraphs)
        except Exception:
            raise DocumentProcessingError(
                ProcessingErrorCategory.TEXT_EXTRACTION_FAILED
            ) from None

    raise DocumentProcessingError(ProcessingErrorCategory.UNSUPPORTED_OR_INVALID_SOURCE)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def prepare_chunks(document_id: UUID, normalized_text: str) -> list[PreparedChunk]:
    chunks: list[PreparedChunk] = []
    seen_hashes: set[str] = set()
    for start in range(0, len(normalized_text), CHUNK_STEP):
        content = normalized_text[start : start + CHUNK_SIZE]
        if not content.strip():
            continue
        content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        if content_hash in seen_hashes:
            continue
        seen_hashes.add(content_hash)
        ordinal = len(chunks)
        chunks.append(
            PreparedChunk(
                id=uuid5(
                    NAMESPACE_URL,
                    f"document-chunk-row:{document_id}:{ordinal}:{content_hash}",
                ),
                ordinal=ordinal,
                content=content,
                content_hash=content_hash,
                qdrant_point_id=uuid5(
                    NAMESPACE_URL,
                    f"document-chunk-vector:{document_id}:{ordinal}:{content_hash}",
                ),
            )
        )
    if not chunks:
        raise DocumentProcessingError(ProcessingErrorCategory.NO_USABLE_TEXT)
    return chunks


@contextmanager
def suppress_parser_logs(logger_name: str):
    """Prevent third-party parsers from logging source bytes or raw parser details."""
    parser_logger = logging.getLogger(logger_name)
    previous_handlers = parser_logger.handlers
    previous_propagation = parser_logger.propagate
    parser_logger.handlers = [logging.NullHandler()]
    parser_logger.propagate = False
    try:
        yield
    finally:
        parser_logger.handlers = previous_handlers
        parser_logger.propagate = previous_propagation
