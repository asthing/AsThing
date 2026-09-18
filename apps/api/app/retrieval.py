from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict, field_validator
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.document_models import Document, DocumentChunk, DocumentStatus
from app.embeddings import EmbeddingServiceError, LocalEmbeddingService
from app.knowledge_bases import KnowledgeBase
from app.vector_store import QdrantVectorStore, VectorSearchHit, VectorStoreError

MAX_RETRIEVAL_RESULTS = 5
MIN_RETRIEVAL_SCORE = 0.35
MAX_EXCERPT_CHARACTERS = 500


class RetrievalOutcome(StrEnum):
    RESULTS = "results"
    NO_READY_DOCUMENTS = "no_ready_documents"
    NO_RESULTS = "no_results"


class SourceSearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        question = value.strip()
        if not question:
            raise ValueError("Enter a question before searching.")
        return question


class SourceResult(BaseModel):
    source_filename: str
    file_type: str
    excerpt: str


class SourceSearchResponse(BaseModel):
    outcome: RetrievalOutcome
    results: list[SourceResult]


@dataclass(frozen=True)
class RetrievedChunk:
    """Internal evidence; identifiers and full text never enter the source-search response."""

    chunk_id: UUID
    content: str
    source_filename: str
    file_type: str


@dataclass(frozen=True)
class EvidenceSearchResult:
    outcome: RetrievalOutcome
    chunks: list[RetrievedChunk]


class RetrievalDependencyError(Exception):
    """Raised when a recognized retrieval dependency cannot serve a search."""


class KnowledgeBaseNotFoundError(Exception):
    """Raised when a search targets a missing knowledge base."""


EmbeddingService = LocalEmbeddingService
VectorStore = QdrantVectorStore
SessionFactory = Callable[[], Session]


class SourceRetrievalService:
    """Find ready source chunks while retaining PostgreSQL as the display authority."""

    def __init__(
        self,
        *,
        embedding_service: EmbeddingService | None = None,
        vector_store: VectorStore | None = None,
        session_factory: SessionFactory = SessionLocal,
    ) -> None:
        self.embedding_service = embedding_service or LocalEmbeddingService()
        self.vector_store = vector_store or QdrantVectorStore()
        self.session_factory = session_factory

    def search(self, knowledge_base_id: UUID, question: str) -> SourceSearchResponse:
        evidence = self.search_evidence(knowledge_base_id, question)
        return SourceSearchResponse(
            outcome=evidence.outcome,
            results=[
                SourceResult(
                    source_filename=chunk.source_filename,
                    file_type=chunk.file_type,
                    excerpt=_excerpt(chunk.content),
                )
                for chunk in evidence.chunks
            ],
        )

    def search_evidence(self, knowledge_base_id: UUID, question: str) -> EvidenceSearchResult:
        normalized_question = question.strip()
        if not normalized_question:
            raise ValueError("Question must not be blank.")

        if not self._has_ready_documents(knowledge_base_id):
            return EvidenceSearchResult(
                outcome=RetrievalOutcome.NO_READY_DOCUMENTS,
                chunks=[],
            )

        try:
            embedding_batch = self.embedding_service.embed([normalized_question])
            hits = self.vector_store.search(
                embedding_batch.vectors[0],
                knowledge_base_id=knowledge_base_id,
                embedding_model=embedding_batch.model.identifier,
                embedding_revision=embedding_batch.model.revision,
                limit=MAX_RETRIEVAL_RESULTS,
                score_threshold=MIN_RETRIEVAL_SCORE,
            )
        except (EmbeddingServiceError, VectorStoreError):
            raise RetrievalDependencyError from None

        ordered_chunk_ids = _ordered_chunk_ids(hits)
        if not ordered_chunk_ids:
            return EvidenceSearchResult(outcome=RetrievalOutcome.NO_RESULTS, chunks=[])

        hydrated = self._hydrate_ready_chunks(knowledge_base_id, ordered_chunk_ids)
        chunks = [hydrated[chunk_id] for chunk_id in ordered_chunk_ids if chunk_id in hydrated]
        if not chunks:
            return EvidenceSearchResult(outcome=RetrievalOutcome.NO_RESULTS, chunks=[])
        return EvidenceSearchResult(outcome=RetrievalOutcome.RESULTS, chunks=chunks)

    def _has_ready_documents(self, knowledge_base_id: UUID) -> bool:
        session = self.session_factory()
        try:
            if session.get(KnowledgeBase, knowledge_base_id) is None:
                raise KnowledgeBaseNotFoundError
            statement = (
                select(Document.id)
                .where(
                    Document.knowledge_base_id == knowledge_base_id,
                    Document.status == DocumentStatus.READY,
                )
                .limit(1)
            )
            return session.scalar(statement) is not None
        except KnowledgeBaseNotFoundError:
            raise
        except SQLAlchemyError:
            raise RetrievalDependencyError from None
        finally:
            session.close()

    def _hydrate_ready_chunks(
        self, knowledge_base_id: UUID, chunk_ids: Sequence[UUID]
    ) -> dict[UUID, RetrievedChunk]:
        session = self.session_factory()
        try:
            rows = session.execute(
                select(
                    DocumentChunk.id,
                    DocumentChunk.content,
                    Document.original_filename,
                    Document.file_type,
                )
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(
                    DocumentChunk.id.in_(chunk_ids),
                    Document.knowledge_base_id == knowledge_base_id,
                    Document.status == DocumentStatus.READY,
                )
            ).all()
        except SQLAlchemyError:
            raise RetrievalDependencyError from None
        finally:
            session.close()

        return {
            row.id: RetrievedChunk(
                chunk_id=row.id,
                content=row.content,
                source_filename=row.original_filename,
                file_type=row.file_type,
            )
            for row in rows
        }


def _ordered_chunk_ids(hits: Sequence[VectorSearchHit]) -> list[UUID]:
    chunk_ids: list[UUID] = []
    seen: set[UUID] = set()
    for hit in hits:
        if hit.score < MIN_RETRIEVAL_SCORE:
            continue
        if hit.chunk_id in seen:
            continue
        seen.add(hit.chunk_id)
        chunk_ids.append(hit.chunk_id)
        if len(chunk_ids) == MAX_RETRIEVAL_RESULTS:
            break
    return chunk_ids


def _excerpt(content: str) -> str:
    if len(content) <= MAX_EXCERPT_CHARACTERS:
        return content
    return f"{content[: MAX_EXCERPT_CHARACTERS - 1].rstrip()}…"


@lru_cache
def get_source_retrieval_service() -> SourceRetrievalService:
    """Build dependencies lazily so API startup remains independent of model availability."""
    return SourceRetrievalService()


router = APIRouter(prefix="/knowledge-bases", tags=["source retrieval"])


@router.post(
    "/{knowledge_base_id}/sources/search",
    response_model=SourceSearchResponse,
)
def search_sources(
    knowledge_base_id: UUID,
    payload: SourceSearchRequest,
    retriever: SourceRetrievalService = Depends(get_source_retrieval_service),
) -> SourceSearchResponse:
    try:
        return retriever.search(knowledge_base_id, payload.question)
    except KnowledgeBaseNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found.",
        ) from None
    except RetrievalDependencyError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Source search is temporarily unavailable. Please try again.",
        ) from None
