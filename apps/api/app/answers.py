"""One-shot, evidence-grounded answers for a single knowledge base."""

from enum import StrEnum
from functools import lru_cache
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.local_llm import LocalAnswerModel, LocalModelError, LocalModelErrorCategory, PromptSource
from app.retrieval import (
    KnowledgeBaseNotFoundError,
    RetrievalDependencyError,
    RetrievalOutcome,
    SourceRetrievalService,
    SourceSearchRequest,
    _excerpt,
    get_source_retrieval_service,
)


class AnswerOutcome(StrEnum):
    ANSWER = "answer"
    NO_ANSWER = "no_answer"


class NoAnswerReason(StrEnum):
    NO_READY_DOCUMENTS = "no_ready_documents"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class AnswerReference(BaseModel):
    reference_key: str
    source_filename: str
    file_type: str
    excerpt: str


class AnswerResponse(BaseModel):
    outcome: AnswerOutcome
    answer: str | None
    reason: NoAnswerReason | None
    message: str | None
    references: list[AnswerReference]


NO_READY_MESSAGE = "Documents must finish processing before they can answer questions."
NO_EVIDENCE_MESSAGE = (
    "The available sources do not contain enough information to answer this question."
)


class AnswerService:
    def __init__(
        self,
        *,
        retriever: SourceRetrievalService | None = None,
        model: LocalAnswerModel | None = None,
    ) -> None:
        self.retriever = retriever or get_source_retrieval_service()
        self.model = model or LocalAnswerModel()

    def answer(self, knowledge_base_id: UUID, question: str) -> AnswerResponse:
        evidence = self.retriever.search_evidence(knowledge_base_id, question.strip())
        if evidence.outcome != RetrievalOutcome.RESULTS:
            reason = (
                NoAnswerReason.NO_READY_DOCUMENTS
                if evidence.outcome == RetrievalOutcome.NO_READY_DOCUMENTS
                else NoAnswerReason.INSUFFICIENT_EVIDENCE
            )
            return self._no_answer(reason)

        by_key = {
            f"source-{index}": chunk
            for index, chunk in enumerate(evidence.chunks, start=1)
        }
        decision = self.model.generate(
            question.strip(),
            [PromptSource(key=key, content=chunk.content) for key, chunk in by_key.items()],
        )
        if decision.outcome == "no_answer":
            return self._no_answer(NoAnswerReason.INSUFFICIENT_EVIDENCE)
        if not set(decision.references).issubset(by_key):
            raise LocalModelError(LocalModelErrorCategory.INVALID_OUTPUT)

        return AnswerResponse(
            outcome=AnswerOutcome.ANSWER,
            answer=decision.answer,
            reason=None,
            message=None,
            references=[
                AnswerReference(
                    reference_key=key,
                    source_filename=by_key[key].source_filename,
                    file_type=by_key[key].file_type,
                    excerpt=_excerpt(by_key[key].content),
                )
                for key in decision.references
            ],
        )

    @staticmethod
    def _no_answer(reason: NoAnswerReason) -> AnswerResponse:
        return AnswerResponse(
            outcome=AnswerOutcome.NO_ANSWER,
            answer=None,
            reason=reason,
            message=(
                NO_READY_MESSAGE
                if reason == NoAnswerReason.NO_READY_DOCUMENTS
                else NO_EVIDENCE_MESSAGE
            ),
            references=[],
        )


@lru_cache
def get_answer_service() -> AnswerService:
    return AnswerService()


router = APIRouter(prefix="/knowledge-bases", tags=["answers"])


@router.post("/{knowledge_base_id}/answers", response_model=AnswerResponse)
def answer_question(
    knowledge_base_id: UUID,
    payload: SourceSearchRequest,
    service: AnswerService = Depends(get_answer_service),
) -> AnswerResponse:
    try:
        return service.answer(knowledge_base_id, payload.question)
    except KnowledgeBaseNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Knowledge base not found.",
        ) from None
    except (RetrievalDependencyError, LocalModelError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Answer generation is temporarily unavailable. Please try again.",
        ) from None
