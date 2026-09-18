from dataclasses import dataclass
from uuid import UUID, uuid4

from app.database import SessionLocal
from app.document_models import Document, DocumentChunk, DocumentStatus
from app.embeddings import EMBEDDING_DIMENSION, EmbeddingBatch, EmbeddingModelMetadata
from app.knowledge_bases import KnowledgeBase
from app.main import app
from app.retrieval import (
    MAX_RETRIEVAL_RESULTS,
    RetrievalDependencyError,
    RetrievalOutcome,
    SourceRetrievalService,
    VectorSearchHit,
    get_source_retrieval_service,
)
from app.vector_store import VectorStoreError, VectorStoreErrorCategory


@dataclass
class FakeEmbeddingService:
    vector: list[float] | None = None
    error: Exception | None = None

    def __post_init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> EmbeddingBatch:
        self.calls.append(texts)
        if self.error is not None:
            raise self.error
        return EmbeddingBatch(
            vectors=[self.vector or [0.1] * EMBEDDING_DIMENSION],
            model=EmbeddingModelMetadata(
                identifier="BAAI/bge-m3",
                version="5617a9f61b028005a4858fdac845db406aefb181",
                dimension=EMBEDDING_DIMENSION,
            ),
        )


@dataclass
class FakeVectorStore:
    hits: list[VectorSearchHit]
    error: Exception | None = None

    def __post_init__(self) -> None:
        self.calls: list[tuple[list[float], UUID, str, str, int, float]] = []

    def search(
        self,
        vector: list[float],
        *,
        knowledge_base_id: UUID,
        embedding_model: str,
        embedding_revision: str,
        limit: int,
        score_threshold: float,
    ) -> list[VectorSearchHit]:
        self.calls.append(
            (
                vector,
                knowledge_base_id,
                embedding_model,
                embedding_revision,
                limit,
                score_threshold,
            )
        )
        if self.error is not None:
            raise self.error
        return self.hits


def create_knowledge_base(name: str) -> KnowledgeBase:
    knowledge_base = KnowledgeBase(name=name, normalized_name=name.casefold())
    session = SessionLocal()
    try:
        session.add(knowledge_base)
        session.commit()
        session.refresh(knowledge_base)
        return knowledge_base
    finally:
        session.close()


def create_chunk(
    knowledge_base_id: UUID,
    *,
    filename: str,
    content: str,
    status: DocumentStatus = DocumentStatus.READY,
    ordinal: int = 0,
) -> DocumentChunk:
    document = Document(
        knowledge_base_id=knowledge_base_id,
        original_filename=filename,
        file_size_bytes=100,
        file_type="txt",
        status=status,
        storage_object_key=f"documents/{uuid4()}",
    )
    chunk = DocumentChunk(
        document_id=document.id,
        ordinal=ordinal,
        content=content,
        content_hash=uuid4().hex,
        qdrant_point_id=uuid4(),
    )
    session = SessionLocal()
    try:
        session.add(document)
        session.flush()
        chunk.document_id = document.id
        session.add(chunk)
        session.commit()
        session.refresh(chunk)
        return chunk
    finally:
        session.close()


def test_retrieval_returns_postgres_metadata_in_qdrant_relevance_order() -> None:
    knowledge_base = create_knowledge_base("Engineering")
    less_relevant = create_chunk(
        knowledge_base.id,
        filename="handbook.txt",
        content="Annual leave is available after probation.",
    )
    more_relevant = create_chunk(
        knowledge_base.id,
        filename="policy.txt",
        content="Employees request annual leave through the HR portal.",
    )
    embeddings = FakeEmbeddingService()
    vectors = FakeVectorStore(
        [
            VectorSearchHit(chunk_id=more_relevant.id, score=0.91),
            VectorSearchHit(chunk_id=less_relevant.id, score=0.78),
        ]
    )

    response = SourceRetrievalService(
        embedding_service=embeddings,
        vector_store=vectors,
    ).search(knowledge_base.id, "  How do I request annual leave?  ")

    assert response.outcome == RetrievalOutcome.RESULTS
    assert [result.model_dump() for result in response.results] == [
        {
            "source_filename": "policy.txt",
            "file_type": "txt",
            "excerpt": "Employees request annual leave through the HR portal.",
        },
        {
            "source_filename": "handbook.txt",
            "file_type": "txt",
            "excerpt": "Annual leave is available after probation.",
        },
    ]
    assert embeddings.calls == [["How do I request annual leave?"]]
    assert vectors.calls[0][1:] == (
        knowledge_base.id,
        "BAAI/bge-m3",
        "5617a9f61b028005a4858fdac845db406aefb181",
        5,
        0.35,
    )


def test_retrieval_hydration_excludes_another_knowledge_base_and_nonready_documents() -> None:
    selected = create_knowledge_base("Selected")
    other = create_knowledge_base("Other")
    selected_ready = create_chunk(
        selected.id,
        filename="selected.txt",
        content="Selected knowledge base evidence.",
    )
    other_ready = create_chunk(
        other.id,
        filename="other.txt",
        content="Other knowledge base evidence.",
    )
    selected_uploaded = create_chunk(
        selected.id,
        filename="not-ready.txt",
        content="This must not be searchable.",
        status=DocumentStatus.UPLOADED,
    )
    vectors = FakeVectorStore(
        [
            VectorSearchHit(chunk_id=other_ready.id, score=0.99),
            VectorSearchHit(chunk_id=selected_uploaded.id, score=0.95),
            VectorSearchHit(chunk_id=selected_ready.id, score=0.70),
        ]
    )

    response = SourceRetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_store=vectors,
    ).search(selected.id, "Where is the evidence?")

    assert response.outcome == RetrievalOutcome.RESULTS
    assert [result.model_dump() for result in response.results] == [
        {
            "source_filename": "selected.txt",
            "file_type": "txt",
            "excerpt": "Selected knowledge base evidence.",
        }
    ]


def test_retrieval_keeps_only_the_first_five_relevance_ordered_hits() -> None:
    knowledge_base = create_knowledge_base("Five results")
    chunks = [
        create_chunk(
            knowledge_base.id,
            filename=f"source-{index}.txt",
            content=f"Source excerpt {index}.",
            ordinal=index,
        )
        for index in range(6)
    ]
    vectors = FakeVectorStore([
        VectorSearchHit(chunk_id=chunk.id, score=1.0 - index / 10)
        for index, chunk in enumerate(chunks)
    ])

    response = SourceRetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_store=vectors,
    ).search(knowledge_base.id, "Show relevant evidence")

    assert len(response.results) == MAX_RETRIEVAL_RESULTS
    assert [result.source_filename for result in response.results] == [
        "source-0.txt",
        "source-1.txt",
        "source-2.txt",
        "source-3.txt",
        "source-4.txt",
    ]


def test_retrieval_skips_embedding_and_qdrant_when_no_document_is_ready() -> None:
    knowledge_base = create_knowledge_base("Not ready")
    create_chunk(
        knowledge_base.id,
        filename="uploading.txt",
        content="Not searchable yet.",
        status=DocumentStatus.PROCESSING,
    )
    embeddings = FakeEmbeddingService()
    vectors = FakeVectorStore([])

    response = SourceRetrievalService(
        embedding_service=embeddings,
        vector_store=vectors,
    ).search(knowledge_base.id, "Can I search this yet?")

    assert response.outcome == RetrievalOutcome.NO_READY_DOCUMENTS
    assert response.results == []
    assert embeddings.calls == []
    assert vectors.calls == []


def test_retrieval_returns_no_results_when_qdrant_has_no_hit_above_threshold() -> None:
    knowledge_base = create_knowledge_base("No results")
    create_chunk(
        knowledge_base.id,
        filename="ready.txt",
        content="Available source text.",
    )

    response = SourceRetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore([]),
    ).search(knowledge_base.id, "A question with no supporting evidence")

    assert response.outcome == RetrievalOutcome.NO_RESULTS
    assert response.results == []


def test_retrieval_discards_a_below_threshold_vector_hit_before_hydration() -> None:
    knowledge_base = create_knowledge_base("Below threshold")
    chunk = create_chunk(
        knowledge_base.id,
        filename="ready.txt",
        content="Available source text.",
    )

    response = SourceRetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore([VectorSearchHit(chunk_id=chunk.id, score=0.34)]),
    ).search(knowledge_base.id, "An unrelated question")

    assert response.outcome == RetrievalOutcome.NO_RESULTS
    assert response.results == []


def test_retrieval_limits_each_returned_excerpt_to_500_characters() -> None:
    knowledge_base = create_knowledge_base("Excerpt limit")
    chunk = create_chunk(
        knowledge_base.id,
        filename="long.txt",
        content="x" * 501,
    )

    response = SourceRetrievalService(
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore([VectorSearchHit(chunk_id=chunk.id, score=0.90)]),
    ).search(knowledge_base.id, "Show the source")

    assert len(response.results[0].excerpt) == 500
    assert response.results[0].excerpt.endswith("…")


def test_source_search_endpoint_trims_questions_and_rejects_blank_input(client) -> None:
    knowledge_base_id = client.post("/knowledge-bases", json={"name": "Endpoint"}).json()["id"]
    calls: list[str] = []

    class EndpointService:
        def search(self, _knowledge_base_id: UUID, question: str):
            calls.append(question)
            return type(
                "Response",
                (),
                {"outcome": RetrievalOutcome.NO_RESULTS, "results": []},
            )()

    app.dependency_overrides[get_source_retrieval_service] = EndpointService
    try:
        blank = client.post(
            f"/knowledge-bases/{knowledge_base_id}/sources/search",
            json={"question": "   "},
        )
        response = client.post(
            f"/knowledge-bases/{knowledge_base_id}/sources/search",
            json={"question": "  Trim this question  "},
        )
    finally:
        app.dependency_overrides.pop(get_source_retrieval_service, None)

    assert blank.status_code == 422
    assert response.status_code == 200
    assert response.json() == {"outcome": "no_results", "results": []}
    assert calls == ["Trim this question"]


def test_source_search_returns_a_safe_503_for_a_known_dependency_failure(client) -> None:
    knowledge_base_id = client.post("/knowledge-bases", json={"name": "Unavailable"}).json()["id"]

    class UnavailableService:
        def search(self, _knowledge_base_id: UUID, _question: str):
            raise RetrievalDependencyError() from VectorStoreError(
                VectorStoreErrorCategory.QUERY_FAILED
            )

    app.dependency_overrides[get_source_retrieval_service] = UnavailableService
    try:
        response = client.post(
            f"/knowledge-bases/{knowledge_base_id}/sources/search",
            json={"question": "What happened?"},
        )
    finally:
        app.dependency_overrides.pop(get_source_retrieval_service, None)

    assert response.status_code == 503
    assert response.json() == {
        "detail": "Source search is temporarily unavailable. Please try again."
    }


def test_source_search_returns_404_for_an_unknown_knowledge_base(client) -> None:
    response = client.post(
        f"/knowledge-bases/{uuid4()}/sources/search",
        json={"question": "Does this exist?"},
    )

    assert response.status_code == 404
    assert response.json() == {"detail": "Knowledge base not found."}
