from uuid import uuid4

import pytest

from app.embeddings import EMBEDDING_MODEL_REVISION
from app.vector_store import (
    DOCUMENT_CHUNKS_COLLECTION,
    VECTOR_DIMENSION,
    VECTOR_DISTANCE,
    ChunkVector,
    QdrantVectorStore,
    VectorCollectionConfig,
    VectorSearchHit,
    VectorStoreError,
    VectorStoreErrorCategory,
)


class FakeVectorStoreAdapter:
    def __init__(self, config: VectorCollectionConfig | None = None) -> None:
        self.config = config
        self.created = []
        self.points_by_id = {}
        self.upsert_calls = []
        self.deleted_documents = []
        self.deleted_point_ids = []
        self.upsert_error: Exception | None = None
        self.search_error: Exception | None = None
        self.search_results: list[VectorSearchHit] = []
        self.search_calls = []
        self.create_error: Exception | None = None
        self.config_after_create_error: VectorCollectionConfig | None = None

    def collection_config(self, collection_name: str):
        return self.config

    def create_collection(self, collection_name: str, *, dimension: int, distance: str):
        self.created.append((collection_name, dimension, distance))
        if self.create_error is not None:
            self.config = self.config_after_create_error
            raise self.create_error
        self.config = VectorCollectionConfig(dimension=dimension, distance=distance)

    def upsert(self, collection_name: str, points) -> None:
        if self.upsert_error is not None:
            raise self.upsert_error
        self.upsert_calls.append((collection_name, list(points)))
        for point in points:
            self.points_by_id[point.point_id] = point

    def search(
        self,
        collection_name: str,
        vector,
        *,
        knowledge_base_id,
        embedding_model: str,
        embedding_revision: str,
        limit: int,
        score_threshold: float,
    ):
        self.search_calls.append(
            (
                collection_name,
                vector,
                knowledge_base_id,
                embedding_model,
                embedding_revision,
                limit,
                score_threshold,
            )
        )
        if self.search_error is not None:
            raise self.search_error
        return self.search_results

    def delete_document(self, collection_name: str, document_id) -> None:
        self.deleted_documents.append((collection_name, document_id))

    def delete_points(self, collection_name: str, point_ids) -> None:
        self.deleted_point_ids.append((collection_name, list(point_ids)))


def make_chunk_vector(*, point_id=None) -> ChunkVector:
    return ChunkVector(
        point_id=point_id or uuid4(),
        vector=[0.1] * VECTOR_DIMENSION,
        chunk_id=uuid4(),
        document_id=uuid4(),
        knowledge_base_id=uuid4(),
        ordinal=0,
        embedding_model="BAAI/bge-m3",
        embedding_version=EMBEDDING_MODEL_REVISION,
    )


def test_qdrant_collection_is_created_with_1024_cosine_configuration() -> None:
    adapter = FakeVectorStoreAdapter()
    store = QdrantVectorStore(adapter_factory=lambda: adapter)

    store.ensure_collection()

    assert adapter.created == [
        (DOCUMENT_CHUNKS_COLLECTION, VECTOR_DIMENSION, VECTOR_DISTANCE)
    ]


def test_qdrant_rejects_an_incompatible_existing_collection() -> None:
    adapter = FakeVectorStoreAdapter(VectorCollectionConfig(384, VECTOR_DISTANCE))
    store = QdrantVectorStore(adapter_factory=lambda: adapter)

    with pytest.raises(VectorStoreError) as error:
        store.ensure_collection()

    assert error.value.category == VectorStoreErrorCategory.COLLECTION_INCOMPATIBLE


def test_qdrant_accepts_collection_created_by_a_concurrent_initializer() -> None:
    adapter = FakeVectorStoreAdapter()
    adapter.create_error = RuntimeError("collection already exists")
    adapter.config_after_create_error = VectorCollectionConfig(
        VECTOR_DIMENSION, VECTOR_DISTANCE
    )
    store = QdrantVectorStore(adapter_factory=lambda: adapter)

    store.ensure_collection()

    assert adapter.created == [
        (DOCUMENT_CHUNKS_COLLECTION, VECTOR_DIMENSION, VECTOR_DISTANCE)
    ]


def test_vector_payload_and_deterministic_point_id_are_upserted_once() -> None:
    adapter = FakeVectorStoreAdapter(
        VectorCollectionConfig(VECTOR_DIMENSION, VECTOR_DISTANCE)
    )
    store = QdrantVectorStore(adapter_factory=lambda: adapter)
    chunk = make_chunk_vector()

    store.upsert([chunk])

    assert len(adapter.upsert_calls) == 1
    collection_name, points = adapter.upsert_calls[0]
    assert collection_name == DOCUMENT_CHUNKS_COLLECTION
    assert [point.point_id for point in points] == [chunk.point_id]
    assert points[0].payload == {
        "chunk_id": str(chunk.chunk_id),
        "document_id": str(chunk.document_id),
        "knowledge_base_id": str(chunk.knowledge_base_id),
        "ordinal": chunk.ordinal,
        "embedding_model": chunk.embedding_model,
        "embedding_revision": EMBEDDING_MODEL_REVISION,
        "embedding_version": chunk.embedding_version,
    }


def test_repeated_upsert_replaces_the_same_point_without_duplicates() -> None:
    adapter = FakeVectorStoreAdapter(
        VectorCollectionConfig(VECTOR_DIMENSION, VECTOR_DISTANCE)
    )
    store = QdrantVectorStore(adapter_factory=lambda: adapter)
    point_id = uuid4()
    chunk = make_chunk_vector(point_id=point_id)

    store.upsert([chunk])
    store.upsert([chunk])

    assert len(adapter.points_by_id) == 1
    assert list(adapter.points_by_id) == [point_id]


def test_vector_store_supports_document_and_point_deletion() -> None:
    adapter = FakeVectorStoreAdapter()
    store = QdrantVectorStore(adapter_factory=lambda: adapter)
    document_id = uuid4()
    point_ids = [uuid4(), uuid4()]

    store.delete_document(document_id)
    store.delete_points(point_ids)

    assert adapter.deleted_documents == [(DOCUMENT_CHUNKS_COLLECTION, document_id)]
    assert adapter.deleted_point_ids == [(DOCUMENT_CHUNKS_COLLECTION, point_ids)]


def test_qdrant_failure_is_typed_without_raw_infrastructure_details() -> None:
    adapter = FakeVectorStoreAdapter(
        VectorCollectionConfig(VECTOR_DIMENSION, VECTOR_DISTANCE)
    )
    adapter.upsert_error = RuntimeError("http://qdrant:6333 private response")
    store = QdrantVectorStore(adapter_factory=lambda: adapter)

    with pytest.raises(VectorStoreError) as error:
        store.upsert([make_chunk_vector()])

    assert error.value.category == VectorStoreErrorCategory.UPSERT_FAILED
    assert str(error.value) == VectorStoreErrorCategory.UPSERT_FAILED.value
    assert error.value.__cause__ is None


def test_vector_store_rejects_wrong_dimension_and_duplicate_point_ids() -> None:
    adapter = FakeVectorStoreAdapter()
    store = QdrantVectorStore(adapter_factory=lambda: adapter)
    wrong_dimension = make_chunk_vector()
    object.__setattr__(wrong_dimension, "vector", [0.1] * 10)
    duplicate_id = uuid4()

    with pytest.raises(VectorStoreError) as dimension_error:
        store.upsert([wrong_dimension])
    with pytest.raises(VectorStoreError) as duplicate_error:
        store.upsert(
            [make_chunk_vector(point_id=duplicate_id), make_chunk_vector(point_id=duplicate_id)]
        )

    assert dimension_error.value.category == VectorStoreErrorCategory.VECTOR_INVALID
    assert duplicate_error.value.category == VectorStoreErrorCategory.VECTOR_INVALID
    assert adapter.upsert_calls == []


def test_vector_search_filters_a_knowledge_base_and_pinned_embedding_identity() -> None:
    adapter = FakeVectorStoreAdapter(
        VectorCollectionConfig(VECTOR_DIMENSION, VECTOR_DISTANCE)
    )
    expected = VectorSearchHit(chunk_id=uuid4(), score=0.86)
    adapter.search_results = [expected]
    store = QdrantVectorStore(adapter_factory=lambda: adapter)
    knowledge_base_id = uuid4()

    results = store.search(
        [0.2] * VECTOR_DIMENSION,
        knowledge_base_id=knowledge_base_id,
        embedding_model="BAAI/bge-m3",
        embedding_revision=EMBEDDING_MODEL_REVISION,
        limit=5,
        score_threshold=0.35,
    )

    assert results == [expected]
    assert adapter.search_calls == [
        (
            DOCUMENT_CHUNKS_COLLECTION,
            [0.2] * VECTOR_DIMENSION,
            knowledge_base_id,
            "BAAI/bge-m3",
            EMBEDDING_MODEL_REVISION,
            5,
            0.35,
        )
    ]


def test_vector_search_returns_a_typed_error_without_raw_qdrant_details() -> None:
    adapter = FakeVectorStoreAdapter(
        VectorCollectionConfig(VECTOR_DIMENSION, VECTOR_DISTANCE)
    )
    adapter.search_error = RuntimeError("http://qdrant:6333 query contained private text")
    store = QdrantVectorStore(adapter_factory=lambda: adapter)

    with pytest.raises(VectorStoreError) as error:
        store.search(
            [0.2] * VECTOR_DIMENSION,
            knowledge_base_id=uuid4(),
            embedding_model="BAAI/bge-m3",
            embedding_revision=EMBEDDING_MODEL_REVISION,
            limit=5,
            score_threshold=0.35,
        )

    assert error.value.category == VectorStoreErrorCategory.QUERY_FAILED
    assert str(error.value) == VectorStoreErrorCategory.QUERY_FAILED.value
    assert error.value.__cause__ is None
