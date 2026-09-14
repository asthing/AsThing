import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol
from uuid import UUID

DOCUMENT_CHUNKS_COLLECTION = "document_chunks"
QDRANT_COLLECTION_ENV = "QDRANT_COLLECTION"
QDRANT_URL_ENV = "QDRANT_URL"
VECTOR_DIMENSION = 1024
VECTOR_DISTANCE = "Cosine"


class VectorStoreErrorCategory(StrEnum):
    CONFIGURATION_INVALID = "VectorStoreConfigurationInvalid"
    COLLECTION_UNAVAILABLE = "VectorCollectionUnavailable"
    COLLECTION_INCOMPATIBLE = "VectorCollectionIncompatible"
    VECTOR_INVALID = "VectorInvalid"
    UPSERT_FAILED = "VectorUpsertFailed"
    DELETE_FAILED = "VectorDeleteFailed"


class VectorStoreError(Exception):
    def __init__(self, category: VectorStoreErrorCategory) -> None:
        self.category = category
        super().__init__(category.value)


@dataclass(frozen=True)
class VectorCollectionConfig:
    dimension: int
    distance: str


@dataclass(frozen=True)
class ChunkVector:
    point_id: UUID
    vector: list[float]
    chunk_id: UUID
    document_id: UUID
    knowledge_base_id: UUID
    ordinal: int
    embedding_model: str
    embedding_version: str


@dataclass(frozen=True)
class VectorPoint:
    point_id: UUID
    vector: list[float]
    payload: dict[str, str | int]


class VectorStoreAdapter(Protocol):
    def collection_config(self, collection_name: str) -> VectorCollectionConfig | None: ...

    def create_collection(
        self, collection_name: str, *, dimension: int, distance: str
    ) -> None: ...

    def upsert(self, collection_name: str, points: Sequence[VectorPoint]) -> None: ...

    def delete_document(self, collection_name: str, document_id: UUID) -> None: ...

    def delete_points(self, collection_name: str, point_ids: Sequence[UUID]) -> None: ...


VectorStoreAdapterFactory = Callable[[], VectorStoreAdapter]


class QdrantVectorStore:
    """Qdrant-only vector persistence behind a small, testable adapter."""

    def __init__(
        self,
        *,
        adapter_factory: VectorStoreAdapterFactory | None = None,
        collection_name: str | None = None,
    ) -> None:
        self.collection_name = collection_name or os.getenv(
            QDRANT_COLLECTION_ENV, DOCUMENT_CHUNKS_COLLECTION
        )
        if self.collection_name != DOCUMENT_CHUNKS_COLLECTION:
            raise VectorStoreError(VectorStoreErrorCategory.CONFIGURATION_INVALID)
        self._adapter_factory = adapter_factory or _create_qdrant_adapter
        self._adapter: VectorStoreAdapter | None = None

    def ensure_collection(self) -> None:
        adapter = self._get_adapter()
        try:
            config = adapter.collection_config(self.collection_name)
        except VectorStoreError:
            raise
        except Exception:
            raise VectorStoreError(VectorStoreErrorCategory.COLLECTION_UNAVAILABLE) from None

        if config is None:
            try:
                adapter.create_collection(
                    self.collection_name,
                    dimension=VECTOR_DIMENSION,
                    distance=VECTOR_DISTANCE,
                )
                return
            except Exception:
                try:
                    config = adapter.collection_config(self.collection_name)
                except Exception:
                    raise VectorStoreError(
                        VectorStoreErrorCategory.COLLECTION_UNAVAILABLE
                    ) from None
                if config is None:
                    raise VectorStoreError(
                        VectorStoreErrorCategory.COLLECTION_UNAVAILABLE
                    ) from None

        if config.dimension != VECTOR_DIMENSION or config.distance.casefold() != (
            VECTOR_DISTANCE.casefold()
        ):
            raise VectorStoreError(VectorStoreErrorCategory.COLLECTION_INCOMPATIBLE)

    def upsert(self, chunks: Sequence[ChunkVector]) -> None:
        points = _build_points(chunks)
        self.ensure_collection()
        try:
            self._get_adapter().upsert(self.collection_name, points)
        except Exception:
            raise VectorStoreError(VectorStoreErrorCategory.UPSERT_FAILED) from None

    def delete_document(self, document_id: UUID) -> None:
        try:
            self._get_adapter().delete_document(self.collection_name, document_id)
        except Exception:
            raise VectorStoreError(VectorStoreErrorCategory.DELETE_FAILED) from None

    def delete_points(self, point_ids: Sequence[UUID]) -> None:
        if not point_ids:
            return
        try:
            self._get_adapter().delete_points(self.collection_name, point_ids)
        except Exception:
            raise VectorStoreError(VectorStoreErrorCategory.DELETE_FAILED) from None

    def _get_adapter(self) -> VectorStoreAdapter:
        if self._adapter is None:
            try:
                self._adapter = self._adapter_factory()
            except VectorStoreError:
                raise
            except Exception:
                raise VectorStoreError(
                    VectorStoreErrorCategory.COLLECTION_UNAVAILABLE
                ) from None
        return self._adapter


def _build_points(chunks: Sequence[ChunkVector]) -> list[VectorPoint]:
    if not chunks:
        raise VectorStoreError(VectorStoreErrorCategory.VECTOR_INVALID)
    if len({chunk.point_id for chunk in chunks}) != len(chunks):
        raise VectorStoreError(VectorStoreErrorCategory.VECTOR_INVALID)

    points = []
    for chunk in chunks:
        if len(chunk.vector) != VECTOR_DIMENSION:
            raise VectorStoreError(VectorStoreErrorCategory.VECTOR_INVALID)
        points.append(
            VectorPoint(
                point_id=chunk.point_id,
                vector=chunk.vector,
                payload={
                    "chunk_id": str(chunk.chunk_id),
                    "document_id": str(chunk.document_id),
                    "knowledge_base_id": str(chunk.knowledge_base_id),
                    "ordinal": chunk.ordinal,
                    "embedding_model": chunk.embedding_model,
                    "embedding_revision": chunk.embedding_version,
                    "embedding_version": chunk.embedding_version,
                },
            )
        )
    return points


def _create_qdrant_adapter() -> VectorStoreAdapter:
    url = os.getenv(QDRANT_URL_ENV)
    if not url:
        raise VectorStoreError(VectorStoreErrorCategory.CONFIGURATION_INVALID)
    return _QdrantClientAdapter(url)


class _QdrantClientAdapter:
    def __init__(self, url: str) -> None:
        # Keep Qdrant imports out of API startup, Alembic, and import-only paths.
        from qdrant_client import QdrantClient

        self.client = QdrantClient(url=url)

    def collection_config(self, collection_name: str) -> VectorCollectionConfig | None:
        if not self.client.collection_exists(collection_name):
            return None
        collection = self.client.get_collection(collection_name)
        vectors = collection.config.params.vectors
        if isinstance(vectors, dict):
            raise VectorStoreError(VectorStoreErrorCategory.COLLECTION_INCOMPATIBLE)
        distance = getattr(vectors.distance, "value", str(vectors.distance))
        return VectorCollectionConfig(dimension=vectors.size, distance=distance)

    def create_collection(
        self, collection_name: str, *, dimension: int, distance: str
    ) -> None:
        from qdrant_client.models import Distance, VectorParams

        if distance.casefold() != VECTOR_DISTANCE.casefold():
            raise VectorStoreError(VectorStoreErrorCategory.CONFIGURATION_INVALID)
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

    def upsert(self, collection_name: str, points: Sequence[VectorPoint]) -> None:
        from qdrant_client.models import PointStruct

        self.client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(id=str(point.point_id), vector=point.vector, payload=point.payload)
                for point in points
            ],
            wait=True,
        )

    def delete_document(self, collection_name: str, document_id: UUID) -> None:
        from qdrant_client.models import FieldCondition, Filter, FilterSelector, MatchValue

        if not self.client.collection_exists(collection_name):
            return
        self.client.delete(
            collection_name=collection_name,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id", match=MatchValue(value=str(document_id))
                        )
                    ]
                )
            ),
            wait=True,
        )

    def delete_points(self, collection_name: str, point_ids: Sequence[UUID]) -> None:
        from qdrant_client.models import PointIdsList

        if not self.client.collection_exists(collection_name):
            return
        self.client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=[str(point_id) for point_id in point_ids]),
            wait=True,
        )
