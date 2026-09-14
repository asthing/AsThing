import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

EMBEDDING_MODEL_IDENTIFIER = "BAAI/bge-m3"
EMBEDDING_MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
EMBEDDING_MODEL_VERSION = EMBEDDING_MODEL_REVISION
EMBEDDING_DIMENSION = 1024
DEFAULT_EMBEDDING_BATCH_SIZE = 16


class EmbeddingErrorCategory(StrEnum):
    INVALID_INPUT = "InvalidEmbeddingInput"
    MODEL_UNAVAILABLE = "EmbeddingModelUnavailable"
    GENERATION_FAILED = "EmbeddingGenerationFailed"
    INVALID_OUTPUT = "InvalidEmbeddingOutput"


class EmbeddingServiceError(Exception):
    def __init__(self, category: EmbeddingErrorCategory) -> None:
        self.category = category
        super().__init__(category.value)


@dataclass(frozen=True)
class EmbeddingModelMetadata:
    identifier: str
    version: str
    dimension: int

    @property
    def revision(self) -> str:
        return self.version


@dataclass(frozen=True)
class EmbeddingBatch:
    vectors: list[list[float]]
    model: EmbeddingModelMetadata


class SentenceEmbeddingModel(Protocol):
    def encode(self, sentences: Sequence[str], **options: Any) -> Any: ...


EmbeddingModelFactory = Callable[[str, str, str], SentenceEmbeddingModel]


class LocalEmbeddingService:
    """Lazy, CPU-only sentence embedding service for persisted chunk text."""

    def __init__(
        self,
        *,
        model_factory: EmbeddingModelFactory | None = None,
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
        metadata: EmbeddingModelMetadata | None = None,
    ) -> None:
        if batch_size < 1:
            raise ValueError("Embedding batch size must be positive.")
        self._model_factory = model_factory or _load_sentence_transformer
        self._model: SentenceEmbeddingModel | None = None
        self.batch_size = batch_size
        self.metadata = metadata or EmbeddingModelMetadata(
            identifier=EMBEDDING_MODEL_IDENTIFIER,
            version=EMBEDDING_MODEL_VERSION,
            dimension=EMBEDDING_DIMENSION,
        )

    def embed(self, texts: Sequence[str]) -> EmbeddingBatch:
        if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
            raise EmbeddingServiceError(EmbeddingErrorCategory.INVALID_INPUT)

        model = self._get_model()
        try:
            encoded = model.encode(
                list(texts),
                batch_size=self.batch_size,
                device="cpu",
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception:
            raise EmbeddingServiceError(EmbeddingErrorCategory.GENERATION_FAILED) from None

        vectors = _validated_vectors(encoded, len(texts), self.metadata.dimension)
        return EmbeddingBatch(vectors=vectors, model=self.metadata)

    def _get_model(self) -> SentenceEmbeddingModel:
        if self._model is None:
            try:
                self._model = self._model_factory(
                    self.metadata.identifier,
                    "cpu",
                    self.metadata.revision,
                )
            except Exception:
                raise EmbeddingServiceError(EmbeddingErrorCategory.MODEL_UNAVAILABLE) from None
        return self._model


def _load_sentence_transformer(
    identifier: str, device: str, revision: str
) -> SentenceEmbeddingModel:
    # Keep this import inside the model factory. API and Alembic imports must stay lightweight.
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(identifier, device=device, revision=revision)


def _validated_vectors(encoded: Any, expected_count: int, dimension: int) -> list[list[float]]:
    try:
        raw_vectors = encoded.tolist() if hasattr(encoded, "tolist") else list(encoded)
        if len(raw_vectors) != expected_count:
            raise ValueError
        vectors = [[float(value) for value in vector] for vector in raw_vectors]
        if any(len(vector) != dimension for vector in vectors):
            raise ValueError
        if any(not math.isfinite(value) for vector in vectors for value in vector):
            raise ValueError
    except (TypeError, ValueError, OverflowError):
        raise EmbeddingServiceError(EmbeddingErrorCategory.INVALID_OUTPUT) from None
    return vectors
