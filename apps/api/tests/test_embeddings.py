import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_IDENTIFIER,
    EMBEDDING_MODEL_REVISION,
    EMBEDDING_MODEL_VERSION,
    EmbeddingErrorCategory,
    EmbeddingServiceError,
    LocalEmbeddingService,
)


class FakeEmbeddingModel:
    def __init__(self, vectors) -> None:
        self.vectors = vectors
        self.calls = []

    def encode(self, sentences, **options):
        self.calls.append((sentences, options))
        if isinstance(self.vectors, Exception):
            raise self.vectors
        return self.vectors


def test_embedding_service_is_lazy_cpu_only_and_exposes_metadata() -> None:
    factory_calls = []
    model = FakeEmbeddingModel([[0.5] * EMBEDDING_DIMENSION, [0.25] * EMBEDDING_DIMENSION])

    def model_factory(identifier: str, device: str, revision: str):
        factory_calls.append((identifier, device, revision))
        return model

    service = LocalEmbeddingService(model_factory=model_factory, batch_size=2)

    assert factory_calls == []
    assert service.metadata.identifier == EMBEDDING_MODEL_IDENTIFIER
    assert service.metadata.revision == EMBEDDING_MODEL_REVISION
    assert service.metadata.version == EMBEDDING_MODEL_REVISION
    assert service.metadata.version == EMBEDDING_MODEL_VERSION
    assert service.metadata.dimension == EMBEDDING_DIMENSION

    result = service.embed(["first chunk", "second chunk"])

    assert factory_calls == [
        (EMBEDDING_MODEL_IDENTIFIER, "cpu", EMBEDDING_MODEL_REVISION)
    ]
    assert len(result.vectors) == 2
    assert all(len(vector) == EMBEDDING_DIMENSION for vector in result.vectors)
    assert model.calls == [
        (
            ["first chunk", "second chunk"],
            {
                "batch_size": 2,
                "device": "cpu",
                "show_progress_bar": False,
                "convert_to_numpy": True,
                "normalize_embeddings": True,
            },
        )
    ]


def test_embedding_service_reuses_model_and_preserves_deterministic_output() -> None:
    vector = [float(index) / EMBEDDING_DIMENSION for index in range(EMBEDDING_DIMENSION)]
    model = FakeEmbeddingModel([vector])
    factory_calls = []

    def model_factory(_identifier: str, _device: str, _revision: str):
        factory_calls.append(True)
        return model

    service = LocalEmbeddingService(model_factory=model_factory)

    first = service.embed(["stable chunk"])
    second = service.embed(["stable chunk"])

    assert first == second
    assert factory_calls == [True]


@pytest.mark.parametrize("texts", [[], [""], ["   "]])
def test_embedding_service_rejects_empty_text(texts: list[str]) -> None:
    service = LocalEmbeddingService(
        model_factory=lambda _identifier, _device, _revision: FakeEmbeddingModel([])
    )

    with pytest.raises(EmbeddingServiceError) as error:
        service.embed(texts)

    assert error.value.category == EmbeddingErrorCategory.INVALID_INPUT


@pytest.mark.parametrize(
    "vectors",
    [
        [],
        [[0.0] * (EMBEDDING_DIMENSION - 1)],
        [[0.0] * EMBEDDING_DIMENSION, [0.0] * EMBEDDING_DIMENSION],
    ],
)
def test_embedding_service_rejects_invalid_output(vectors) -> None:
    service = LocalEmbeddingService(
        model_factory=lambda _identifier, _device, _revision: FakeEmbeddingModel(vectors)
    )

    with pytest.raises(EmbeddingServiceError) as error:
        service.embed(["one chunk"])

    assert error.value.category == EmbeddingErrorCategory.INVALID_OUTPUT


def test_embedding_failures_are_typed_and_do_not_include_internal_details() -> None:
    service = LocalEmbeddingService(
        model_factory=lambda _identifier, _device, _revision: FakeEmbeddingModel(
            RuntimeError("private model cache path and source text")
        )
    )

    with pytest.raises(EmbeddingServiceError) as error:
        service.embed(["secret chunk text"])

    assert error.value.category == EmbeddingErrorCategory.GENERATION_FAILED
    assert str(error.value) == EmbeddingErrorCategory.GENERATION_FAILED.value
    assert error.value.__cause__ is None


def test_api_and_alembic_imports_do_not_import_embedding_or_qdrant_runtimes() -> None:
    api_directory = Path(__file__).parents[1]
    environment = os.environ.copy()
    environment["DATABASE_URL"] = "postgresql+psycopg://unused:unused@localhost/unused"
    script = """
import contextlib
import io
import sys
from alembic import command
from alembic.config import Config

import app.main
import app.embeddings
import app.vector_store

assert 'sentence_transformers' not in sys.modules
assert 'torch' not in sys.modules
assert 'qdrant_client' not in sys.modules
with contextlib.redirect_stdout(io.StringIO()):
    command.upgrade(Config('alembic.ini'), 'head', sql=True)
assert 'sentence_transformers' not in sys.modules
assert 'torch' not in sys.modules
assert 'qdrant_client' not in sys.modules
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=api_directory,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
