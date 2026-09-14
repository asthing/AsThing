import hashlib
from datetime import UTC, datetime, timedelta
from itertools import count
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError

from app.database import SessionLocal
from app.document_models import (
    Document,
    DocumentChunk,
    DocumentProcessingJob,
    DocumentStatus,
    ProcessingJobState,
)
from app.embeddings import (
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_IDENTIFIER,
    EMBEDDING_MODEL_VERSION,
    EmbeddingBatch,
    EmbeddingErrorCategory,
    EmbeddingModelMetadata,
    EmbeddingServiceError,
)
from app.knowledge_bases import KnowledgeBase
from app.processing import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DocumentProcessingError,
    ProcessingErrorCategory,
    StoredDocumentProcessor,
    extract_text,
    normalize_text,
    prepare_chunks,
)
from app.processing_jobs import claim_processing_job
from app.storage import ObjectStorageError
from app.vector_store import VectorStoreError, VectorStoreErrorCategory
from app.worker_execution import run_processing_task

FIXTURE_DIRECTORY = Path(__file__).parent / "fixtures"


class FakeStorage:
    def __init__(self, content: bytes = b"") -> None:
        self.content = content
        self.downloaded_keys: list[str] = []
        self.fail_download = False

    def download_file(self, object_key: str) -> bytes:
        self.downloaded_keys.append(object_key)
        if self.fail_download:
            raise ObjectStorageError("private MinIO endpoint and credentials")
        return self.content


class FakeEmbeddingService:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.embedded_texts: list[list[str]] = []
        self.metadata = EmbeddingModelMetadata(
            identifier=EMBEDDING_MODEL_IDENTIFIER,
            version=EMBEDDING_MODEL_VERSION,
            dimension=EMBEDDING_DIMENSION,
        )

    def embed(self, texts) -> EmbeddingBatch:
        self.embedded_texts.append(list(texts))
        if self.error is not None:
            raise self.error
        return EmbeddingBatch(
            vectors=[[0.1] * EMBEDDING_DIMENSION for _ in texts],
            model=self.metadata,
        )


class FakeVectorStore:
    def __init__(
        self,
        upsert_error: Exception | None = None,
        delete_error: Exception | None = None,
    ) -> None:
        self.upsert_error = upsert_error
        self.delete_error = delete_error
        self.upserted = []
        self.deleted_point_ids = []
        self.deleted_document_ids = []

    def upsert(self, vectors) -> None:
        if self.upsert_error is not None:
            raise self.upsert_error
        self.upserted.append(list(vectors))

    def delete_points(self, point_ids) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted_point_ids.append(list(point_ids))

    def delete_document(self, document_id) -> None:
        if self.delete_error is not None:
            raise self.delete_error
        self.deleted_document_ids.append(document_id)


def create_document_and_job(file_type: str = "txt") -> tuple[UUID, UUID, str]:
    object_key = f"knowledge-bases/test/documents/{uuid4()}.{file_type}"
    session = SessionLocal()
    try:
        knowledge_base = KnowledgeBase(
            name=f"Extraction {uuid4()}", normalized_name=f"extraction-{uuid4()}"
        )
        session.add(knowledge_base)
        session.flush()
        document = Document(
            knowledge_base_id=knowledge_base.id,
            original_filename=f"source.{file_type}",
            file_size_bytes=100,
            file_type=file_type,
            storage_object_key=object_key,
        )
        session.add(document)
        session.flush()
        job = DocumentProcessingJob(document_id=document.id)
        session.add(job)
        session.commit()
        return document.id, job.id, object_key
    finally:
        session.close()


def load_processing_state(
    document_id: UUID, job_id: UUID
) -> tuple[Document, DocumentProcessingJob, list[DocumentChunk]]:
    session = SessionLocal()
    try:
        document = session.get(Document, document_id)
        job = session.get(DocumentProcessingJob, job_id)
        chunks = list(
            session.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.ordinal)
            )
        )
        assert document is not None
        assert job is not None
        session.expunge(document)
        session.expunge(job)
        for chunk in chunks:
            session.expunge(chunk)
        return document, job, chunks
    finally:
        session.close()


def clock_from(start: datetime):
    ticks = count()
    return lambda: start + timedelta(seconds=next(ticks))


@pytest.mark.parametrize(
    ("file_type", "fixture_name", "expected_text"),
    [
        ("txt", "sample.txt", "First line."),
        ("pdf", "sample.pdf", "PDF fixture text layer."),
        ("docx", "sample.docx", "DOCX fixture first paragraph."),
    ],
)
def test_extracts_supported_fixture_files(
    file_type: str, fixture_name: str, expected_text: str
) -> None:
    content = (FIXTURE_DIRECTORY / fixture_name).read_bytes()

    extracted = extract_text(file_type, content)

    assert expected_text in extracted


def test_normalizes_whitespace() -> None:
    assert normalize_text(" \n First\tline.   Second\r\nline.  ") == "First line. Second line."


def test_chunks_have_maximum_size_and_exact_overlap() -> None:
    normalized_text = "".join(str(index % 10) for index in range(1000))

    chunks = prepare_chunks(uuid4(), normalized_text)

    assert [chunk.ordinal for chunk in chunks] == [0, 1]
    assert len(chunks[0].content) == CHUNK_SIZE
    assert len(chunks[1].content) == 1000 - (CHUNK_SIZE - CHUNK_OVERLAP)
    assert chunks[0].content[-CHUNK_OVERLAP:] == chunks[1].content[:CHUNK_OVERLAP]
    assert all(len(chunk.content) <= CHUNK_SIZE for chunk in chunks)


def test_chunk_ordinals_and_hashes_are_stable() -> None:
    document_id = uuid4()
    normalized_text = "a" * 850

    first = prepare_chunks(document_id, normalized_text)
    second = prepare_chunks(document_id, normalized_text)

    assert first == second
    assert [chunk.ordinal for chunk in first] == list(range(len(first)))
    assert [chunk.content_hash for chunk in first] == [
        hashlib.sha256(chunk.content.encode("utf-8")).hexdigest() for chunk in first
    ]


@pytest.mark.parametrize(
    ("file_type", "fixture_name", "inline_content", "expected_category"),
    [
        ("txt", None, b"", ProcessingErrorCategory.NO_USABLE_TEXT),
        ("txt", None, b"\xff\xfe", ProcessingErrorCategory.TEXT_EXTRACTION_FAILED),
        ("pdf", "malformed.pdf", None, ProcessingErrorCategory.TEXT_EXTRACTION_FAILED),
        ("pdf", "image_only.pdf", None, ProcessingErrorCategory.NO_USABLE_TEXT),
        (
            "pdf",
            "encrypted.pdf",
            None,
            ProcessingErrorCategory.UNSUPPORTED_OR_INVALID_SOURCE,
        ),
        (
            "csv",
            None,
            b"name,value",
            ProcessingErrorCategory.UNSUPPORTED_OR_INVALID_SOURCE,
        ),
    ],
)
def test_invalid_or_empty_source_is_categorized_and_persists_no_chunks(
    file_type: str,
    fixture_name: str | None,
    inline_content: bytes | None,
    expected_category: ProcessingErrorCategory,
    caplog,
) -> None:
    document_id, job_id, _ = create_document_and_job(file_type)
    content = (
        (FIXTURE_DIRECTORY / fixture_name).read_bytes()
        if fixture_name is not None
        else inline_content or b""
    )
    storage = FakeStorage(content)
    processor = StoredDocumentProcessor(
        storage_provider=lambda: storage,
        vector_store=FakeVectorStore(),
    )

    result = run_processing_task(
        document_id,
        processor=processor,
        clock=clock_from(datetime(2026, 9, 14, 9, 0, tzinfo=UTC)),
    )

    document, job, chunks = load_processing_state(document_id, job_id)
    assert result is False
    assert chunks == []
    assert document.status == DocumentStatus.FAILED
    assert job.state == ProcessingJobState.COMPLETED
    assert caplog.records[-1].processing_error_category == expected_category.value
    assert "private" not in caplog.text
    assert "b'This '" not in caplog.text


def test_source_object_failure_is_categorized_and_persists_no_chunks(caplog) -> None:
    document_id, job_id, object_key = create_document_and_job()
    storage = FakeStorage()
    storage.fail_download = True
    processor = StoredDocumentProcessor(
        storage_provider=lambda: storage,
        vector_store=FakeVectorStore(),
    )

    result = run_processing_task(
        document_id,
        processor=processor,
        clock=clock_from(datetime(2026, 9, 14, 9, 0, tzinfo=UTC)),
    )

    document, job, chunks = load_processing_state(document_id, job_id)
    assert result is False
    assert storage.downloaded_keys == [object_key]
    assert chunks == []
    assert document.status == DocumentStatus.FAILED
    assert job.state == ProcessingJobState.COMPLETED
    assert (
        caplog.records[-1].processing_error_category
        == ProcessingErrorCategory.SOURCE_OBJECT_UNAVAILABLE.value
    )
    assert "credentials" not in caplog.text


def test_worker_sets_ready_only_after_chunks_embeddings_and_vectors_succeed() -> None:
    document_id, job_id, object_key = create_document_and_job()
    storage = FakeStorage((FIXTURE_DIRECTORY / "sample.txt").read_bytes())
    embedding_service = FakeEmbeddingService()
    vector_store = FakeVectorStore()
    processor = StoredDocumentProcessor(
        storage_provider=lambda: storage,
        embedding_service=embedding_service,
        vector_store=vector_store,
    )

    result = run_processing_task(
        document_id,
        processor=processor,
        clock=clock_from(datetime(2026, 9, 14, 9, 0, tzinfo=UTC)),
    )

    document, job, chunks = load_processing_state(document_id, job_id)
    assert result is True
    assert storage.downloaded_keys == [object_key]
    assert [chunk.content for chunk in chunks] == ["First line. Second line with spaces."]
    assert chunks[0].content_hash == hashlib.sha256(chunks[0].content.encode()).hexdigest()
    assert embedding_service.embedded_texts == [[chunks[0].content]]
    assert [[vector.point_id for vector in batch] for batch in vector_store.upserted] == [
        [chunks[0].qdrant_point_id]
    ]
    assert document.status == DocumentStatus.READY
    assert document.processing_completed_at is not None
    assert job.state == ProcessingJobState.COMPLETED
    assert job.completed_at is not None


def test_embedding_failure_is_safe_and_does_not_mark_document_ready(caplog) -> None:
    document_id, job_id, _ = create_document_and_job()
    storage = FakeStorage((FIXTURE_DIRECTORY / "sample.txt").read_bytes())
    processor = StoredDocumentProcessor(
        storage_provider=lambda: storage,
        embedding_service=FakeEmbeddingService(
            EmbeddingServiceError(EmbeddingErrorCategory.GENERATION_FAILED)
        ),
        vector_store=FakeVectorStore(),
    )

    result = run_processing_task(
        document_id,
        processor=processor,
        clock=clock_from(datetime(2026, 9, 14, 9, 0, tzinfo=UTC)),
    )

    document, job, chunks = load_processing_state(document_id, job_id)
    assert result is False
    assert chunks == []
    assert document.status == DocumentStatus.FAILED
    assert document.failure_code == "EMBEDDING_FAILED"
    assert job.state == ProcessingJobState.COMPLETED
    assert job.failure_code == "EMBEDDING_FAILED"
    assert (
        caplog.records[-1].processing_error_category
        == EmbeddingErrorCategory.GENERATION_FAILED.value
    )


def test_vector_failure_is_safe_and_does_not_mark_document_ready(caplog) -> None:
    document_id, job_id, _ = create_document_and_job()
    storage = FakeStorage((FIXTURE_DIRECTORY / "sample.txt").read_bytes())
    processor = StoredDocumentProcessor(
        storage_provider=lambda: storage,
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(
            VectorStoreError(VectorStoreErrorCategory.UPSERT_FAILED)
        ),
    )

    result = run_processing_task(
        document_id,
        processor=processor,
        clock=clock_from(datetime(2026, 9, 14, 9, 0, tzinfo=UTC)),
    )

    document, job, chunks = load_processing_state(document_id, job_id)
    assert result is False
    assert chunks == []
    assert document.status == DocumentStatus.FAILED
    assert document.failure_code == "VECTOR_PERSISTENCE_FAILED"
    assert job.state == ProcessingJobState.COMPLETED
    assert job.failure_code == "VECTOR_PERSISTENCE_FAILED"
    assert (
        caplog.records[-1].processing_error_category
        == VectorStoreErrorCategory.UPSERT_FAILED.value
    )


def test_cleanup_failure_remains_pending_and_logs_no_internal_details(caplog) -> None:
    document_id, job_id, _ = create_document_and_job()
    storage = FakeStorage((FIXTURE_DIRECTORY / "sample.txt").read_bytes())
    processor = StoredDocumentProcessor(
        storage_provider=lambda: storage,
        embedding_service=FakeEmbeddingService(),
        vector_store=FakeVectorStore(
            upsert_error=VectorStoreError(VectorStoreErrorCategory.UPSERT_FAILED),
            delete_error=RuntimeError("http://qdrant:6333 private response"),
        ),
    )

    result = run_processing_task(
        document_id,
        processor=processor,
        clock=clock_from(datetime(2026, 9, 14, 9, 0, tzinfo=UTC)),
    )

    document, job, chunks = load_processing_state(document_id, job_id)
    assert result is False
    assert chunks
    assert document.status == DocumentStatus.PROCESSING
    assert job.state == ProcessingJobState.CLEANUP_PENDING
    assert job.failure_code == "VECTOR_PERSISTENCE_FAILED"
    assert caplog.records[-1].processing_failure_code == "CLEANUP_FAILED"
    assert "http://qdrant:6333" not in caplog.text
    assert "private response" not in caplog.text


def test_chunk_persistence_rolls_back_complete_replacement_on_commit_failure() -> None:
    document_id, _, _ = create_document_and_job()
    claim = claim_processing_job(
        document_id,
        now=datetime(2026, 9, 14, 9, 0, tzinfo=UTC),
    )
    assert claim is not None
    processor = StoredDocumentProcessor()
    original_chunks = prepare_chunks(document_id, "Original usable chunk set")
    processor.persist(claim, original_chunks)
    replacement_chunks = prepare_chunks(document_id, "x" * 900)
    failing_session = SessionLocal()

    def fail_commit() -> None:
        raise OperationalError("COMMIT with private database details", {}, Exception("private"))

    failing_session.commit = fail_commit
    failing_processor = StoredDocumentProcessor(session_factory=lambda: failing_session)

    with pytest.raises(DocumentProcessingError) as error:
        failing_processor.persist(claim, replacement_chunks)

    _, _, persisted_chunks = load_processing_state(document_id, claim.job_id)
    assert error.value.category == ProcessingErrorCategory.CHUNK_PERSISTENCE_FAILED
    assert [(chunk.ordinal, chunk.content_hash) for chunk in persisted_chunks] == [
        (chunk.ordinal, chunk.content_hash) for chunk in original_chunks
    ]


def test_reprocessing_replaces_chunks_without_duplicates() -> None:
    document_id, _, _ = create_document_and_job()
    claim = claim_processing_job(
        document_id,
        now=datetime(2026, 9, 14, 9, 0, tzinfo=UTC),
    )
    assert claim is not None
    processor = StoredDocumentProcessor()
    prepared_chunks = prepare_chunks(document_id, "z" * 900)

    processor.persist(claim, prepared_chunks)
    processor.persist(claim, prepared_chunks)

    _, _, persisted_chunks = load_processing_state(document_id, claim.job_id)
    assert len(persisted_chunks) == len(prepared_chunks)
    assert [chunk.id for chunk in persisted_chunks] == [chunk.id for chunk in prepared_chunks]
    assert len({chunk.content_hash for chunk in persisted_chunks}) == len(persisted_chunks)
