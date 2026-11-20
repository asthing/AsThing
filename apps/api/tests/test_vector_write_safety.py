"""An accepted-but-unacknowledged write is not settled by a later delete."""

from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, sessionmaker
from test_document_deletion import remaining, seed_document

from app.database import SessionLocal, engine
from app.document_deletion import DocumentDeletionService, get_document_deletion_service
from app.document_models import Document, DocumentChunk, DocumentProcessingJob, utc_now
from app.main import app
from app.processing import StoredDocumentProcessor, prepare_chunks
from app.processing_jobs import ClaimedProcessingJob, claim_processing_job
from app.processing_outcomes import complete_processing_success
from app.vector_store import (
    VectorPoint,
    VectorStoreError,
    VectorStoreErrorCategory,
    _QdrantClientAdapter,
)
from app.worker_execution import run_processing_task, run_recovery_batch


class AcceptedWrite:
    def __init__(self, document_id, *, worker_loss=False):
        self.document_id = document_id
        self.worker_loss = worker_loss
        self.pending = []
        self.points = {}
        self.calls = 0

    def upsert(self, vectors):
        self.calls += 1
        self.pending.extend(vectors)
        if self.worker_loss:
            raise SystemExit("simulated worker loss")
        raise VectorStoreError(VectorStoreErrorCategory.UPSERT_FAILED)

    def finish_late(self):
        for vector in self.pending:
            self.points[vector.point_id] = vector.document_id
        self.pending.clear()

    def delete_document(self, document_id):
        self.points = {key: value for key, value in self.points.items() if value != document_id}

    def delete_points(self, point_ids):
        for point_id in point_ids:
            self.points.pop(point_id, None)


class Storage:
    def download_file(self, _key):
        return b"A fictional source fact."

    def delete_file_strict(self, _key):
        pass


class Embeddings:
    def embed(self, texts):
        return SimpleNamespace(
            model=SimpleNamespace(identifier="fixture", version="fixture"),
            vectors=[[0.1] * 1024 for _ in texts],
        )


def processor(remote):
    return StoredDocumentProcessor(
        storage_provider=Storage, embedding_service=Embeddings(), vector_store=remote,
    )


def test_accepted_timeout_must_not_allow_204_then_orphan_recreation(client):
    kb, target, _, _ = seed_document("Uploaded")
    _, keeper, _, _ = seed_document(knowledge_base_id=kb)
    remote = AcceptedWrite(target)
    service = DocumentDeletionService(
        storage_provider=Storage, vector_store_provider=lambda: remote,
    )
    app.dependency_overrides[get_document_deletion_service] = lambda: service
    try:
        assert not run_processing_task(target, processor=processor(remote))
        response = client.delete(f"/knowledge-bases/{kb}/documents/{target}")
        remote.finish_late()
        # Before the fix this returned 204, removed all identifiers, and then recreated a vector.
        assert response.status_code != 204
        document, chunks, jobs = remaining(target)
        assert document.status == "Processing"
        assert chunks and jobs[0].vector_write_token is not None
        assert jobs[0].state == "CleanupPending"
        assert remote.points and remaining(keeper)[0] is not None
    finally:
        app.dependency_overrides.pop(get_document_deletion_service, None)


@pytest.mark.parametrize("worker_loss", [False, True])
def test_unresolved_write_survives_worker_loss_and_repeated_recovery(client, worker_loss):
    _, target, _, job_id = seed_document("Uploaded")
    remote = AcceptedWrite(target, worker_loss=worker_loss)
    phases = processor(remote)
    if worker_loss:
        claim = claim_processing_job(target)
        phases.persist(claim, prepare_chunks(target, "A fictional source fact."))
        with pytest.raises(SystemExit):
            phases.upsert_vectors(claim, phases.embed(claim))
    else:
        assert not run_processing_task(target, processor=phases)
    with SessionLocal() as session:
        job = session.get(DocumentProcessingJob, job_id)
        token, attempt = job.vector_write_token, job.attempt_count
        assert token is not None
        job.lease_expires_at = utc_now() - timedelta(seconds=1)
        session.commit()

    class Dispatch:
        def dispatch(self, *_args):
            return "recovery-task"

    for late in (False, True, True):
        if late:
            remote.finish_late()
        result = run_recovery_batch(dispatcher=Dispatch())
        assert result.found == 1
        assert not run_processing_task(target, processor=phases)
        assert claim_processing_job(target) is None
        document, chunks, jobs = remaining(target)
        assert document.status == "Processing" and chunks
        assert jobs[0].vector_write_token == token
        assert jobs[0].attempt_count == attempt
        assert jobs[0].state == "CleanupPending"
    assert remote.calls == 1


def test_delete_pending_keeps_identifiers_even_after_empty_query_and_successful_delete(client):
    kb, target, _, job_id = seed_document("Failed")
    _, keeper, _, _ = seed_document(knowledge_base_id=kb)
    remote = AcceptedWrite(target)
    service = DocumentDeletionService(
        storage_provider=Storage, vector_store_provider=lambda: remote,
    )

    with SessionLocal() as session:
        job = session.get(DocumentProcessingJob, job_id)
        job.vector_write_token = uuid4()
        job.vector_write_started_at = utc_now()
        session.commit()
    app.dependency_overrides[get_document_deletion_service] = lambda: service
    try:
        for _ in range(3):
            remote.delete_document(target)  # Not a settlement barrier, even when acknowledged.
            response = client.delete(f"/knowledge-bases/{kb}/documents/{target}")
            assert response.status_code == 503
            assert remaining(target)[0].deletion_pending
            assert remaining(target)[1] and remaining(target)[2]
        assert remaining(keeper)[0] is not None
    finally:
        app.dependency_overrides.pop(get_document_deletion_service, None)


def test_successful_ack_clears_durable_intent_before_ready_and_allows_deletion(client):
    kb, target, _, job_id = seed_document("Uploaded")

    class Acknowledged(AcceptedWrite):
        def upsert(self, vectors):
            # A separate transaction can see intent BEFORE anything reaches the remote service.
            with SessionLocal() as session:
                job = session.get(DocumentProcessingJob, job_id)
                assert job.vector_write_token is not None
                assert job.vector_write_started_at is not None
                assert session.scalar(select(DocumentChunk.id).where(
                    DocumentChunk.document_id == target,
                )) is not None
            self.pending.extend(vectors)
            self.finish_late()

    remote = Acknowledged(target)
    assert run_processing_task(target, processor=processor(remote))
    with SessionLocal() as session:
        assert session.get(Document, target).status == "Ready"
        assert session.get(DocumentProcessingJob, job_id).vector_write_token is None
    DocumentDeletionService(storage_provider=Storage, vector_store_provider=lambda: remote).delete(
        kb, target,
    )
    remote.finish_late()
    assert remaining(target) == (None, [], []) and not remote.points


def test_late_completion_during_cleanup_cannot_finalize_or_discard_identifiers(client):
    _, target, _, job_id = seed_document("Uploaded")

    class CompletesDuringCleanup(AcceptedWrite):
        def delete_document(self, document_id):
            super().delete_document(document_id)
            self.finish_late()  # Earlier accepted upsert completes AFTER acknowledged delete.

    remote = CompletesDuringCleanup(target)
    assert not run_processing_task(target, processor=processor(remote))
    document, chunks, jobs = remaining(target)
    assert document.status == "Processing" and chunks
    assert jobs[0].state == "CleanupPending" and jobs[0].vector_write_token is not None
    assert remote.points  # Still tracked, never declared deleted or retrieval-ready.
    claim = ClaimedProcessingJob(job_id, target, jobs[0].attempt_count, utc_now())
    assert not complete_processing_success(claim)


@pytest.mark.parametrize("fail_commit", [1, 2])
def test_intent_commit_failure_prevents_dispatch_and_ack_commit_failure_retains_intent(
    client, fail_commit,
):
    kb, target, _, job_id = seed_document("Processing")
    remote = AcceptedWrite(target)

    class Acknowledged(AcceptedWrite):
        def upsert(self, vectors):
            self.calls += 1

    remote = Acknowledged(target)

    class FailingCommit(Session):
        commits = 0

        def commit(self):
            self.commits += 1
            if self.commits == fail_commit:
                raise OperationalError("safe synthetic failure", {}, Exception())
            return super().commit()

    from app.vector_store import ChunkVector

    phases = StoredDocumentProcessor(
        session_factory=sessionmaker(engine, class_=FailingCommit), vector_store=remote,
    )
    with SessionLocal() as session:
        chunk = session.scalar(select(DocumentChunk).where(DocumentChunk.document_id == target))
        vector = ChunkVector(
            chunk.qdrant_point_id, [0.1] * 1024, chunk.id, target, kb, 0, "model", "revision",
        )
    claim = ClaimedProcessingJob(job_id, target, 1, utc_now() + timedelta(minutes=5))
    with pytest.raises(VectorStoreError):
        phases.upsert_vectors(claim, [vector])
    assert remote.calls == fail_commit - 1
    assert (remaining(target)[2][0].vector_write_token is not None) == (fail_commit == 2)


@pytest.mark.parametrize("status", ["completed", "acknowledged"])
def test_pinned_adapter_requires_completed_acknowledgement(status):
    calls = []

    class Client:
        def upsert(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(status=status)

    adapter = _QdrantClientAdapter.__new__(_QdrantClientAdapter)
    adapter.client = Client()
    points = [VectorPoint(uuid4(), [0.1] * 1024, {})]
    if status == "completed":
        adapter.upsert("document_chunks", points)
    else:
        with pytest.raises(VectorStoreError):
            adapter.upsert("document_chunks", points)
    assert calls[0]["wait"] is True
