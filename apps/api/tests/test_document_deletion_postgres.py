"""Opt-in lock tests. Use only the disposable deletion stack; never the default database."""

import os
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from threading import Event
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, event, select, text
from sqlalchemy.exc import ProgrammingError
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.schema import CreateSchema, DropSchema

from alembic.migration import MigrationContext
from alembic.operations import Operations
from app.database import Base
from app.document_deletion import (
    DeletionDocumentNotFound,
    DeletionProcessingConflict,
    DocumentDeletionService,
)
from app.document_models import Document, DocumentChunk, DocumentProcessingJob, utc_now
from app.knowledge_bases import KnowledgeBase
from app.processing import DocumentProcessingError, StoredDocumentProcessor, prepare_chunks
from app.processing_jobs import ClaimedProcessingJob, claim_processing_job
from app.vector_store import ChunkVector, VectorStoreError

pytestmark = pytest.mark.skipif(
    not os.getenv("DELETION_TEST_DATABASE_URL"), reason="requires disposable PostgreSQL stack"
)


@pytest.fixture
def database():
    url = os.environ["DELETION_TEST_DATABASE_URL"]
    schema = f"deletion_test_{uuid4().hex}"
    root_engine = create_engine(url, connect_args={"application_name": schema})
    with root_engine.begin() as connection:
        connection.execute(CreateSchema(schema))
    scoped = root_engine.execution_options(schema_translate_map={None: schema})
    Base.metadata.create_all(scoped)
    factory = sessionmaker(scoped, expire_on_commit=False)
    try:
        yield factory, scoped
    finally:
        with root_engine.begin() as connection:
            connection.execute(DropSchema(schema, cascade=True))
        root_engine.dispose()


def seed(factory, status="Uploaded"):
    with factory() as session:
        name = f"isolated-{uuid4()}"
        kb = KnowledgeBase(name=name, normalized_name=name)
        session.add(kb)
        session.flush()
        document = Document(
            knowledge_base_id=kb.id, original_filename="same.txt", file_type="txt",
            file_size_bytes=10, storage_object_key=f"test/{uuid4()}", status=status,
            processing_attempt_count=1,
        )
        session.add(document)
        session.flush()
        job = DocumentProcessingJob(
            document_id=document.id, state="Queued" if status == "Uploaded" else
            "Processing" if status == "Processing" else "Completed",
            attempt_count=1, lease_expires_at=utc_now() + timedelta(minutes=5),
        )
        session.add(job)
        session.commit()
        return kb.id, document.id, job.id


class Externals:
    def __init__(self):
        self.calls = []
        self.entered = Event()
        self.release = Event()
        self.release.set()

    def delete_document(self, document_id):
        self.calls.append(("delete_vectors", document_id))
        self.entered.set()
        assert self.release.wait(10), "cleanup gate timed out"

    def delete_file_strict(self, key):
        self.calls.append(("delete_object", key))


def deletion(factory, externals):
    return DocumentDeletionService(
        session_factory=factory, storage_provider=lambda: externals,
        vector_store_provider=lambda: externals,
    )


def pausing_factory(engine, entered, release):
    class PausedCommit(Session):
        def commit(self):
            entered.set()
            assert release.wait(10), "transaction gate timed out"
            return super().commit()

    return sessionmaker(engine, class_=PausedCommit, expire_on_commit=False)


def assert_database_wait(factory):
    """Require PostgreSQL itself to report a blocked connection, not just an unfinished Future."""
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        with factory() as session:
            if session.scalar(text(
                "SELECT EXISTS(SELECT 1 FROM pg_stat_activity "
                "WHERE application_name = current_setting('application_name') "
                "AND pid <> pg_backend_pid() AND wait_event_type = 'Lock')"
            )):
                return
        Event().wait(0.01)
    pytest.fail("the competing PostgreSQL transaction did not block on the coordination lock")


def test_postgres_claim_wins_then_delete_is_409(database):
    factory, engine = database
    kb, document, _ = seed(factory)
    entered, release = Event(), Event()
    externals = Externals()
    service = deletion(factory, externals)
    with ThreadPoolExecutor(2) as pool:
        claim = pool.submit(
            claim_processing_job, document,
            session_factory=pausing_factory(engine, entered, release),
        )
        assert entered.wait(5)
        delete = pool.submit(service.delete, kb, document)
        try:
            assert_database_wait(factory)
            assert not delete.done()
        finally:
            release.set()
        assert claim.result(10) is not None
        with pytest.raises(DeletionProcessingConflict):
            delete.result(10)
    with factory() as session:
        retained = session.get(Document, document)
        assert retained.status == "Processing" and not retained.deletion_pending
    assert externals.calls == []


def test_postgres_deletion_intent_wins_and_claim_cannot_start(database):
    factory, engine = database
    kb, document, _ = seed(factory)
    entered, release = Event(), Event()
    externals = Externals()
    service = deletion(pausing_factory(engine, entered, release), externals)
    with ThreadPoolExecutor(2) as pool:
        delete = pool.submit(service.delete, kb, document)
        assert entered.wait(5)  # Marker is uncommitted; the job lock is already held.
        claim = pool.submit(claim_processing_job, document, session_factory=factory)
        try:
            assert_database_wait(factory)
            assert not claim.done()
        finally:
            release.set()
        delete.result(10)
        assert claim.result(10) is None
    with factory() as session:
        assert session.get(Document, document) is None


@pytest.mark.parametrize("operation", ["chunks", "vectors"])
def test_postgres_worker_waiting_behind_cleanup_cannot_recreate_data(database, operation):
    factory, _ = database
    kb, document, job_id = seed(factory, "Ready")
    externals = Externals()
    externals.release.clear()
    service = deletion(factory, externals)
    vector_calls = []

    class Vectors:
        def upsert(self, vectors):
            vector_calls.append(vectors)

    processor = StoredDocumentProcessor(session_factory=factory, vector_store=Vectors())
    old_claim = ClaimedProcessingJob(job_id, document, 1, utc_now() + timedelta(minutes=5))
    vector = ChunkVector(uuid4(), [0.1] * 1024, uuid4(), document, kb, 0, "model", "revision")
    started = Event()

    def stale_write():
        started.set()
        if operation == "chunks":
            processor.persist(old_claim, prepare_chunks(document, "must never be recreated"))
        else:
            processor.upsert_vectors(old_claim, [vector])

    with ThreadPoolExecutor(2) as pool:
        delete = pool.submit(service.delete, kb, document)
        assert externals.entered.wait(5)
        write = pool.submit(stale_write)
        assert started.wait(5)
        try:
            assert_database_wait(factory)
            assert not write.done()
        finally:
            externals.release.set()
        delete.result(10)
        with pytest.raises((DocumentProcessingError, VectorStoreError)):
            write.result(10)
    assert vector_calls == []
    with factory() as session:
        assert session.get(Document, document) is None
        assert list(session.scalars(select(DocumentChunk))) == []


def test_postgres_inflight_vector_write_holds_lock_until_acknowledged(database):
    factory, _ = database
    kb, document, job_id = seed(factory, "Processing")
    entered, release = Event(), Event()
    externals = Externals()

    class Vectors:
        def upsert(self, _vectors):
            entered.set()
            assert release.wait(10)

    processor = StoredDocumentProcessor(session_factory=factory, vector_store=Vectors())
    claim = ClaimedProcessingJob(job_id, document, 1, utc_now() + timedelta(minutes=5))
    vector = ChunkVector(uuid4(), [0.1] * 1024, uuid4(), document, kb, 0, "model", "revision")
    with ThreadPoolExecutor(2) as pool:
        write = pool.submit(processor.upsert_vectors, claim, [vector])
        assert entered.wait(5)
        with factory() as session:
            assert session.get(DocumentProcessingJob, job_id).vector_write_token is not None
        delete = pool.submit(deletion(factory, externals).delete, kb, document)
        try:
            assert_database_wait(factory)
            assert not delete.done()
        finally:
            release.set()
        write.result(10)
        with pytest.raises(DeletionProcessingConflict):
            delete.result(10)
    assert externals.calls == []


@pytest.mark.parametrize("second_request", ["delete", "recovery"])
def test_postgres_duplicate_cleanup_is_serialized(database, second_request):
    factory, _ = database
    kb, document, _ = seed(factory, "Ready")
    externals = Externals()
    externals.release.clear()
    service = deletion(factory, externals)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(service.delete, kb, document)
        assert externals.entered.wait(5)
        second = pool.submit(service.delete, kb, document) if second_request == "delete" else (
            pool.submit(service.resume, document)
        )
        try:
            assert_database_wait(factory)
            assert not second.done()
        finally:
            externals.release.set()
        first.result(10)
        if second_request == "delete":
            with pytest.raises(DeletionDocumentNotFound):
                second.result(10)
        else:
            assert second.result(10)
    assert [call[0] for call in externals.calls] == ["delete_vectors", "delete_object"]


def deletion_migration():
    path = Path(__file__).resolve().parents[1] / (
        "alembic/versions/20261107_01_add_document_deletion.py"
    )
    spec = spec_from_file_location("deletion_migration", path)
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_postgres_downgrade_locks_before_check_until_column_drop(database, monkeypatch):
    factory, engine = database
    _, document, _ = seed(factory)
    schema = engine.get_execution_options()["schema_translate_map"][None]
    checked, release = Event(), Event()
    migration = deletion_migration()

    def downgrade():
        with engine.connect() as connection, connection.begin():
            connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
            monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))

            def pause_after_check(_conn, _cursor, statement, *_args):
                if "SELECT EXISTS(SELECT 1 FROM documents WHERE deletion_requested_at" in statement:
                    checked.set()
                    assert release.wait(10)

            event.listen(connection, "after_cursor_execute", pause_after_check)
            migration.downgrade()

    def create_intent():
        with engine.begin() as connection:
            connection.execute(text(
                f'UPDATE "{schema}".documents SET deletion_requested_at = CURRENT_TIMESTAMP '
                "WHERE id = :id"
            ), {"id": document})

    with ThreadPoolExecutor(2) as pool:
        drop = pool.submit(downgrade)
        assert checked.wait(5)
        intent = pool.submit(create_intent)
        try:
            assert_database_wait(factory)
            assert not intent.done()
        finally:
            release.set()
        drop.result(10)
        # The concurrent UPDATE cannot commit intent into a column being removed.
        with pytest.raises(ProgrammingError) as error:
            intent.result(10)
        assert error.value.orig.sqlstate == "42703"
    with engine.connect() as connection:
        assert not connection.scalar(text(
            "SELECT EXISTS(SELECT 1 FROM information_schema.columns "
            "WHERE table_schema=:schema AND table_name='documents' "
            "AND column_name='deletion_requested_at')"
        ), {"schema": schema})


def test_postgres_downgrade_refuses_already_pending_intent(database, monkeypatch):
    factory, engine = database
    _, document, _ = seed(factory)
    with factory() as session:
        session.get(Document, document).deletion_requested_at = utc_now()
        session.commit()
    schema = engine.get_execution_options()["schema_translate_map"][None]
    migration = deletion_migration()
    with pytest.raises(RuntimeError, match="Finish pending document deletions"):
        with engine.connect() as connection, connection.begin():
            connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
            monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
            migration.downgrade()
    with factory() as session:
        assert session.get(Document, document).deletion_pending


def test_postgres_control_unlocked_check_can_discard_concurrent_intent(database):
    """Reproduce the original TOCTOU with independent real PostgreSQL transactions."""
    factory, engine = database
    _, document, _ = seed(factory)
    schema = engine.get_execution_options()["schema_translate_map"][None]
    with engine.begin() as first:
        assert not first.scalar(text(
            f'SELECT EXISTS(SELECT 1 FROM "{schema}".documents '
            "WHERE deletion_requested_at IS NOT NULL)"
        ))
        with factory() as concurrent:
            concurrent.get(Document, document).deletion_requested_at = utc_now()
            concurrent.commit()  # The original SELECT lock does not prevent this.
        assert first.scalar(text(
            f'SELECT deletion_requested_at IS NOT NULL FROM "{schema}".documents WHERE id=:id'
        ), {"id": document})
        first.execute(text(f'ALTER TABLE "{schema}".documents DROP COLUMN deletion_requested_at'))


def test_postgres_vector_intent_downgrade_refuses_unresolved_write(database, monkeypatch):
    factory, engine = database
    _, _, job_id = seed(factory)
    with factory() as session:
        job = session.get(DocumentProcessingJob, job_id)
        job.vector_write_token = uuid4()
        job.vector_write_started_at = utc_now()
        session.commit()
    schema = engine.get_execution_options()["schema_translate_map"][None]
    path = Path(__file__).resolve().parents[1] / (
        "alembic/versions/20261108_01_track_vector_write_intent.py"
    )
    spec = spec_from_file_location("vector_intent_migration", path)
    migration = module_from_spec(spec)
    spec.loader.exec_module(migration)
    with pytest.raises(RuntimeError, match="Resolve pending vector writes"):
        with engine.connect() as connection, connection.begin():
            connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
            monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
            migration.downgrade()
    with factory() as session:
        assert session.get(DocumentProcessingJob, job_id).vector_write_token is not None


def test_postgres_vector_intent_upgrade_quarantines_legacy_ambiguous_work(database, monkeypatch):
    factory, engine = database
    _, _, processing = seed(factory, "Processing")
    _, _, failed = seed(factory, "Failed")
    _, _, ready = seed(factory, "Ready")
    _, _, queued = seed(factory)
    with factory() as session:
        session.get(DocumentProcessingJob, failed).failure_code = "VECTOR_PERSISTENCE_FAILED"
        session.commit()
    schema = engine.get_execution_options()["schema_translate_map"][None]
    path = Path(__file__).resolve().parents[1] / (
        "alembic/versions/20261108_01_track_vector_write_intent.py"
    )
    spec = spec_from_file_location("vector_intent_upgrade", path)
    migration = module_from_spec(spec)
    spec.loader.exec_module(migration)
    with engine.connect() as connection, connection.begin():
        connection.execute(text(f'SET LOCAL search_path TO "{schema}"'))
        connection.execute(text(
            "ALTER TABLE document_processing_jobs DROP COLUMN vector_write_started_at, "
            "DROP COLUMN vector_write_token"
        ))
        monkeypatch.setattr(migration, "op", Operations(MigrationContext.configure(connection)))
        migration.upgrade()
    with factory() as session:
        for job_id in (processing, failed):
            job = session.get(DocumentProcessingJob, job_id)
            assert job.vector_write_token == job_id and job.vector_write_started_at is not None
        for job_id in (ready, queued):
            assert session.get(DocumentProcessingJob, job_id).vector_write_token is None
