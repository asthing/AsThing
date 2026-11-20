"""Real PostgreSQL parent/FK coordination; only a disposable database is supported."""

import os
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from uuid import uuid4

import pytest
import test_document_deletion_postgres as postgres_tests
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker
from test_document_deletion_postgres import (
    Externals,
    assert_database_wait,
    seed,
)

from app.document_deletion import DocumentDeletionService
from app.document_models import Document
from app.knowledge_base_deletion import KnowledgeBaseDeletionConflict, KnowledgeBaseDeletionService
from app.knowledge_bases import KnowledgeBase

pytestmark = pytest.mark.skipif(
    not os.getenv("DELETION_TEST_DATABASE_URL"), reason="requires disposable PostgreSQL stack"
)


@pytest.fixture
def database():
    yield from postgres_tests.database.__wrapped__()


def service(factory, externals):
    return KnowledgeBaseDeletionService(
        session_factory=factory,
        documents=DocumentDeletionService(
            session_factory=factory, storage_provider=lambda: externals,
            vector_store_provider=lambda: externals,
        ),
    )


def insert_document(factory, kb):
    with factory() as session:
        document = Document(
            knowledge_base_id=kb, original_filename="new.txt", file_type="txt",
            file_size_bytes=1, storage_object_key=f"test/{uuid4()}", status="Uploaded",
        )
        session.add(document)
        session.commit()
        return document.id


def test_new_upload_before_parent_lock_prevents_final_kb_removal(database):
    factory, _ = database
    kb, original, _ = seed(factory, "Ready")
    externals = Externals()
    externals.release.clear()
    deletion = service(factory, externals)
    with ThreadPoolExecutor(2) as pool:
        pending = pool.submit(deletion.delete, kb)
        assert externals.entered.wait(5)
        uploaded = insert_document(factory, kb)
        externals.release.set()
        with pytest.raises(KnowledgeBaseDeletionConflict):
            pending.result(10)
    with factory() as session:
        assert session.get(Document, original) is None
        assert session.get(Document, uploaded) is not None
        assert session.get(KnowledgeBase, kb) is not None


def test_parent_lock_wins_and_fk_prevents_late_upload_orphan(database):
    factory, engine = database
    with factory() as session:
        kb = KnowledgeBase(name="empty", normalized_name="empty")
        session.add(kb)
        session.commit()
        kb_id = kb.id
    entered, release = Event(), Event()

    class PausedParentLock(Session):
        def scalar(self, statement, *args, **kwargs):
            result = super().scalar(statement, *args, **kwargs)
            if getattr(statement, "_for_update_arg", None) is not None:
                entered.set()
                assert release.wait(10)
            return result

    deletion = service(sessionmaker(engine, class_=PausedParentLock), Externals())
    with ThreadPoolExecutor(2) as pool:
        pending = pool.submit(deletion.delete, kb_id)
        assert entered.wait(5)
        upload = pool.submit(insert_document, factory, kb_id)
        try:
            assert_database_wait(factory)
            assert not upload.done()
        finally:
            release.set()
        pending.result(10)
        with pytest.raises(IntegrityError) as error:
            upload.result(10)
        assert error.value.orig.sqlstate == "23503"
    with factory() as session:
        assert session.get(KnowledgeBase, kb_id) is None
        assert not list(session.scalars(select(Document)))


def test_concurrent_knowledge_base_deletes_share_idempotent_document_cleanup(database):
    factory, _ = database
    kb, _, _ = seed(factory, "Ready")
    externals = Externals()
    externals.release.clear()
    deletion = service(factory, externals)
    with ThreadPoolExecutor(2) as pool:
        first = pool.submit(deletion.delete, kb)
        assert externals.entered.wait(5)
        second = pool.submit(deletion.delete, kb)
        try:
            assert_database_wait(factory)
        finally:
            externals.release.set()
        first.result(10)
        second.result(10)
    assert [call[0] for call in externals.calls] == ["delete_vectors", "delete_object"]
    with factory() as session:
        assert session.get(KnowledgeBase, kb) is None
