import time
from sys import modules
from threading import Event
from types import ModuleType, SimpleNamespace
from uuid import uuid4

import pytest

from app import task_dispatch
from app.task_dispatch import CeleryProcessingTaskDispatcher


def test_celery_dispatch_uses_job_id_as_stable_task_id(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def send_task(name: str, **options: object) -> SimpleNamespace:
        calls.append({"name": name, **options})
        return SimpleNamespace(id=options["task_id"])

    celery_module = ModuleType("app.celery_app")
    celery_module.celery_app = SimpleNamespace(send_task=send_task)
    monkeypatch.setitem(modules, "app.celery_app", celery_module)
    job_id = uuid4()
    document_id = uuid4()

    task_id = CeleryProcessingTaskDispatcher().dispatch(job_id, document_id)

    assert task_id == str(job_id)
    assert calls == [
        {
            "name": "documents.process",
            "args": [str(document_id)],
            "task_id": str(job_id),
            "retry": False,
            "ignore_result": True,
        }
    ]


def test_celery_dispatch_has_a_hard_deadline(monkeypatch) -> None:
    release_slow_dispatch = Event()

    def send_task(_name: str, **_options: object) -> SimpleNamespace:
        release_slow_dispatch.wait(timeout=1)
        return SimpleNamespace(id="too-late")

    celery_module = ModuleType("app.celery_app")
    celery_module.celery_app = SimpleNamespace(send_task=send_task)
    monkeypatch.setitem(modules, "app.celery_app", celery_module)
    monkeypatch.setattr(
        task_dispatch, "TASK_DISPATCH_TIMEOUT_SECONDS", 0.01, raising=False
    )

    started = time.perf_counter()
    try:
        with pytest.raises(TimeoutError):
            CeleryProcessingTaskDispatcher().dispatch(uuid4(), uuid4())
    finally:
        release_slow_dispatch.set()

    assert time.perf_counter() - started < 0.2
