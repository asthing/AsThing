import importlib
import sys
from types import ModuleType


def test_worker_app_registers_document_task_module(monkeypatch) -> None:
    captured_options: dict[str, object] = {}
    captured_configuration: dict[str, object] = {}

    class FakeConfiguration:
        def update(self, **options: object) -> None:
            captured_configuration.update(options)

    class FakeCelery:
        def __init__(self, _name: str, **options: object) -> None:
            captured_options.update(options)
            self.conf = FakeConfiguration()

    fake_celery = ModuleType("celery")
    fake_celery.Celery = FakeCelery
    monkeypatch.setitem(sys.modules, "celery", fake_celery)
    monkeypatch.setenv("REDIS_URL", "redis://example.invalid:6379/0")
    monkeypatch.delitem(sys.modules, "app.celery_app", raising=False)

    try:
        importlib.import_module("app.celery_app")
    finally:
        sys.modules.pop("app.celery_app", None)

    assert captured_options["include"] == ["app.tasks"]
    assert "backend" not in captured_options
    assert captured_configuration["task_ignore_result"] is True
    assert captured_configuration["worker_prefetch_multiplier"] == 1
    assert captured_configuration["broker_transport_options"] == {
        "max_retries": 3,
        "socket_connect_timeout": 1.0,
        "socket_timeout": 1.0,
    }
