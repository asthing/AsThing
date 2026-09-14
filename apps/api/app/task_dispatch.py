from queue import Empty, Queue
from threading import Thread
from typing import Protocol
from uuid import UUID

PROCESS_DOCUMENT_TASK_NAME = "documents.process"
TASK_DISPATCH_TIMEOUT_SECONDS = 1.0


class ProcessingTaskDispatchError(TimeoutError):
    """Raised when task publication fails or exceeds the request-safe deadline."""


class ProcessingTaskDispatcher(Protocol):
    def dispatch(self, job_id: UUID, document_id: UUID) -> str:
        """Send one processing task and return the broker task ID."""
        ...


class CeleryProcessingTaskDispatcher:
    def dispatch(self, job_id: UUID, document_id: UUID) -> str:
        # Keep Celery and worker modules out of the normal API import path.
        from app.celery_app import celery_app

        outcome: Queue[tuple[str | None, bool]] = Queue(maxsize=1)

        def publish() -> None:
            try:
                result = celery_app.send_task(
                    PROCESS_DOCUMENT_TASK_NAME,
                    args=[str(document_id)],
                    task_id=str(job_id),
                    retry=False,
                    ignore_result=True,
                )
            except Exception:
                outcome.put((None, False))
                return
            outcome.put((str(result.id), True))

        Thread(
            target=publish,
            name=f"document-task-dispatch-{job_id}",
            daemon=True,
        ).start()
        try:
            task_id, succeeded = outcome.get(timeout=TASK_DISPATCH_TIMEOUT_SECONDS)
        except Empty:
            raise ProcessingTaskDispatchError from None
        if not succeeded or task_id is None:
            raise ProcessingTaskDispatchError
        return task_id


def get_processing_task_dispatcher() -> ProcessingTaskDispatcher:
    return CeleryProcessingTaskDispatcher()
