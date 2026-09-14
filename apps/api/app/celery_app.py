import os

from celery import Celery

celery_app = Celery(
    "asthing",
    broker=os.environ["REDIS_URL"],
    include=["app.tasks"],
)
celery_app.conf.update(
    broker_transport_options={
        "max_retries": 0,
        "socket_connect_timeout": 1.0,
        "socket_timeout": 1.0,
    },
    task_acks_late=True,
    task_ignore_result=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
)
