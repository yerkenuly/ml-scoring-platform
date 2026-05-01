from celery import Celery

from app.config import settings

celery_app = Celery(
    "ml_scoring",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.worker.tasks.training_tasks",
        "app.worker.tasks.drift_tasks",
        "app.worker.tasks.cleanup_tasks",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    beat_schedule_filename="/tmp/celerybeat-schedule",
)

celery_app.config_from_object("app.worker.schedules")
