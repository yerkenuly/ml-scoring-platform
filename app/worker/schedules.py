from celery.schedules import crontab

from app.config import settings

beat_schedule = {
    "periodic-drift-check": {
        "task": "app.worker.tasks.drift_tasks.run_periodic_drift_check",
        "schedule": settings.drift_check_interval_seconds,
    },
    "daily-cleanup": {
        "task": "app.worker.tasks.cleanup_tasks.cleanup_old_prediction_logs",
        "schedule": crontab(hour=2, minute=0),
    },
}
