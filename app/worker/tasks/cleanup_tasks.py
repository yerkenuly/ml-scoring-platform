import logging
from datetime import datetime, timedelta, timezone

from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.worker.tasks.cleanup_tasks.cleanup_old_prediction_logs")
def cleanup_old_prediction_logs(days_to_keep: int = 90) -> dict:
    from sqlalchemy import create_engine, delete
    from sqlalchemy.orm import Session

    from app.config import settings
    from app.db.models.prediction_log import PredictionLog

    cutoff = datetime.now(timezone.utc) - timedelta(days=days_to_keep)
    engine = create_engine(settings.sync_database_url)

    with Session(engine) as session:
        result = session.execute(
            delete(PredictionLog).where(
                PredictionLog.created_at < cutoff,
                PredictionLog.true_label.isnot(None),
            )
        )
        session.commit()
        deleted = result.rowcount

    logger.info("Cleaned up %d old prediction logs (older than %d days)", deleted, days_to_keep)
    return {"deleted_count": deleted}
