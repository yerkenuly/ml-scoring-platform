import logging
from datetime import datetime, timedelta, timezone

from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(name="app.worker.tasks.drift_tasks.run_periodic_drift_check")
def run_periodic_drift_check() -> None:
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session

    from app.config import settings
    from app.core.drift.detector import DriftDetector
    from app.db.models.drift_report import DriftReport
    from app.db.models.model_version import ModelVersion
    from app.db.models.prediction_log import PredictionLog

    engine = create_engine(settings.sync_database_url)
    detector = DriftDetector()

    with Session(engine) as session:
        production_models = session.execute(
            select(ModelVersion).where(ModelVersion.stage == "production")
        ).scalars().all()

        for mv in production_models:
            try:
                cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
                logs = session.execute(
                    select(PredictionLog)
                    .where(
                        PredictionLog.model_version_id == mv.id,
                        PredictionLog.created_at >= cutoff,
                    )
                    .limit(5000)
                ).scalars().all()

                if len(logs) < 50:
                    logger.info("Not enough prediction logs for model %s, skipping drift check", mv.id)
                    continue

                import pandas as pd
                current_df = pd.DataFrame([log.features_json for log in logs])

                from app.db.models.dataset import Dataset
                dataset = session.get(Dataset, str(mv.dataset_id))
                if not dataset:
                    continue

                if dataset.file_path.endswith(".parquet"):
                    reference_df = pd.read_parquet(dataset.file_path)
                else:
                    reference_df = pd.read_csv(dataset.file_path)

                feature_cols = [c for c in current_df.columns if c in reference_df.columns]
                data_report = detector.check_data_drift(
                    reference_df[feature_cols], current_df[feature_cols]
                )

                triggered_retraining = False
                if data_report.overall_drift_flag:
                    logger.warning("Drift detected for model %s, triggering retraining", mv.id)
                    from app.db.models.experiment import Experiment
                    orig_exp = session.get(Experiment, str(mv.experiment_id))
                    if orig_exp:
                        from app.worker.tasks.training_tasks import run_training_pipeline
                        run_training_pipeline.delay(str(orig_exp.id))
                        triggered_retraining = True

                report = DriftReport(
                    model_version_id=mv.id,
                    check_type="data_drift",
                    overall_drift_flag=data_report.overall_drift_flag,
                    drift_score=data_report.drift_score,
                    feature_drift_json=data_report.feature_psi,
                    triggered_retraining=triggered_retraining,
                )
                session.add(report)
                session.commit()

            except Exception as exc:
                logger.exception("Drift check failed for model %s: %s", mv.id, exc)


@celery_app.task(name="app.worker.tasks.drift_tasks.run_on_demand_drift_check")
def run_on_demand_drift_check(model_version_id: str | None = None) -> dict:
    run_periodic_drift_check()
    return {"status": "completed"}
