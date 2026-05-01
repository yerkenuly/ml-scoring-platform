import logging
from datetime import datetime, timezone

from app.worker.celery_app import celery_app

logger = logging.getLogger(__name__)


@celery_app.task(bind=True, max_retries=3, acks_late=True, name="app.worker.tasks.training_tasks.run_training_pipeline")
def run_training_pipeline(self, experiment_id: str) -> dict:
    from sqlalchemy import create_engine
    from sqlalchemy.orm import Session

    from app.config import settings
    from app.core.pipeline.orchestrator import PipelineOrchestrator
    from app.db.models.experiment import Experiment
    from app.db.models.model_version import ModelVersion

    engine = create_engine(settings.sync_database_url)

    with Session(engine) as session:
        exp = session.get(Experiment, experiment_id)
        if not exp:
            logger.error("Experiment %s not found", experiment_id)
            return {"status": "error", "reason": "experiment not found"}

        dataset_id = exp.dataset_id
        config = exp.config_json or {}

        from app.db.models.dataset import Dataset
        dataset = session.get(Dataset, str(dataset_id))
        if not dataset:
            logger.error("Dataset %s not found", dataset_id)
            return {"status": "error", "reason": "dataset not found"}

        exp.status = "running"
        exp.started_at = datetime.now(timezone.utc)
        session.commit()

    try:
        orchestrator = PipelineOrchestrator()
        result = orchestrator.run(
            experiment_id=experiment_id,
            file_path=dataset.file_path,
            target_column=dataset.target_column,
            model_types=exp.model_types or ["xgboost", "lightgbm", "catboost"],
            n_trials=exp.n_trials,
            cv_strategy=exp.cv_strategy or "auto",
            date_column=dataset.date_column,
        )

        best = result.best_candidate
        holdout = best.holdout_eval
        stability = best.stability_result
        registered = result.registered_version

        with Session(engine) as session:
            mv = ModelVersion(
                experiment_id=experiment_id,
                model_type=best.fitted_model.model_type,
                mlflow_model_uri=registered.mlflow_model_uri,
                stage="staging",
                roc_auc=holdout.roc_auc if holdout else best.eval_result.roc_auc,
                gini=holdout.gini if holdout else best.eval_result.gini,
                f1_score=holdout.f1 if holdout else best.eval_result.f1,
                pr_auc=holdout.pr_auc if holdout else best.eval_result.pr_auc,
                ks_stat=holdout.ks_stat if holdout else best.eval_result.ks_stat,
                brier_score=holdout.brier_score if holdout else best.eval_result.brier_score,
                stability_score=stability.stability_score,
                roc_auc_std=stability.roc_auc_std,
                trend_slope=stability.trend_slope,
                is_stable=stability.is_stable,
                hyperparams_json=best.fitted_model.params,
                feature_names=best.fitted_model.feature_names,
                feature_importance={
                    k: float(v)
                    for k, v in best.fitted_model.model.get_feature_importance().items()
                },
                dataset_id=dataset_id,
            )
            session.add(mv)

            all_candidates_summary = [
                {
                    "model_type": c.fitted_model.model_type,
                    "roc_auc": c.eval_result.roc_auc,
                    "gini": c.eval_result.gini,
                    "f1": c.eval_result.f1,
                    "stability_score": c.stability_result.stability_score,
                    "is_stable": c.stability_result.is_stable,
                }
                for c in result.all_candidates
            ]

            exp = session.get(Experiment, experiment_id)
            exp.status = "completed"
            exp.completed_at = datetime.now(timezone.utc)
            exp.results_json = {
                "best_model_type": best.fitted_model.model_type,
                "best_roc_auc": holdout.roc_auc if holdout else best.eval_result.roc_auc,
                "best_gini": holdout.gini if holdout else best.eval_result.gini,
                "best_f1": holdout.f1 if holdout else best.eval_result.f1,
                "best_stability_score": stability.stability_score,
                "all_candidates": all_candidates_summary,
            }
            session.commit()

        logger.info("Experiment %s completed. Best: %s ROC-AUC=%.4f", experiment_id, best.fitted_model.model_type, holdout.roc_auc if holdout else 0)
        return {"status": "completed", "model_type": best.fitted_model.model_type}

    except Exception as exc:
        logger.exception("Training failed for experiment %s", experiment_id)
        from sqlalchemy.orm import Session
        with Session(engine) as session:
            exp = session.get(Experiment, experiment_id)
            if exp:
                exp.status = "failed"
                exp.error_message = str(exc)
                exp.completed_at = datetime.now(timezone.utc)
                session.commit()
        try:
            raise self.retry(exc=exc, countdown=60)
        except self.MaxRetriesExceededError:
            return {"status": "failed", "error": str(exc)}
