import joblib
import os
import tempfile
from dataclasses import dataclass

import mlflow
import mlflow.sklearn

from app.config import settings
from app.core.pipeline.model_selector import ModelCandidate


@dataclass
class RegisteredModelVersion:
    mlflow_run_id: str
    mlflow_model_uri: str
    model_type: str
    metrics: dict


class RegistryManager:
    def __init__(self):
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)

    def register(
        self,
        candidate: ModelCandidate,
        experiment_id: str,
        dataset_hash: str | None = None,
    ) -> RegisteredModelVersion:
        mlflow.set_experiment(settings.mlflow_experiment_name)

        with mlflow.start_run(run_name=f"{candidate.fitted_model.model_type}_{experiment_id[:8]}") as run:
            e = candidate.holdout_eval or candidate.eval_result
            s = candidate.stability_result

            metrics = {
                "roc_auc": e.roc_auc,
                "gini": e.gini,
                "f1": e.f1,
                "pr_auc": e.pr_auc,
                "ks_stat": e.ks_stat,
                "brier_score": e.brier_score,
                "stability_score": s.stability_score,
                "roc_auc_std": s.roc_auc_std,
                "trend_slope": s.trend_slope,
            }
            mlflow.log_metrics(metrics)
            mlflow.log_params(candidate.fitted_model.params)
            mlflow.set_tags({
                "model_type": candidate.fitted_model.model_type,
                "experiment_id": experiment_id,
                "is_stable": str(s.is_stable),
                "dataset_hash": dataset_hash or "",
            })

            with tempfile.TemporaryDirectory() as tmpdir:
                pipeline_path = os.path.join(tmpdir, "feature_pipeline.pkl")
                joblib.dump(candidate.feature_pipeline, pipeline_path)
                mlflow.log_artifact(pipeline_path, artifact_path="feature_pipeline")

            model_uri = mlflow.sklearn.log_model(
                sk_model=candidate.fitted_model.model.get_model(),
                artifact_path="model",
                registered_model_name=settings.mlflow_model_registry_name,
            ).model_uri

        return RegisteredModelVersion(
            mlflow_run_id=run.info.run_id,
            mlflow_model_uri=model_uri,
            model_type=candidate.fitted_model.model_type,
            metrics=metrics,
        )
