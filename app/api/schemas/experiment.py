from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, Field


class ExperimentCreate(BaseModel):
    dataset_id: UUID
    experiment_name: str
    model_types: list[str] = Field(
        default=["xgboost", "lightgbm", "catboost", "random_forest", "logistic_regression"]
    )
    n_trials: int = Field(default=50, ge=10, le=200)
    cv_strategy: str = Field(default="auto", pattern="^(auto|stratified_kfold|timeseries_split)$")
    target_metric: str = Field(default="roc_auc")


class ExperimentResponse(BaseModel):
    experiment_id: UUID
    job_id: str
    status: str
    created_at: datetime


class ExperimentResult(BaseModel):
    experiment_id: UUID
    status: str
    best_model_type: str | None
    best_roc_auc: float | None
    best_gini: float | None
    best_f1: float | None
    best_stability_score: float | None
    all_candidates: list[dict] | None
    duration_seconds: float | None
    started_at: datetime | None
    completed_at: datetime | None
    error_message: str | None
