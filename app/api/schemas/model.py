from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class ModelVersionInfo(BaseModel):
    id: UUID
    model_type: str
    stage: str
    roc_auc: float | None
    gini: float | None
    f1_score: float | None
    stability_score: float | None
    is_stable: bool
    created_at: datetime
    promoted_at: datetime | None


class ModelCard(ModelVersionInfo):
    pr_auc: float | None
    ks_stat: float | None
    brier_score: float | None
    roc_auc_std: float | None
    trend_slope: float | None
    hyperparams: dict | None
    feature_importance: dict | None
    mlflow_model_uri: str | None


class PromoteResponse(BaseModel):
    model_version_id: UUID
    stage: str
    promoted_at: datetime
