from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class DriftReport(BaseModel):
    id: UUID
    model_version_id: UUID
    check_type: str
    overall_drift_flag: bool
    drift_score: float | None
    feature_drift: dict | None
    performance_delta: float | None
    triggered_retraining: bool
    created_at: datetime


class LabelSubmitRequest(BaseModel):
    records: list[dict]  # [{prediction_id: str, true_label: int}]


class LabelSubmitResponse(BaseModel):
    updated_count: int


class DriftCheckResponse(BaseModel):
    job_id: str
    status: str
