from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.monitoring import (
    DriftCheckResponse,
    DriftReport,
    LabelSubmitRequest,
    LabelSubmitResponse,
)
from app.dependencies import get_db, verify_api_key
from app.exceptions import NotFoundError

router = APIRouter(prefix="/monitoring", tags=["monitoring"])


@router.get("/drift", response_model=list[DriftReport])
async def get_drift_reports(
    model_version_id: UUID | None = None,
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    from app.db.models.drift_report import DriftReport as DriftReportModel
    from sqlalchemy import select

    query = select(DriftReportModel).order_by(DriftReportModel.created_at.desc()).limit(limit)
    if model_version_id:
        query = query.where(DriftReportModel.model_version_id == model_version_id)

    result = await db.execute(query)
    reports = result.scalars().all()

    return [
        DriftReport(
            id=r.id,
            model_version_id=r.model_version_id,
            check_type=r.check_type,
            overall_drift_flag=r.overall_drift_flag,
            drift_score=r.drift_score,
            feature_drift=r.feature_drift_json,
            performance_delta=r.performance_delta,
            triggered_retraining=r.triggered_retraining,
            created_at=r.created_at,
        )
        for r in reports
    ]


@router.post("/check", response_model=DriftCheckResponse, status_code=202)
async def trigger_drift_check(
    model_version_id: UUID | None = None,
    _: str = Depends(verify_api_key),
):
    from app.worker.tasks.drift_tasks import run_on_demand_drift_check

    task = run_on_demand_drift_check.delay(str(model_version_id) if model_version_id else None)
    return DriftCheckResponse(job_id=task.id, status="queued")


@router.post("/labels", response_model=LabelSubmitResponse)
async def submit_labels(
    body: LabelSubmitRequest,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    from app.db.models.prediction_log import PredictionLog
    from sqlalchemy import select, update

    updated = 0
    for record in body.records:
        pred_id = record.get("prediction_id")
        true_label = record.get("true_label")
        if pred_id and true_label is not None:
            await db.execute(
                update(PredictionLog)
                .where(PredictionLog.id == pred_id)
                .values(true_label=true_label)
            )
            updated += 1

    await db.commit()
    return LabelSubmitResponse(updated_count=updated)
