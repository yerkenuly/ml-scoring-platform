from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.dependencies import get_db, verify_api_key

router = APIRouter(prefix="/admin", tags=["admin"])


@router.post("/retrain")
async def trigger_retrain(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    from app.worker.tasks.training_tasks import run_training_pipeline

    task = run_training_pipeline.delay(str(experiment_id))
    return {"job_id": task.id, "status": "queued"}


@router.get("/jobs")
async def list_jobs(
    limit: int = 20,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    from app.db.models.job import Job
    from sqlalchemy import select

    result = await db.execute(
        select(Job).order_by(Job.created_at.desc()).limit(limit)
    )
    jobs = result.scalars().all()
    return [
        {
            "id": str(j.id),
            "job_type": j.job_type,
            "status": j.status,
            "progress": j.progress,
            "started_at": j.started_at,
            "completed_at": j.completed_at,
        }
        for j in jobs
    ]
