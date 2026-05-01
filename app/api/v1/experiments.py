from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.experiment import ExperimentCreate, ExperimentResponse, ExperimentResult
from app.dependencies import get_db, verify_api_key
from app.exceptions import NotFoundError

router = APIRouter(prefix="/experiments", tags=["experiments"])


@router.post("", response_model=ExperimentResponse, status_code=202)
async def create_experiment(
    body: ExperimentCreate,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    from app.db.models.experiment import Experiment
    from app.worker.tasks.training_tasks import run_training_pipeline

    experiment = Experiment(
        name=body.experiment_name,
        dataset_id=body.dataset_id,
        model_types=body.model_types,
        cv_strategy=body.cv_strategy,
        n_trials=body.n_trials,
        target_metric=body.target_metric,
        status="queued",
        config_json=body.model_dump(),
    )
    db.add(experiment)
    await db.commit()
    await db.refresh(experiment)

    task = run_training_pipeline.delay(str(experiment.id))

    return ExperimentResponse(
        experiment_id=experiment.id,
        job_id=task.id,
        status="queued",
        created_at=experiment.created_at,
    )


@router.get("/{experiment_id}", response_model=ExperimentResult)
async def get_experiment(
    experiment_id: UUID,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    from app.db.models.experiment import Experiment
    from sqlalchemy import select

    result = await db.execute(select(Experiment).where(Experiment.id == experiment_id))
    exp = result.scalar_one_or_none()
    if not exp:
        raise NotFoundError(f"Experiment {experiment_id} not found")

    results = exp.results_json or {}
    duration = None
    if exp.started_at and exp.completed_at:
        duration = (exp.completed_at - exp.started_at).total_seconds()

    return ExperimentResult(
        experiment_id=exp.id,
        status=exp.status,
        best_model_type=results.get("best_model_type"),
        best_roc_auc=results.get("best_roc_auc"),
        best_gini=results.get("best_gini"),
        best_f1=results.get("best_f1"),
        best_stability_score=results.get("best_stability_score"),
        all_candidates=results.get("all_candidates"),
        duration_seconds=duration,
        started_at=exp.started_at,
        completed_at=exp.completed_at,
        error_message=exp.error_message,
    )
