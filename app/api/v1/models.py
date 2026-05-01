from datetime import datetime, timezone
from uuid import UUID

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.schemas.model import ModelCard, ModelVersionInfo, PromoteResponse
from app.dependencies import get_db, verify_api_key
from app.exceptions import NotFoundError

router = APIRouter(prefix="/models", tags=["models"])


@router.get("", response_model=list[ModelVersionInfo])
async def list_models(
    stage: str | None = None,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    from app.db.models.model_version import ModelVersion
    from sqlalchemy import select

    query = select(ModelVersion)
    if stage:
        query = query.where(ModelVersion.stage == stage)
    result = await db.execute(query.order_by(ModelVersion.created_at.desc()))
    versions = result.scalars().all()

    return [
        ModelVersionInfo(
            id=v.id,
            model_type=v.model_type,
            stage=v.stage,
            roc_auc=v.roc_auc,
            gini=v.gini,
            f1_score=v.f1_score,
            stability_score=v.stability_score,
            is_stable=v.is_stable or False,
            created_at=v.created_at,
            promoted_at=v.promoted_at,
        )
        for v in versions
    ]


@router.get("/{model_id}", response_model=ModelCard)
async def get_model(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    from app.db.models.model_version import ModelVersion
    from sqlalchemy import select

    result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_id))
    v = result.scalar_one_or_none()
    if not v:
        raise NotFoundError(f"Model {model_id} not found")

    return ModelCard(
        id=v.id,
        model_type=v.model_type,
        stage=v.stage,
        roc_auc=v.roc_auc,
        gini=v.gini,
        f1_score=v.f1_score,
        pr_auc=v.pr_auc,
        ks_stat=v.ks_stat,
        brier_score=v.brier_score,
        stability_score=v.stability_score,
        roc_auc_std=v.roc_auc_std,
        trend_slope=v.trend_slope,
        is_stable=v.is_stable or False,
        hyperparams=v.hyperparams_json,
        feature_importance=v.feature_importance,
        mlflow_model_uri=v.mlflow_model_uri,
        created_at=v.created_at,
        promoted_at=v.promoted_at,
    )


@router.post("/{model_id}/promote", response_model=PromoteResponse)
async def promote_model(
    model_id: UUID,
    db: AsyncSession = Depends(get_db),
    _: str = Depends(verify_api_key),
):
    from app.db.models.model_version import ModelVersion
    from sqlalchemy import select, update

    result = await db.execute(select(ModelVersion).where(ModelVersion.id == model_id))
    v = result.scalar_one_or_none()
    if not v:
        raise NotFoundError(f"Model {model_id} not found")

    await db.execute(
        update(ModelVersion)
        .where(ModelVersion.stage == "production")
        .values(stage="archived", archived_at=datetime.now(timezone.utc))
    )

    v.stage = "production"
    v.promoted_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(v)

    return PromoteResponse(model_version_id=v.id, stage=v.stage, promoted_at=v.promoted_at)
