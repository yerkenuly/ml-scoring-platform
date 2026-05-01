import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiments.id")
    )
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)
    mlflow_version: Mapped[int | None] = mapped_column(Integer)
    mlflow_model_uri: Mapped[str | None] = mapped_column(String(500))
    stage: Mapped[str] = mapped_column(String(30), default="staging")

    roc_auc: Mapped[float | None] = mapped_column(Float)
    gini: Mapped[float | None] = mapped_column(Float)
    f1_score: Mapped[float | None] = mapped_column(Float)
    pr_auc: Mapped[float | None] = mapped_column(Float)
    ks_stat: Mapped[float | None] = mapped_column(Float)
    brier_score: Mapped[float | None] = mapped_column(Float)

    stability_score: Mapped[float | None] = mapped_column(Float)
    roc_auc_std: Mapped[float | None] = mapped_column(Float)
    trend_slope: Mapped[float | None] = mapped_column(Float)
    is_stable: Mapped[bool | None] = mapped_column(Boolean, default=False)

    hyperparams_json: Mapped[dict | None] = mapped_column(JSON)
    feature_names: Mapped[list | None] = mapped_column(ARRAY(String))
    feature_importance: Mapped[dict | None] = mapped_column(JSON)

    dataset_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id")
    )
    dataset_hash: Mapped[str | None] = mapped_column(String(64))

    promoted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    archived_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
