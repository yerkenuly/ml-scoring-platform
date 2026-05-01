import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    dataset_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("datasets.id"), nullable=False
    )
    mlflow_run_id: Mapped[str | None] = mapped_column(String(255))
    model_types: Mapped[list | None] = mapped_column(ARRAY(String))
    cv_strategy: Mapped[str | None] = mapped_column(String(50))
    n_trials: Mapped[int] = mapped_column(Integer, default=50)
    target_metric: Mapped[str] = mapped_column(String(50), default="roc_auc")
    status: Mapped[str] = mapped_column(String(30), default="queued")
    config_json: Mapped[dict | None] = mapped_column(JSON)
    results_json: Mapped[dict | None] = mapped_column(JSON)
    error_message: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
