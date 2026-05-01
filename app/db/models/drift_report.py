import uuid
from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class DriftReport(Base):
    __tablename__ = "drift_reports"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    model_version_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("model_versions.id"), nullable=False
    )
    check_type: Mapped[str | None] = mapped_column(String(30))
    overall_drift_flag: Mapped[bool] = mapped_column(Boolean, nullable=False)
    drift_score: Mapped[float | None] = mapped_column(Float)
    feature_drift_json: Mapped[dict | None] = mapped_column(JSON)
    performance_delta: Mapped[float | None] = mapped_column(Float)
    triggered_retraining: Mapped[bool] = mapped_column(Boolean, default=False)
    retrain_experiment_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("experiments.id")
    )
    report_json: Mapped[dict | None] = mapped_column(JSON)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
