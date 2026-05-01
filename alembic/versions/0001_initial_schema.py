"""initial schema

Revision ID: 0001
Revises:
Create Date: 2026-05-01

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY, JSON, UUID

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "datasets",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("description", sa.Text),
        sa.Column("file_path", sa.Text, nullable=False),
        sa.Column("file_format", sa.String(20)),
        sa.Column("row_count", sa.Integer),
        sa.Column("column_count", sa.Integer),
        sa.Column("target_column", sa.String(100), nullable=False),
        sa.Column("date_column", sa.String(100)),
        sa.Column("class_balance", JSON),
        sa.Column("schema_json", JSON),
        sa.Column("content_hash", sa.String(64)),
        sa.Column("validation_report", JSON),
        sa.Column("is_deleted", sa.Boolean, default=False),
        sa.Column("created_at", sa.DateTime(timezone=True)),
        sa.Column("updated_at", sa.DateTime(timezone=True)),
    )

    op.create_table(
        "experiments",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("dataset_id", UUID(as_uuid=True), sa.ForeignKey("datasets.id"), nullable=False),
        sa.Column("mlflow_run_id", sa.String(255)),
        sa.Column("model_types", ARRAY(sa.String)),
        sa.Column("cv_strategy", sa.String(50)),
        sa.Column("n_trials", sa.Integer, default=50),
        sa.Column("target_metric", sa.String(50), default="roc_auc"),
        sa.Column("status", sa.String(30), default="queued"),
        sa.Column("config_json", JSON),
        sa.Column("results_json", JSON),
        sa.Column("error_message", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )

    op.create_table(
        "model_versions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("experiment_id", UUID(as_uuid=True), sa.ForeignKey("experiments.id")),
        sa.Column("model_type", sa.String(50), nullable=False),
        sa.Column("mlflow_version", sa.Integer),
        sa.Column("mlflow_model_uri", sa.String(500)),
        sa.Column("stage", sa.String(30), default="staging"),
        sa.Column("roc_auc", sa.Float),
        sa.Column("gini", sa.Float),
        sa.Column("f1_score", sa.Float),
        sa.Column("pr_auc", sa.Float),
        sa.Column("ks_stat", sa.Float),
        sa.Column("brier_score", sa.Float),
        sa.Column("stability_score", sa.Float),
        sa.Column("roc_auc_std", sa.Float),
        sa.Column("trend_slope", sa.Float),
        sa.Column("is_stable", sa.Boolean, default=False),
        sa.Column("hyperparams_json", JSON),
        sa.Column("feature_names", ARRAY(sa.String)),
        sa.Column("feature_importance", JSON),
        sa.Column("dataset_id", UUID(as_uuid=True), sa.ForeignKey("datasets.id")),
        sa.Column("dataset_hash", sa.String(64)),
        sa.Column("promoted_at", sa.DateTime(timezone=True)),
        sa.Column("archived_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_model_versions_stage", "model_versions", ["stage"])
    op.create_index("idx_model_versions_experiment", "model_versions", ["experiment_id"])

    op.create_table(
        "prediction_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("model_version_id", UUID(as_uuid=True), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("request_id", sa.String(100)),
        sa.Column("features_json", JSON, nullable=False),
        sa.Column("score", sa.Float, nullable=False),
        sa.Column("prediction", sa.SmallInteger, nullable=False),
        sa.Column("true_label", sa.SmallInteger),
        sa.Column("latency_ms", sa.Float),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_prediction_logs_model_version", "prediction_logs", ["model_version_id"])
    op.create_index("idx_prediction_logs_created_at", "prediction_logs", ["created_at"])

    op.create_table(
        "drift_reports",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("model_version_id", UUID(as_uuid=True), sa.ForeignKey("model_versions.id"), nullable=False),
        sa.Column("check_type", sa.String(30)),
        sa.Column("overall_drift_flag", sa.Boolean, nullable=False),
        sa.Column("drift_score", sa.Float),
        sa.Column("feature_drift_json", JSON),
        sa.Column("performance_delta", sa.Float),
        sa.Column("triggered_retraining", sa.Boolean, default=False),
        sa.Column("retrain_experiment_id", UUID(as_uuid=True), sa.ForeignKey("experiments.id")),
        sa.Column("report_json", JSON),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )
    op.create_index("idx_drift_reports_model_version", "drift_reports", ["model_version_id"])

    op.create_table(
        "jobs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("celery_task_id", sa.String(255), unique=True),
        sa.Column("job_type", sa.String(50)),
        sa.Column("entity_id", UUID(as_uuid=True)),
        sa.Column("status", sa.String(30), default="queued"),
        sa.Column("progress", sa.Float, default=0.0),
        sa.Column("log_messages", ARRAY(sa.Text)),
        sa.Column("error_message", sa.Text),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("completed_at", sa.DateTime(timezone=True)),
        sa.Column("created_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    op.drop_table("jobs")
    op.drop_table("drift_reports")
    op.drop_table("prediction_logs")
    op.drop_table("model_versions")
    op.drop_table("experiments")
    op.drop_table("datasets")
