from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    # Database
    database_url: str = "postgresql+asyncpg://postgres:password@localhost:5432/ml_scoring"
    sync_database_url: str = "postgresql://postgres:password@localhost:5432/ml_scoring"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"
    mlflow_experiment_name: str = "ml-scoring"
    mlflow_model_registry_name: str = "scoring-model"

    # Storage
    artifact_store_path: str = "./mlruns"
    dataset_upload_path: str = "./uploads"

    # Training
    default_cv_folds: int = 5
    stability_window_count: int = 3
    min_stability_score: float = 0.02
    roc_auc_floor: float = 0.70
    target_metric: str = "roc_auc"

    # Drift
    drift_check_interval_seconds: int = 3600
    psi_threshold: float = 0.2
    ks_threshold: float = 0.05
    performance_degradation_threshold: float = 0.03

    # API
    api_key_header: str = "X-API-Key"
    max_batch_predict_size: int = 10_000

    # App
    debug: bool = False
    log_level: str = "INFO"


settings = Settings()
