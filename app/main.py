import logging
from contextlib import asynccontextmanager

import mlflow
from fastapi import FastAPI
import app.db.models  # noqa: F401 — registers all ORM models with SQLAlchemy metadata
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.exceptions import AppError, app_error_handler
from app.api.router import api_router

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    logger.info("MLflow tracking URI: %s", settings.mlflow_tracking_uri)
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="ML Scoring Platform",
    description="Automated ML model training and serving with stability guarantees",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_exception_handler(AppError, app_error_handler)
app.include_router(api_router)


@app.get("/health", tags=["system"])
async def health():
    return {"status": "ok"}


@app.get("/ready", tags=["system"])
async def ready():
    return {"status": "ready"}
