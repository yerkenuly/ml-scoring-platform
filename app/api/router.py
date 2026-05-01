from fastapi import APIRouter

from app.api.v1 import datasets, experiments, models, predictions, monitoring, admin

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(datasets.router)
api_router.include_router(experiments.router)
api_router.include_router(models.router)
api_router.include_router(predictions.router)
api_router.include_router(monitoring.router)
api_router.include_router(admin.router)
