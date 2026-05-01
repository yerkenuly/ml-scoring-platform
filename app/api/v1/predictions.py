import time

from fastapi import APIRouter, Depends

from app.api.schemas.prediction import (
    BatchPredictRequest,
    BatchPredictResponse,
    PredictRequest,
    PredictResponse,
)
from app.dependencies import verify_api_key
from app.exceptions import ModelNotReadyError

router = APIRouter(prefix="/predict", tags=["predictions"])

_predictor = None


def get_predictor():
    from app.core.serving.predictor import Predictor

    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


@router.post("", response_model=PredictResponse)
async def predict(
    body: PredictRequest,
    _: str = Depends(verify_api_key),
):
    predictor = get_predictor()
    if not predictor.is_ready():
        raise ModelNotReadyError()

    start = time.perf_counter()
    result = predictor.predict(body.features)
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        score=result["score"],
        prediction=result["prediction"],
        model_version=result["model_version"],
        latency_ms=round(latency_ms, 2),
    )


@router.post("/batch", response_model=BatchPredictResponse)
async def predict_batch(
    body: BatchPredictRequest,
    _: str = Depends(verify_api_key),
):
    predictor = get_predictor()
    if not predictor.is_ready():
        raise ModelNotReadyError()

    start = time.perf_counter()
    results = predictor.predict_batch(body.records)
    total_latency_ms = (time.perf_counter() - start) * 1000

    predictions = [
        PredictResponse(
            score=r["score"],
            prediction=r["prediction"],
            model_version=r["model_version"],
            latency_ms=0.0,
        )
        for r in results
    ]

    return BatchPredictResponse(
        predictions=predictions,
        model_version=results[0]["model_version"] if results else "",
        total_latency_ms=round(total_latency_ms, 2),
    )
