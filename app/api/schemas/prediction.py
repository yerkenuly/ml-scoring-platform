from pydantic import BaseModel


class PredictRequest(BaseModel):
    features: dict[str, float | int | str | None]


class PredictResponse(BaseModel):
    score: float
    prediction: int
    model_version: str
    latency_ms: float


class BatchPredictRequest(BaseModel):
    records: list[dict[str, float | int | str | None]]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    model_version: str
    total_latency_ms: float
