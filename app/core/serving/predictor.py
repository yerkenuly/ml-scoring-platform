import logging
import threading

import joblib
import mlflow.sklearn
import numpy as np

from app.config import settings
from app.core.serving.feature_transformer import FeatureTransformer

logger = logging.getLogger(__name__)


class Predictor:
    def __init__(self):
        self._model = None
        self._transformer: FeatureTransformer | None = None
        self._model_version: str = ""
        self._lock = threading.RLock()

    def is_ready(self) -> bool:
        return self._model is not None

    def load_production_model(self) -> None:
        try:
            model_uri = f"models:/{settings.mlflow_model_registry_name}/Production"
            model = mlflow.sklearn.load_model(model_uri)
            logger.info("Loaded production model from %s", model_uri)
            with self._lock:
                self._model = model
                self._model_version = model_uri
        except Exception as exc:
            logger.warning("Could not load production model: %s", exc)

    def load_from_uri(self, model_uri: str, feature_pipeline_path: str | None = None) -> None:
        model = mlflow.sklearn.load_model(model_uri)
        transformer = None
        if feature_pipeline_path:
            fitted_pipeline = joblib.load(feature_pipeline_path)
            transformer = FeatureTransformer(fitted_pipeline)

        with self._lock:
            self._model = model
            self._transformer = transformer
            self._model_version = model_uri

    def predict(self, features: dict) -> dict:
        with self._lock:
            model = self._model
            transformer = self._transformer
            version = self._model_version

        if transformer:
            X = transformer.transform(features)
        else:
            import pandas as pd
            X = pd.DataFrame([features])

        proba = model.predict_proba(X)[:, 1]
        score = float(proba[0])
        prediction = int(score >= 0.5)

        return {"score": score, "prediction": prediction, "model_version": version}

    def predict_batch(self, records: list[dict]) -> list[dict]:
        with self._lock:
            model = self._model
            transformer = self._transformer
            version = self._model_version

        if transformer:
            X = transformer.transform_batch(records)
        else:
            import pandas as pd
            X = pd.DataFrame(records)

        probas = model.predict_proba(X)[:, 1]

        return [
            {
                "score": float(p),
                "prediction": int(p >= 0.5),
                "model_version": version,
            }
            for p in probas
        ]
