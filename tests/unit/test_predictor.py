import numpy as np
import pytest


class _MockModel:
    def predict_proba(self, X):
        return np.array([[0.3, 0.7]] * len(X))


def test_predictor_predict_without_model():
    from app.core.serving.predictor import Predictor

    p = Predictor()
    assert not p.is_ready()


def test_predictor_predict_returns_score():
    from app.core.serving.predictor import Predictor

    p = Predictor()
    p._model = _MockModel()
    p._model_version = "test-v1"

    result = p.predict({"feature_0": 1.0, "feature_1": 2.0})
    assert 0.0 <= result["score"] <= 1.0
    assert result["prediction"] in (0, 1)


def test_predictor_batch():
    from app.core.serving.predictor import Predictor

    p = Predictor()
    p._model = _MockModel()
    p._model_version = "test-v1"

    records = [{"feature_0": i, "feature_1": i + 1} for i in range(5)]
    results = p.predict_batch(records)
    assert len(results) == 5
    assert all(0.0 <= r["score"] <= 1.0 for r in results)
