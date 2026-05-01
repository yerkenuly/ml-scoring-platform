import pandas as pd
import pytest

from app.core.pipeline.evaluator import ModelEvaluator
from app.core.pipeline.feature_engineering import FeatureEngineeringStep
from app.core.pipeline.trainer import FittedModel, ModelTrainer


def _build_fitted_model(X_y):
    X, y = X_y
    fe = FeatureEngineeringStep()
    X_fe, fitted_pipeline = fe.fit_transform(X, y)
    trainer = ModelTrainer()
    fitted = trainer.train("logistic_regression", {"C": 1.0}, X_fe, y)
    return fitted, X_fe, y


def test_evaluate_returns_valid_metrics(X_y):
    fitted, X_fe, y = _build_fitted_model(X_y)
    evaluator = ModelEvaluator()
    result = evaluator.evaluate(fitted, X_fe, y)
    assert 0.0 <= result.roc_auc <= 1.0
    assert -1.0 <= result.gini <= 1.0
    assert 0.0 <= result.f1 <= 1.0
    assert 0.0 <= result.brier_score <= 1.0


def test_gini_equals_2_roc_minus_1(X_y):
    fitted, X_fe, y = _build_fitted_model(X_y)
    evaluator = ModelEvaluator()
    result = evaluator.evaluate(fitted, X_fe, y)
    assert abs(result.gini - (2 * result.roc_auc - 1)) < 1e-6
