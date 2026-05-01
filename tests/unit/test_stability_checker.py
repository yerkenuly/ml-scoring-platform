import pandas as pd
import pytest
from sklearn.datasets import make_classification

from app.core.pipeline.feature_engineering import FeatureEngineeringStep
from app.core.pipeline.stability_checker import StabilityChecker
from app.core.pipeline.trainer import ModelTrainer


def test_stable_model_passes(sample_df):
    X = sample_df.drop(columns=["target", "date"])
    y = sample_df["target"]
    fe = FeatureEngineeringStep()
    X_fe, fitted_pipeline = fe.fit_transform(X, y)
    trainer = ModelTrainer()
    fitted = trainer.train("logistic_regression", {"C": 1.0}, X_fe, y)
    fitted.feature_names = list(X_fe.columns)

    df_fe = fe.transform(X, fitted_pipeline)
    df_fe["target"] = y.values

    checker = StabilityChecker()
    result = checker.check(fitted, df_fe, "target", n_windows=3)
    assert isinstance(result.is_stable, bool)
    assert 0.0 <= result.stability_score <= 1.0
    assert result.roc_auc_std >= 0.0


def test_stability_result_has_window_scores(sample_df):
    X = sample_df.drop(columns=["target", "date"])
    y = sample_df["target"]
    fe = FeatureEngineeringStep()
    X_fe, fitted_pipeline = fe.fit_transform(X, y)
    trainer = ModelTrainer()
    fitted = trainer.train("logistic_regression", {"C": 1.0}, X_fe, y)
    fitted.feature_names = list(X_fe.columns)
    df_fe = fe.transform(X, fitted_pipeline)
    df_fe["target"] = y.values

    checker = StabilityChecker()
    result = checker.check(fitted, df_fe, "target", n_windows=3)
    assert len(result.window_scores) > 0
