import numpy as np
import pandas as pd
import pytest

from app.core.pipeline.feature_engineering import FeatureEngineeringStep


def test_fit_transform_returns_dataframe(X_y):
    X, y = X_y
    fe = FeatureEngineeringStep()
    X_out, fitted = fe.fit_transform(X, y)
    assert isinstance(X_out, pd.DataFrame)
    assert len(X_out) == len(X)


def test_transform_consistent_columns(X_y):
    X, y = X_y
    fe = FeatureEngineeringStep()
    X_train = X.iloc[:700]
    X_val = X.iloc[700:]
    y_train = y.iloc[:700]
    _, fitted = fe.fit_transform(X_train, y_train)
    X_val_out = fe.transform(X_val, fitted)
    assert list(X_val_out.columns) == fitted.feature_names


def test_clips_outliers(X_y):
    X, y = X_y
    X_dirty = X.copy()
    X_dirty.iloc[0, 0] = 1e9
    fe = FeatureEngineeringStep()
    X_out, _ = fe.fit_transform(X_dirty, y)
    assert X_out.iloc[0, 0] < 1e8
