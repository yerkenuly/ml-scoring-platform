import os
import tempfile

import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def temp_csv():
    X, y = make_classification(n_samples=500, n_features=8, n_informative=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(8)])
    df["target"] = y
    df["date"] = pd.date_range("2022-01-01", periods=500, freq="D")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        df.to_csv(f.name, index=False)
        yield f.name

    os.unlink(f.name)


def test_data_ingestion_splits(temp_csv):
    from app.core.pipeline.data_ingestion import DataIngestionStep

    step = DataIngestionStep()
    split = step.load_and_split(temp_csv, "target", date_column="date")
    assert len(split.X_train) > 0
    assert len(split.X_val) > 0
    assert len(split.X_holdout) > 0
    total = len(split.X_train) + len(split.X_val) + len(split.X_holdout)
    assert total == 500


def test_feature_engineering_no_leakage(temp_csv):
    from app.core.pipeline.data_ingestion import DataIngestionStep
    from app.core.pipeline.feature_engineering import FeatureEngineeringStep

    split = DataIngestionStep().load_and_split(temp_csv, "target", date_column="date")
    fe = FeatureEngineeringStep()
    X_train_fe, fitted = fe.fit_transform(split.X_train, split.y_train)
    X_val_fe = fe.transform(split.X_val, fitted)
    assert list(X_train_fe.columns) == list(X_val_fe.columns)
