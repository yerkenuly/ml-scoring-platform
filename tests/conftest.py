import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def sample_df() -> pd.DataFrame:
    X, y = make_classification(
        n_samples=1000,
        n_features=10,
        n_informative=6,
        n_redundant=2,
        weights=[0.7, 0.3],
        random_state=42,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(10)])
    df["target"] = y
    df["date"] = pd.date_range("2023-01-01", periods=1000, freq="D")
    return df


@pytest.fixture
def X_y(sample_df):
    X = sample_df.drop(columns=["target", "date"])
    y = sample_df["target"]
    return X, y
