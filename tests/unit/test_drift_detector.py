import numpy as np
import pandas as pd
import pytest

from app.core.drift.detector import DriftDetector


def test_no_drift_identical_data():
    np.random.seed(42)
    df = pd.DataFrame({"a": np.random.normal(0, 1, 500), "b": np.random.normal(5, 2, 500)})
    detector = DriftDetector()
    report = detector.check_data_drift(df, df.copy())
    assert report.overall_drift_flag is False


def test_drift_detected_on_shifted_distribution():
    np.random.seed(42)
    reference = pd.DataFrame({"a": np.random.normal(0, 1, 500)})
    current = pd.DataFrame({"a": np.random.normal(5, 1, 500)})
    detector = DriftDetector()
    report = detector.check_data_drift(reference, current)
    assert report.overall_drift_flag is True
    assert "a" in report.drifted_features


def test_psi_zero_for_identical():
    np.random.seed(42)
    arr = np.random.normal(0, 1, 1000)
    detector = DriftDetector()
    psi = detector._compute_psi(arr, arr)
    assert psi < 0.01
