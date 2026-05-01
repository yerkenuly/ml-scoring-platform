import logging

import pandas as pd

from app.core.drift.detector import DataDriftReport, DriftDetector, PerformanceDriftReport

logger = logging.getLogger(__name__)


class DataDriftMonitor:
    def __init__(self):
        self.detector = DriftDetector()

    def run(self, reference_df: pd.DataFrame, current_df: pd.DataFrame) -> DataDriftReport:
        report = self.detector.check_data_drift(reference_df, current_df)
        if report.overall_drift_flag:
            logger.warning("Data drift detected in features: %s", report.drifted_features)
        return report


class ConceptDriftMonitor:
    def __init__(self):
        self.detector = DriftDetector()

    def run(
        self,
        model,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        target_col: str,
        feature_names: list[str],
    ) -> PerformanceDriftReport:
        report = self.detector.check_model_performance_drift(
            model, reference_df, current_df, target_col, feature_names
        )
        if report.is_degraded:
            logger.warning(
                "Performance drift: ROC-AUC dropped %.4f (%.4f → %.4f)",
                report.roc_auc_drop, report.roc_auc_reference, report.roc_auc_current,
            )
        return report
