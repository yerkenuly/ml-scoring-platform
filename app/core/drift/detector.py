import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from app.core.drift.thresholds import DEFAULT_THRESHOLDS, DriftThresholdConfig

logger = logging.getLogger(__name__)


@dataclass
class DataDriftReport:
    overall_drift_flag: bool
    drift_score: float
    feature_psi: dict[str, float] = field(default_factory=dict)
    feature_ks_pvalue: dict[str, float] = field(default_factory=dict)
    drifted_features: list[str] = field(default_factory=list)


@dataclass
class PerformanceDriftReport:
    is_degraded: bool
    roc_auc_reference: float
    roc_auc_current: float
    roc_auc_drop: float


@dataclass
class ConceptDriftReport:
    label_psi: float
    score_psi: float
    is_drifted: bool


class DriftDetector:
    def __init__(self, thresholds: DriftThresholdConfig = DEFAULT_THRESHOLDS):
        self.thresholds = thresholds

    def check_data_drift(
        self, reference_df: pd.DataFrame, current_df: pd.DataFrame
    ) -> DataDriftReport:
        num_cols = reference_df.select_dtypes(include=[np.number]).columns.tolist()
        feature_psi = {}
        feature_ks_pvalue = {}
        drifted_features = []

        for col in num_cols:
            if col not in current_df.columns:
                continue
            ref = reference_df[col].dropna().values
            cur = current_df[col].dropna().values
            if len(ref) < 10 or len(cur) < 10:
                continue

            psi = self._compute_psi(ref, cur)
            feature_psi[col] = round(psi, 4)

            ks_stat, p_value = stats.ks_2samp(ref, cur)
            feature_ks_pvalue[col] = round(p_value, 4)

            if psi > self.thresholds.psi_threshold or p_value < self.thresholds.ks_p_value_threshold:
                drifted_features.append(col)

        overall_drift = len(drifted_features) > 0
        drift_score = np.mean(list(feature_psi.values())) if feature_psi else 0.0

        return DataDriftReport(
            overall_drift_flag=overall_drift,
            drift_score=round(float(drift_score), 4),
            feature_psi=feature_psi,
            feature_ks_pvalue=feature_ks_pvalue,
            drifted_features=drifted_features,
        )

    def check_model_performance_drift(
        self,
        model,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        target_col: str,
        feature_names: list[str],
    ) -> PerformanceDriftReport:
        from sklearn.metrics import roc_auc_score

        def _score(df: pd.DataFrame) -> float:
            available = [f for f in feature_names if f in df.columns]
            X = df[available]
            y = df[target_col]
            if len(y.unique()) < 2 or len(y) < 10:
                return float("nan")
            proba = model.predict_proba(X)[:, 1]
            return float(roc_auc_score(y, proba))

        ref_auc = _score(reference_df)
        cur_auc = _score(current_df)

        if np.isnan(ref_auc) or np.isnan(cur_auc):
            return PerformanceDriftReport(
                is_degraded=False, roc_auc_reference=ref_auc, roc_auc_current=cur_auc, roc_auc_drop=0.0
            )

        drop = ref_auc - cur_auc
        is_degraded = drop > self.thresholds.performance_degradation_threshold

        return PerformanceDriftReport(
            is_degraded=is_degraded,
            roc_auc_reference=round(ref_auc, 6),
            roc_auc_current=round(cur_auc, 6),
            roc_auc_drop=round(drop, 6),
        )

    def check_concept_drift(
        self,
        reference_scores: np.ndarray,
        current_scores: np.ndarray,
        reference_labels: np.ndarray | None = None,
        current_labels: np.ndarray | None = None,
    ) -> ConceptDriftReport:
        score_psi = self._compute_psi(reference_scores, current_scores)
        label_psi = 0.0

        if reference_labels is not None and current_labels is not None:
            ref_rate = np.mean(reference_labels)
            cur_rate = np.mean(current_labels)
            if ref_rate > 0 and cur_rate > 0 and ref_rate < 1 and cur_rate < 1:
                label_psi = abs(
                    (ref_rate * np.log(ref_rate / cur_rate))
                    + ((1 - ref_rate) * np.log((1 - ref_rate) / (1 - cur_rate)))
                )

        is_drifted = (
            score_psi > self.thresholds.score_psi_threshold
            or label_psi > self.thresholds.label_psi_threshold
        )

        return ConceptDriftReport(
            label_psi=round(float(label_psi), 4),
            score_psi=round(float(score_psi), 4),
            is_drifted=is_drifted,
        )

    def should_retrain(
        self, data_report: DataDriftReport, perf_report: PerformanceDriftReport
    ) -> bool:
        return data_report.overall_drift_flag or perf_report.is_degraded

    def _compute_psi(self, reference: np.ndarray, current: np.ndarray, buckets: int = 10) -> float:
        breakpoints = np.percentile(reference, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)
        if len(breakpoints) < 2:
            return 0.0

        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]

        ref_pct = (ref_counts + 1e-6) / len(reference)
        cur_pct = (cur_counts + 1e-6) / len(current)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)
