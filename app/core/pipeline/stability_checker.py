from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from app.config import settings
from app.core.pipeline.evaluator import EvaluationResult, ModelEvaluator
from app.core.pipeline.trainer import FittedModel


@dataclass
class StabilityResult:
    is_stable: bool
    stability_score: float
    roc_auc_std: float
    gini_std: float
    f1_std: float
    trend_slope: float
    window_scores: list[EvaluationResult] = field(default_factory=list)
    failure_reason: str | None = None


class StabilityChecker:
    def __init__(self):
        self.evaluator = ModelEvaluator()

    def check(
        self,
        fitted_model: FittedModel,
        df_full: pd.DataFrame,
        target_column: str,
        n_windows: int | None = None,
    ) -> StabilityResult:
        n_windows = n_windows or settings.stability_window_count
        feature_names = fitted_model.feature_names

        window_scores = self.evaluator.evaluate_on_time_windows(
            fitted_model, df_full, target_column, feature_names, n_windows
        )

        if len(window_scores) < 2:
            return StabilityResult(
                is_stable=False,
                stability_score=0.0,
                roc_auc_std=1.0,
                gini_std=1.0,
                f1_std=1.0,
                trend_slope=-1.0,
                window_scores=window_scores,
                failure_reason="Not enough windows to assess stability",
            )

        roc_aucs = np.array([w.roc_auc for w in window_scores])
        ginis = np.array([w.gini for w in window_scores])
        f1s = np.array([w.f1 for w in window_scores])

        roc_auc_std = float(np.std(roc_aucs))
        gini_std = float(np.std(ginis))
        f1_std = float(np.std(f1s))
        mean_roc_auc = float(np.mean(roc_aucs))
        min_roc_auc = float(np.min(roc_aucs))

        indices = np.arange(len(roc_aucs))
        trend_slope = float(np.polyfit(indices, roc_aucs, 1)[0])

        stability_score = max(0.0, 1.0 - (roc_auc_std / mean_roc_auc)) if mean_roc_auc > 0 else 0.0

        failure_reason = None
        if roc_auc_std >= settings.min_stability_score:
            failure_reason = f"ROC-AUC std {roc_auc_std:.4f} >= threshold {settings.min_stability_score}"
        elif trend_slope < -0.005:
            failure_reason = f"Declining ROC-AUC trend: slope={trend_slope:.4f}"
        elif min_roc_auc < settings.roc_auc_floor:
            failure_reason = f"Min ROC-AUC {min_roc_auc:.4f} below floor {settings.roc_auc_floor}"

        is_stable = failure_reason is None

        return StabilityResult(
            is_stable=is_stable,
            stability_score=round(stability_score, 6),
            roc_auc_std=round(roc_auc_std, 6),
            gini_std=round(gini_std, 6),
            f1_std=round(f1_std, 6),
            trend_slope=round(trend_slope, 6),
            window_scores=window_scores,
            failure_reason=failure_reason,
        )
