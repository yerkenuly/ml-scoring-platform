from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    roc_auc_score,
)

from app.core.models.base import BaseModelWrapper
from app.core.pipeline.trainer import FittedModel


@dataclass
class EvaluationResult:
    roc_auc: float
    gini: float
    f1: float
    pr_auc: float
    ks_stat: float
    brier_score: float
    window_id: int | None = None
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ModelEvaluator:
    def evaluate(
        self,
        fitted_model: FittedModel,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        window_id: int | None = None,
    ) -> EvaluationResult:
        proba = fitted_model.model.predict_proba(X_val)[:, 1]

        roc_auc = float(roc_auc_score(y_val, proba))
        gini = 2 * roc_auc - 1
        f1 = float(f1_score(y_val, (proba >= 0.5).astype(int), zero_division=0))
        pr_auc = float(average_precision_score(y_val, proba))
        ks = self._ks_stat(y_val.values, proba)
        brier = float(brier_score_loss(y_val, proba))

        return EvaluationResult(
            roc_auc=round(roc_auc, 6),
            gini=round(gini, 6),
            f1=round(f1, 6),
            pr_auc=round(pr_auc, 6),
            ks_stat=round(ks, 6),
            brier_score=round(brier, 6),
            window_id=window_id,
        )

    def evaluate_on_time_windows(
        self,
        fitted_model: FittedModel,
        df_full: pd.DataFrame,
        target_column: str,
        feature_names: list[str],
        n_windows: int = 3,
    ) -> list[EvaluationResult]:
        results = []
        window_size = len(df_full) // n_windows

        for i in range(n_windows):
            start = i * window_size
            end = (i + 1) * window_size if i < n_windows - 1 else len(df_full)
            window_df = df_full.iloc[start:end]

            available_features = [f for f in feature_names if f in window_df.columns]
            X_w = window_df[available_features]
            y_w = window_df[target_column]

            if len(y_w.unique()) < 2 or len(y_w) < 10:
                continue

            result = self.evaluate(fitted_model, X_w, y_w, window_id=i)
            results.append(result)

        return results

    def _ks_stat(self, y_true: np.ndarray, scores: np.ndarray) -> float:
        pos_scores = scores[y_true == 1]
        neg_scores = scores[y_true == 0]
        if len(pos_scores) == 0 or len(neg_scores) == 0:
            return 0.0
        pos_scores.sort()
        neg_scores.sort()
        all_scores = np.sort(np.concatenate([pos_scores, neg_scores]))
        pos_cdf = np.searchsorted(pos_scores, all_scores, side="right") / len(pos_scores)
        neg_cdf = np.searchsorted(neg_scores, all_scores, side="right") / len(neg_scores)
        return float(np.max(np.abs(pos_cdf - neg_cdf)))
