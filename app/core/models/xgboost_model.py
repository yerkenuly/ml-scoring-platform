import numpy as np
import pandas as pd

from app.core.models.base import BaseModelWrapper


class XGBoostWrapper(BaseModelWrapper):
    model_type = "xgboost"
    supports_early_stopping = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGBoostWrapper":
        from xgboost import XGBClassifier

        neg = (y == 0).sum()
        pos = (y == 1).sum()
        scale_pos_weight = neg / pos if pos > 0 else 1.0

        self._model = XGBClassifier(
            n_estimators=self.params.get("n_estimators", 300),
            max_depth=self.params.get("max_depth", 6),
            learning_rate=self.params.get("learning_rate", 0.05),
            subsample=self.params.get("subsample", 0.8),
            colsample_bytree=self.params.get("colsample_bytree", 0.8),
            min_child_weight=self.params.get("min_child_weight", 5),
            reg_alpha=self.params.get("reg_alpha", 0.1),
            reg_lambda=self.params.get("reg_lambda", 1.0),
            scale_pos_weight=scale_pos_weight,
            eval_metric="auc",
            use_label_encoder=False,
            n_jobs=-1,
            random_state=42,
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        scores = self._model.feature_importances_
        names = self._model.feature_names_in_ if hasattr(self._model, "feature_names_in_") else range(len(scores))
        return pd.Series(scores, index=names)
