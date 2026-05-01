import numpy as np
import pandas as pd

from app.core.models.base import BaseModelWrapper


class LightGBMWrapper(BaseModelWrapper):
    model_type = "lightgbm"
    supports_early_stopping = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LightGBMWrapper":
        from lightgbm import LGBMClassifier

        self._model = LGBMClassifier(
            n_estimators=self.params.get("n_estimators", 300),
            max_depth=self.params.get("max_depth", -1),
            num_leaves=self.params.get("num_leaves", 63),
            learning_rate=self.params.get("learning_rate", 0.05),
            subsample=self.params.get("subsample", 0.8),
            colsample_bytree=self.params.get("colsample_bytree", 0.8),
            reg_alpha=self.params.get("reg_alpha", 0.1),
            reg_lambda=self.params.get("reg_lambda", 1.0),
            is_unbalance=True,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        scores = self._model.feature_importances_
        names = self._model.feature_name_ if hasattr(self._model, "feature_name_") else range(len(scores))
        return pd.Series(scores, index=names)
