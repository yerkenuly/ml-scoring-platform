import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

from app.core.models.base import BaseModelWrapper


class GradientBoostingWrapper(BaseModelWrapper):
    model_type = "gradient_boosting"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "GradientBoostingWrapper":
        self._model = GradientBoostingClassifier(
            n_estimators=self.params.get("n_estimators", 200),
            max_depth=self.params.get("max_depth", 4),
            learning_rate=self.params.get("learning_rate", 0.05),
            subsample=self.params.get("subsample", 0.8),
            min_samples_leaf=self.params.get("min_samples_leaf", 20),
            random_state=42,
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        return pd.Series(
            self._model.feature_importances_,
            index=self._model.feature_names_in_ if hasattr(self._model, "feature_names_in_") else range(len(self._model.feature_importances_)),
        )
