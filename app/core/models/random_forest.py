import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from app.core.models.base import BaseModelWrapper


class RandomForestWrapper(BaseModelWrapper):
    model_type = "random_forest"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestWrapper":
        self._model = RandomForestClassifier(
            n_estimators=self.params.get("n_estimators", 200),
            max_depth=self.params.get("max_depth", None),
            min_samples_leaf=self.params.get("min_samples_leaf", 1),
            max_features=self.params.get("max_features", "sqrt"),
            class_weight="balanced",
            n_jobs=-1,
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
