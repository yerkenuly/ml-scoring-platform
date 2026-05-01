import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from app.core.models.base import BaseModelWrapper


class LogisticRegressionWrapper(BaseModelWrapper):
    model_type = "logistic_regression"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LogisticRegressionWrapper":
        self._model = LogisticRegression(
            class_weight="balanced",
            max_iter=self.params.get("max_iter", 1000),
            C=self.params.get("C", 1.0),
            solver=self.params.get("solver", "lbfgs"),
            random_state=42,
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        coef = np.abs(self._model.coef_[0])
        return pd.Series(coef, index=self._model.feature_names_in_ if hasattr(self._model, "feature_names_in_") else range(len(coef)))
