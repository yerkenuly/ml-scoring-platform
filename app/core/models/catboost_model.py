import numpy as np
import pandas as pd

from app.core.models.base import BaseModelWrapper


class CatBoostWrapper(BaseModelWrapper):
    model_type = "catboost"
    supports_early_stopping = True
    native_categorical_support = True

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "CatBoostWrapper":
        from catboost import CatBoostClassifier

        self._model = CatBoostClassifier(
            iterations=self.params.get("iterations", 300),
            depth=self.params.get("depth", 6),
            learning_rate=self.params.get("learning_rate", 0.05),
            l2_leaf_reg=self.params.get("l2_leaf_reg", 3.0),
            bagging_temperature=self.params.get("bagging_temperature", 1.0),
            auto_class_weights="Balanced",
            eval_metric="AUC",
            random_seed=42,
            verbose=0,
        )
        self._model.fit(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self._model.predict_proba(X)

    def get_feature_importance(self) -> pd.Series:
        scores = self._model.get_feature_importance()
        names = self._model.feature_names_ if hasattr(self._model, "feature_names_") else range(len(scores))
        return pd.Series(scores, index=names)
