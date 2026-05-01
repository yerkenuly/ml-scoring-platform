from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModelWrapper(ABC):
    model_type: str = "base"
    supports_early_stopping: bool = False
    native_categorical_support: bool = False

    def __init__(self, params: dict):
        self.params = params
        self._model = None

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModelWrapper":
        pass

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        pass

    @abstractmethod
    def get_feature_importance(self) -> pd.Series:
        pass

    def get_params(self) -> dict:
        return self.params

    def get_model(self):
        return self._model
