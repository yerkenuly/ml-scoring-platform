from dataclasses import dataclass

import pandas as pd

from app.core.models.base import BaseModelWrapper
from app.core.models.model_factory import ModelFactory


@dataclass
class FittedModel:
    model: BaseModelWrapper
    model_type: str
    feature_names: list[str]
    params: dict


class ModelTrainer:
    def train(
        self,
        model_type: str,
        params: dict,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> FittedModel:
        model = ModelFactory.build(model_type, params)
        model.fit(X_train, y_train)

        return FittedModel(
            model=model,
            model_type=model_type,
            feature_names=list(X_train.columns),
            params=params,
        )
