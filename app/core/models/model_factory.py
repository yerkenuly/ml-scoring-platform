from app.core.models.base import BaseModelWrapper
from app.core.models.catboost_model import CatBoostWrapper
from app.core.models.gradient_boosting import GradientBoostingWrapper
from app.core.models.lightgbm_model import LightGBMWrapper
from app.core.models.logistic_regression import LogisticRegressionWrapper
from app.core.models.random_forest import RandomForestWrapper
from app.core.models.xgboost_model import XGBoostWrapper

_REGISTRY: dict[str, type[BaseModelWrapper]] = {
    "logistic_regression": LogisticRegressionWrapper,
    "random_forest": RandomForestWrapper,
    "gradient_boosting": GradientBoostingWrapper,
    "xgboost": XGBoostWrapper,
    "lightgbm": LightGBMWrapper,
    "catboost": CatBoostWrapper,
}


class ModelFactory:
    @staticmethod
    def build(model_type: str, params: dict) -> BaseModelWrapper:
        cls = _REGISTRY.get(model_type)
        if cls is None:
            raise ValueError(f"Unknown model type: {model_type}. Available: {list(_REGISTRY)}")
        return cls(params)

    @staticmethod
    def available_types() -> list[str]:
        return list(_REGISTRY.keys())
