import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from app.core.models.model_factory import ModelFactory
from app.core.tuning.search_spaces import get_space

optuna.logging.set_verbosity(optuna.logging.WARNING)
logger = logging.getLogger(__name__)


class OptunaTuner:
    def optimize(
        self,
        model_type: str,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        n_trials: int = 50,
    ) -> dict:
        def objective(trial: optuna.Trial) -> float:
            params = get_space(model_type, trial)
            try:
                model = ModelFactory.build(model_type, params)
                cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
                scores = cross_val_score(
                    model.get_model() if model.get_model() else self._build_sklearn(model_type, params),
                    X_train, y_train,
                    cv=cv,
                    scoring="roc_auc",
                    n_jobs=-1,
                )
                return float(np.mean(scores))
            except Exception as exc:
                logger.debug("Trial failed for %s: %s", model_type, exc)
                raise optuna.TrialPruned()

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.MedianPruner(n_warmup_steps=10),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

        if study.best_trial is None:
            return {}

        logger.info(
            "Best %s params (ROC-AUC=%.4f): %s",
            model_type, study.best_value, study.best_params,
        )
        return study.best_params

    def _build_sklearn(self, model_type: str, params: dict):
        model_wrapper = ModelFactory.build(model_type, params)
        from sklearn.base import BaseEstimator, ClassifierMixin

        class _Adapter(BaseEstimator, ClassifierMixin):
            def __init__(self, wrapper):
                self.wrapper = wrapper

            def fit(self, X, y):
                self.wrapper.fit(pd.DataFrame(X), pd.Series(y))
                return self

            def predict_proba(self, X):
                return self.wrapper.predict_proba(pd.DataFrame(X))

            def predict(self, X):
                proba = self.predict_proba(X)
                return (proba[:, 1] >= 0.5).astype(int)

        return _Adapter(model_wrapper)
