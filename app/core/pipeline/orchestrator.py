import logging
from dataclasses import dataclass

import pandas as pd

from app.core.pipeline.data_ingestion import DataIngestionStep
from app.core.pipeline.evaluator import ModelEvaluator
from app.core.pipeline.feature_engineering import FeatureEngineeringStep
from app.core.pipeline.model_selector import ModelCandidate, ModelSelector
from app.core.pipeline.registry_manager import RegisteredModelVersion, RegistryManager
from app.core.pipeline.stability_checker import StabilityChecker
from app.core.pipeline.trainer import ModelTrainer
from app.core.tuning.optuna_tuner import OptunaTuner

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResult:
    best_candidate: ModelCandidate
    registered_version: RegisteredModelVersion
    all_candidates: list[ModelCandidate]
    experiment_id: str


class PipelineOrchestrator:
    def __init__(self):
        self.ingestion = DataIngestionStep()
        self.feature_eng = FeatureEngineeringStep()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.stability = StabilityChecker()
        self.selector = ModelSelector()
        self.registry = RegistryManager()

    def run(
        self,
        experiment_id: str,
        file_path: str,
        target_column: str,
        model_types: list[str],
        n_trials: int = 50,
        cv_strategy: str = "auto",
        date_column: str | None = None,
    ) -> ExperimentResult:
        logger.info("Starting pipeline for experiment %s", experiment_id)

        data = self.ingestion.load_and_split(
            file_path, target_column, date_column, cv_strategy
        )
        logger.info("Data loaded: %d train, %d val, %d holdout rows", len(data.X_train), len(data.X_val), len(data.X_holdout))

        X_train_fe, feature_pipeline = self.feature_eng.fit_transform(data.X_train, data.y_train)
        X_val_fe = self.feature_eng.transform(data.X_val, feature_pipeline)
        X_holdout_fe = self.feature_eng.transform(data.X_holdout, feature_pipeline)

        X_trainval = pd.concat([X_train_fe, X_val_fe])
        y_trainval = pd.concat([data.y_train, data.y_val])

        df_full_fe = self.feature_eng.transform(
            data.df_full[[c for c in data.df_full.columns if c != target_column and c != date_column]],
            feature_pipeline,
        )
        df_full_with_target = df_full_fe.copy()
        df_full_with_target[target_column] = data.df_full[target_column].values

        candidates: list[ModelCandidate] = []
        tuner = OptunaTuner()

        for model_type in model_types:
            logger.info("Tuning %s...", model_type)
            try:
                best_params = tuner.optimize(
                    model_type, X_train_fe, data.y_train, X_val_fe, data.y_val, n_trials=n_trials
                )
                fitted_model = self.trainer.train(model_type, best_params, X_trainval, y_trainval)
                eval_result = self.evaluator.evaluate(fitted_model, X_val_fe, data.y_val)
                stability_result = self.stability.check(fitted_model, df_full_with_target, target_column)

                candidates.append(ModelCandidate(
                    fitted_model=fitted_model,
                    feature_pipeline=feature_pipeline,
                    eval_result=eval_result,
                    stability_result=stability_result,
                ))
                logger.info(
                    "%s — ROC-AUC=%.4f, stable=%s",
                    model_type, eval_result.roc_auc, stability_result.is_stable,
                )
            except Exception as exc:
                logger.warning("Failed to train %s: %s", model_type, exc)

        best = self.selector.select_best(candidates)

        holdout_eval = self.evaluator.evaluate(best.fitted_model, X_holdout_fe, data.y_holdout)
        best.holdout_eval = holdout_eval
        logger.info("Best model: %s, holdout ROC-AUC=%.4f", best.fitted_model.model_type, holdout_eval.roc_auc)

        registered = self.registry.register(best, experiment_id, data.content_hash)

        return ExperimentResult(
            best_candidate=best,
            registered_version=registered,
            all_candidates=candidates,
            experiment_id=experiment_id,
        )
