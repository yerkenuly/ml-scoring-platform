from dataclasses import dataclass

from app.core.pipeline.evaluator import EvaluationResult
from app.core.pipeline.stability_checker import StabilityResult
from app.core.pipeline.trainer import FittedModel
from app.core.pipeline.feature_engineering import FittedFeaturePipeline
from app.exceptions import NoStableModelError


@dataclass
class ModelCandidate:
    fitted_model: FittedModel
    feature_pipeline: FittedFeaturePipeline
    eval_result: EvaluationResult
    stability_result: StabilityResult
    holdout_eval: EvaluationResult | None = None

    def composite_score(self) -> float:
        e = self.eval_result
        s = self.stability_result
        return (
            0.40 * e.roc_auc
            + 0.20 * s.stability_score
            + 0.20 * e.f1
            + 0.10 * e.pr_auc
            + 0.10 * (1.0 - e.brier_score)
        )


class ModelSelector:
    def select_best(self, candidates: list[ModelCandidate]) -> ModelCandidate:
        stable = [c for c in candidates if c.stability_result.is_stable]
        if not stable:
            reasons = [
                f"{c.fitted_model.model_type}: {c.stability_result.failure_reason}"
                for c in candidates
            ]
            raise NoStableModelError(
                f"No stable model found. Reasons: {'; '.join(reasons)}"
            )

        return max(stable, key=lambda c: c.composite_score())

    def rank_candidates(self, candidates: list[ModelCandidate]) -> list[tuple[ModelCandidate, float]]:
        return sorted(
            [(c, c.composite_score()) for c in candidates],
            key=lambda x: x[1],
            reverse=True,
        )
