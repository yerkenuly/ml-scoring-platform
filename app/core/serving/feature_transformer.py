import pandas as pd

from app.core.pipeline.feature_engineering import FittedFeaturePipeline


class FeatureTransformer:
    def __init__(self, fitted_pipeline: FittedFeaturePipeline):
        self.fitted_pipeline = fitted_pipeline

    def transform(self, features: dict) -> pd.DataFrame:
        df = pd.DataFrame([features])
        from app.core.pipeline.feature_engineering import FeatureEngineeringStep
        fe = FeatureEngineeringStep()
        return fe.transform(df, self.fitted_pipeline)

    def transform_batch(self, records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        from app.core.pipeline.feature_engineering import FeatureEngineeringStep
        fe = FeatureEngineeringStep()
        return fe.transform(df, self.fitted_pipeline)
