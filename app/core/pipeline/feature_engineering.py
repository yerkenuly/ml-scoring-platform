from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, RobustScaler
from sklearn.feature_selection import VarianceThreshold


@dataclass
class FittedFeaturePipeline:
    pipeline: Pipeline
    feature_names: list[str]
    numerical_cols: list[str]
    categorical_cols: list[str]


class FeatureEngineeringStep:
    LOW_VARIANCE_THRESHOLD = 0.01
    HIGH_CARDINALITY_THRESHOLD = 20
    OUTLIER_CLIP_PERCENTILE = (1, 99)

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, FittedFeaturePipeline]:
        X = X.copy()
        num_cols, cat_cols = self._detect_column_types(X)

        num_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])

        cat_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])

        transformers = []
        if num_cols:
            transformers.append(("num", num_pipeline, num_cols))
        if cat_cols:
            transformers.append(("cat", cat_pipeline, cat_cols))

        preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

        full_pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("variance_filter", VarianceThreshold(threshold=self.LOW_VARIANCE_THRESHOLD)),
        ])

        X_clipped = self._clip_outliers(X, num_cols)
        X_transformed = full_pipeline.fit_transform(X_clipped, y)

        feature_names = self._get_feature_names(full_pipeline, num_cols, cat_cols)

        X_out = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

        fitted = FittedFeaturePipeline(
            pipeline=full_pipeline,
            feature_names=feature_names,
            numerical_cols=num_cols,
            categorical_cols=cat_cols,
        )
        return X_out, fitted

    def transform(self, X: pd.DataFrame, fitted: FittedFeaturePipeline) -> pd.DataFrame:
        X = X.copy()
        X_clipped = self._clip_outliers(X, fitted.numerical_cols)
        X_transformed = fitted.pipeline.transform(X_clipped)
        return pd.DataFrame(X_transformed, columns=fitted.feature_names, index=X.index)

    def _detect_column_types(self, X: pd.DataFrame) -> tuple[list[str], list[str]]:
        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        return num_cols, cat_cols

    def _clip_outliers(self, X: pd.DataFrame, num_cols: list[str]) -> pd.DataFrame:
        X = X.copy()
        for col in num_cols:
            if col in X.columns:
                lo = X[col].quantile(self.OUTLIER_CLIP_PERCENTILE[0] / 100)
                hi = X[col].quantile(self.OUTLIER_CLIP_PERCENTILE[1] / 100)
                X[col] = X[col].clip(lo, hi)
        return X

    def _get_feature_names(self, pipeline: Pipeline, num_cols: list[str], cat_cols: list[str]) -> list[str]:
        try:
            preprocessor = pipeline.named_steps["preprocessor"]
            names = []
            for name, _, cols in preprocessor.transformers_:
                if name != "remainder":
                    names.extend(cols)
            variance_filter = pipeline.named_steps["variance_filter"]
            mask = variance_filter.get_support()
            return [n for n, keep in zip(names, mask) if keep]
        except Exception:
            n_features = pipeline.named_steps["variance_filter"].n_features_in_
            return [f"feature_{i}" for i in range(n_features)]
