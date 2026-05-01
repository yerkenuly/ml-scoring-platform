import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class DataSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_holdout: pd.DataFrame
    y_holdout: pd.Series
    df_full: pd.DataFrame
    target_column: str
    date_column: str | None
    content_hash: str
    class_balance: dict[str, float]
    cv_strategy: str


class DataIngestionStep:
    MIN_ROWS = 200

    def load_and_split(
        self,
        file_path: str,
        target_column: str,
        date_column: str | None = None,
        cv_strategy: str = "auto",
    ) -> DataSplit:
        df = self._read_file(file_path)
        self._validate(df, target_column)

        content_hash = hashlib.sha256(pd.util.hash_pandas_object(df).values.tobytes()).hexdigest()
        class_balance = self._class_balance(df[target_column])

        if date_column and date_column in df.columns:
            df = df.sort_values(date_column).reset_index(drop=True)
            effective_cv = "timeseries_split" if cv_strategy == "auto" else cv_strategy
        else:
            date_column = None
            effective_cv = "stratified_kfold" if cv_strategy == "auto" else cv_strategy

        n = len(df)
        train_end = int(n * 0.70)
        val_end = int(n * 0.85)

        train_df = df.iloc[:train_end]
        val_df = df.iloc[train_end:val_end]
        holdout_df = df.iloc[val_end:]

        feature_cols = [c for c in df.columns if c != target_column and c != date_column]

        return DataSplit(
            X_train=train_df[feature_cols],
            y_train=train_df[target_column],
            X_val=val_df[feature_cols],
            y_val=val_df[target_column],
            X_holdout=holdout_df[feature_cols],
            y_holdout=holdout_df[target_column],
            df_full=df,
            target_column=target_column,
            date_column=date_column,
            content_hash=content_hash,
            class_balance=class_balance,
            cv_strategy=effective_cv,
        )

    def _read_file(self, file_path: str) -> pd.DataFrame:
        if file_path.endswith(".parquet"):
            return pd.read_parquet(file_path)
        return pd.read_csv(file_path)

    def _validate(self, df: pd.DataFrame, target_column: str) -> None:
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        if len(df) < self.MIN_ROWS:
            raise ValueError(f"Dataset too small: {len(df)} rows, minimum {self.MIN_ROWS}")
        null_target = df[target_column].isna().sum()
        if null_target > 0:
            raise ValueError(f"Target column has {null_target} null values")

    def _class_balance(self, y: pd.Series) -> dict[str, float]:
        counts = y.value_counts(normalize=True)
        return {str(k): round(float(v), 4) for k, v in counts.items()}
