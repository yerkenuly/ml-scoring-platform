"""Generate a sample classification dataset for testing the platform."""
import argparse
import os

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def generate(output_path: str, n_samples: int = 5000, seed: int = 42) -> None:
    np.random.seed(seed)
    X, y = make_classification(
        n_samples=n_samples,
        n_features=15,
        n_informative=8,
        n_redundant=3,
        weights=[0.75, 0.25],
        random_state=seed,
    )
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(15)])
    df["target"] = y
    df["event_date"] = pd.date_range("2022-01-01", periods=n_samples, freq="H")
    df["category"] = np.random.choice(["A", "B", "C", "D"], size=n_samples)

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Generated {n_samples} rows → {output_path}")
    print(f"Class balance: {df['target'].value_counts(normalize=True).to_dict()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="uploads/sample_dataset.csv")
    parser.add_argument("--n-samples", type=int, default=5000)
    args = parser.parse_args()
    generate(args.output, args.n_samples)
