"""Baseline linear regression model."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


class LinearRegressionModel:
    """Wrapper around scikit-learn LinearRegression."""

    def __init__(self) -> None:
        """Initialize the linear regression model."""
        self.model = LinearRegression()

    def train(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Train baseline model on unscaled data.

        Args:
            x_train: Training features.
            y_train: Training target values.

        Returns:
            None.
        """
        self.model.fit(x_train, y_train)

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """Generate predictions.

        Args:
            x_test: Test features.

        Returns:
            Predicted target values.
        """
        return self.model.predict(x_test)

    def save(self, path: str = "./models/linear_regression.pkl") -> None:
        """Persist trained model to disk.

        Args:
            path: Output model path.

        Returns:
            None.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out_path)
        print(f"[SAVED] {out_path.as_posix()}")

