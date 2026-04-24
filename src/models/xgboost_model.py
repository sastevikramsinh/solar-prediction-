"""XGBoost model for solar power regression."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBRegressor


class XGBoostModel:
    """Wrapper around XGBRegressor."""

    def __init__(self) -> None:
        """Initialize XGBoost model with project parameters."""
        self.model = XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            early_stopping_rounds=50,
            random_state=42,
        )

    def train(
        self,
        x_train: pd.DataFrame,
        y_train: pd.Series,
        x_val: pd.DataFrame,
        y_val: pd.Series,
    ) -> None:
        """Train XGBoost model using validation set for early stopping.

        Args:
            x_train: Training features.
            y_train: Training targets.
            x_val: Validation features.
            y_val: Validation targets.

        Returns:
            None.
        """
        self.model.fit(
            x_train,
            y_train,
            eval_set=[(x_val, y_val)],
            verbose=50,
        )

    def predict(self, x_test: pd.DataFrame) -> np.ndarray:
        """Predict AC power directly in kW.

        Args:
            x_test: Test features.

        Returns:
            Predicted kW values.
        """
        return self.model.predict(x_test)

    def plot_feature_importance(
        self, feature_names: list[str], x_test: pd.DataFrame, y_test: pd.Series
    ) -> None:
        """Create SHAP summary plot from a test subset.

        Args:
            feature_names: Ordered feature names.
            x_test: Test feature matrix.
            y_test: Test targets (unused but retained for API completeness).

        Returns:
            None.
        """
        _ = y_test
        out_path = Path("./outputs/shap_feature_importance.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sample_x = x_test.iloc[:200]

        try:
            explainer = shap.TreeExplainer(self.model)
            shap_values = explainer.shap_values(sample_x)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(
                shap_values,
                sample_x,
                feature_names=feature_names,
                show=False,
            )
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[SAVED] SHAP plot -> {out_path.as_posix()}")
        except Exception as exc:  # pylint: disable=broad-except
            # Fallback keeps training pipeline robust if SHAP/XGBoost versions conflict.
            importances = self.model.feature_importances_
            order = np.argsort(importances)[::-1][:20]
            plt.figure(figsize=(10, 6))
            plt.barh(
                [feature_names[i] for i in order][::-1],
                importances[order][::-1],
            )
            plt.title("Feature Importance (XGBoost Fallback)")
            plt.xlabel("Importance")
            plt.tight_layout()
            plt.savefig(out_path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"[WARN] SHAP failed: {exc}. Used fallback importance plot.")

        print(f"[SAVED] {out_path.as_posix()}")

    def save(self, path: str = "./models/xgboost_model.pkl") -> None:
        """Save trained XGBoost model.

        Args:
            path: Output pickle path.

        Returns:
            None.
        """
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, out_path)
        print(f"[SAVED] {out_path.as_posix()}")

