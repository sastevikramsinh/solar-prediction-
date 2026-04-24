"""Evaluation and visualization utilities."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Compute core regression metrics.

    Args:
        y_true: Ground truth targets.
        y_pred: Predicted targets.
        model_name: Model display name.

    Returns:
        Dictionary of rounded metric values.
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    metrics = {
        "model": model_name,
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
        "MAPE": round(mape, 2),
    }
    print(f"\n{'=' * 50}\n{model_name} Results:\n{metrics}\n{'=' * 50}")
    return metrics


def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Plot and save actual vs predicted chart for last 150 points.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        model_name: Model display name for output naming.

    Returns:
        None.
    """
    sns.set_style("whitegrid")
    n_plot = min(150, len(y_true), len(y_pred))
    y_true_tail = np.array(y_true)[-n_plot:]
    y_pred_tail = np.array(y_pred)[-n_plot:]

    plt.figure(figsize=(14, 5))
    plt.plot(y_true_tail, label="Actual", linewidth=2.0)
    plt.plot(y_pred_tail, label="Predicted", linewidth=2.0, alpha=0.9)
    plt.title(f"{model_name} - Actual vs Predicted (Last {n_plot} points)")
    plt.xlabel("Time Step")
    plt.ylabel("AC Power (kW)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    out_path = Path(f"./outputs/{model_name}_predictions.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {out_path.as_posix()}")


def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> None:
    """Plot residual scatter and histogram diagnostics.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        model_name: Model display name for output naming.

    Returns:
        None.
    """
    residuals = np.array(y_true) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].scatter(y_pred, residuals, alpha=0.5)
    axes[0].axhline(0, color="red", linestyle="--", linewidth=1.2)
    axes[0].set_title(f"{model_name} - Residuals vs Predictions")
    axes[0].set_xlabel("Predicted AC Power (kW)")
    axes[0].set_ylabel("Residuals")
    axes[0].grid(alpha=0.3)

    axes[1].hist(residuals, bins=30, alpha=0.8)
    axes[1].set_title(f"{model_name} - Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    out_path = Path(f"./outputs/{model_name}_residuals.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[SAVED] {out_path.as_posix()}")


def plot_all_models_comparison(results_list: list[dict]) -> None:
    """Plot comparison chart and persist final leaderboard results.

    Args:
        results_list: List of model metric dictionaries.

    Returns:
        None.
    """
    model_names = [r["model"] for r in results_list]
    rmse_vals = [r["RMSE"] for r in results_list]
    r2_vals = [r["R2"] for r in results_list]

    color_map = {
        "Linear Regression": "gray",
        "ANN": "teal",
        "XGBoost": "orange",
        "CNN-LSTM": "purple",
    }
    colors = [color_map.get(name, "steelblue") for name in model_names]

    x = np.arange(len(model_names))
    width = 0.38

    plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, rmse_vals, width=width, label="RMSE", color=colors, alpha=0.85)
    plt.bar(
        x + width / 2,
        r2_vals,
        width=width,
        label="R2",
        color=colors,
        alpha=0.45,
        hatch="//",
    )
    plt.xticks(x, model_names, rotation=10)
    plt.ylabel("Metric Value")
    plt.title("Model Comparison: RMSE and R2")
    plt.legend()
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    chart_path = Path("./outputs/model_comparison_chart.png")
    chart_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {chart_path.as_posix()}")

    print("\n" + "=" * 65)
    print("  FINAL MODEL LEADERBOARD (ranked by lower RMSE = better)")
    print("=" * 65)
    sorted_results = sorted(results_list, key=lambda item: item["RMSE"])
    for i, result in enumerate(sorted_results):
        print(
            f"  #{i + 1}  {result['model']:<25} "
            f"RMSE:{result['RMSE']:<10} R2:{result['R2']:<8} MAPE:{result['MAPE']}%"
        )
    print("=" * 65)

    json_path = Path("./outputs/all_model_results.json")
    with json_path.open("w", encoding="utf-8") as fp:
        json.dump(results_list, fp, indent=2)
    print(f"[SAVED] {json_path.as_posix()}")

