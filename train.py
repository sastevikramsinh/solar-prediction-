"""Main training entry point for the solar prediction project."""

import os
import time
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import joblib
import numpy as np
import tensorflow as tf
import json

from src.data_loader import load_and_merge_data
from src.evaluator import (
    compute_metrics,
    plot_all_models_comparison,
    plot_predictions,
    plot_residuals,
)
from src.feature_engineer import engineer_features
from src.models.ann_model import ANNModel
from src.models.baseline_regression import LinearRegressionModel
from src.models.cnn_lstm_model import CNNLSTMModel
from src.models.xgboost_model import XGBoostModel
from src.preprocessor import preprocess
from src.trainer import chronological_split, get_scaled_data

CONFIG = {
    "DATA_DIR": "./data/raw",
    "WINDOW_SIZE": 12,
    "VAL_SIZE": 0.15,
    "TEST_SIZE": 0.15,
    "RANDOM_STATE": 42,
    "BATCH_SIZE": 32,
    "MAX_EPOCHS": 200,
}


def run_pipeline() -> None:
    """Execute complete ML pipeline from data loading to model comparison.

    Args:
        None.

    Returns:
        None.
    """
    np.random.seed(CONFIG["RANDOM_STATE"])
    tf.random.set_seed(CONFIG["RANDOM_STATE"])

    Path("./models").mkdir(exist_ok=True)
    Path("./outputs").mkdir(exist_ok=True)

    df = load_and_merge_data(CONFIG["DATA_DIR"])
    df_clean, cleaning_stats = preprocess(df)
    print(f"Cleaning stats: {cleaning_stats}")

    df_featured = engineer_features(df_clean)

    x_train, x_val, x_test, y_train, y_val, y_test = chronological_split(
        df_featured, val_size=CONFIG["VAL_SIZE"], test_size=CONFIG["TEST_SIZE"]
    )

    (
        x_train_sc,
        x_val_sc,
        x_test_sc,
        y_train_sc,
        y_val_sc,
        y_test_sc,
        x_train_raw,
        x_val_raw,
        x_test_raw,
        y_train_raw,
        y_val_raw,
        y_test_raw,
    ) = get_scaled_data(x_train, x_val, x_test, y_train, y_val, y_test)

    y_scaler = joblib.load("./models/y_scaler.pkl")

    print("\n" + "=" * 70)
    print("Training Model 1/4: Linear Regression")
    print("=" * 70)
    lr_model = LinearRegressionModel()
    lr_model.train(x_train_raw, y_train_raw)
    y_pred_lr = lr_model.predict(x_test_raw)
    lr_metrics = compute_metrics(y_test_raw.values, y_pred_lr, "Linear Regression")
    plot_predictions(y_test_raw.values, y_pred_lr, "Linear Regression")
    plot_residuals(y_test_raw.values, y_pred_lr, "Linear Regression")
    lr_model.save()

    print("\n" + "=" * 70)
    print("Training Model 2/4: ANN")
    print("=" * 70)
    ann_model = ANNModel()
    ann_model.build_model(input_dim=x_train_sc.shape[1])
    ann_model.train(x_train_sc, y_train_sc, x_val_sc, y_val_sc)
    y_pred_ann = ann_model.predict(x_test_sc, y_scaler)
    ann_metrics = compute_metrics(y_test_raw.values, y_pred_ann, "ANN")
    plot_predictions(y_test_raw.values, y_pred_ann, "ANN")
    plot_residuals(y_test_raw.values, y_pred_ann, "ANN")
    ann_model.save()

    print("\n" + "=" * 70)
    print("Training Model 3/4: XGBoost")
    print("=" * 70)
    xgb_model = XGBoostModel()
    xgb_model.train(x_train_raw, y_train_raw, x_val_raw, y_val_raw)
    y_pred_xgb = xgb_model.predict(x_test_raw)
    xgb_metrics = compute_metrics(y_test_raw.values, y_pred_xgb, "XGBoost")
    xgb_model.plot_feature_importance(
        feature_names=list(x_test_raw.columns),
        x_test=x_test_raw,
        y_test=y_test_raw,
    )
    plot_predictions(y_test_raw.values, y_pred_xgb, "XGBoost")
    plot_residuals(y_test_raw.values, y_pred_xgb, "XGBoost")
    xgb_model.save()

    print("\n" + "=" * 70)
    print("Training Model 4/4: CNN-LSTM (Proposed Model)")
    print("=" * 70)
    cnn_lstm_model = CNNLSTMModel(window_size=CONFIG["WINDOW_SIZE"])
    cnn_lstm_model.build_model(
        n_features=x_train_sc.shape[1], window_size=CONFIG["WINDOW_SIZE"]
    )
    _, _, x_test_seq = cnn_lstm_model.train(
        x_train_sc, y_train_sc, x_val_sc, y_val_sc, x_test_sc, y_test_sc
    )
    y_test_seq = y_test_raw.values[CONFIG["WINDOW_SIZE"] :]
    y_pred_cnnlstm = cnn_lstm_model.predict(x_test_seq, y_scaler)
    cnn_metrics = compute_metrics(y_test_seq, y_pred_cnnlstm, "CNN-LSTM")
    plot_predictions(y_test_seq, y_pred_cnnlstm, "CNN-LSTM")
    plot_residuals(y_test_seq, y_pred_cnnlstm, "CNN-LSTM")
    cnn_lstm_model.save()

    results = [lr_metrics, ann_metrics, xgb_metrics, cnn_metrics]
    plot_all_models_comparison(results)

    best = min(results, key=lambda x: x["RMSE"])
    production_meta = {
        "best_model": best["model"],
        "metrics": best,
        "feature_columns": list(x_train_raw.columns),
        "feature_medians": x_train_raw.median(numeric_only=True).to_dict(),
    }
    with open("./models/production_metadata.json", "w", encoding="utf-8") as fp:
        json.dump(production_meta, fp, indent=2)
    print("[SAVED] models/production_metadata.json")
    print(f"[OK] Production model selected: {best['model']}")


if __name__ == "__main__":
    start = time.time()
    run_pipeline()
    print(f"\nTotal pipeline time: {(time.time() - start) / 60:.1f} minutes")

