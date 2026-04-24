"""Export trained ANN model to TensorFlow Lite for offline Android inference."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf


def _build_scaler_snapshot(scaler) -> dict:
    """Extract scaler parameters needed for on-device min-max normalization."""
    return {
        "feature_range": list(scaler.feature_range),
        "min_": scaler.min_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "data_min_": scaler.data_min_.tolist(),
        "data_max_": scaler.data_max_.tolist(),
    }


def export_ann_offline_bundle(models_dir: str = "./models", out_dir: str = "./models/offline") -> dict:
    """Create offline bundle: TFLite ANN + normalization metadata.

    Args:
        models_dir: Directory containing trained ANN and scalers.
        out_dir: Directory for generated offline artifacts.

    Returns:
        Dictionary with output file paths and metadata.
    """
    models_path = Path(models_dir)
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    ann_path = models_path / "ann_final.keras"
    x_scaler_path = models_path / "scaler.pkl"
    y_scaler_path = models_path / "y_scaler.pkl"
    meta_path = models_path / "production_metadata.json"

    missing = [p.as_posix() for p in [ann_path, x_scaler_path, y_scaler_path, meta_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required model artifacts for offline conversion:\n" + "\n".join(missing)
        )

    model = tf.keras.models.load_model(ann_path)
    x_scaler = joblib.load(x_scaler_path)
    y_scaler = joblib.load(y_scaler_path)
    with meta_path.open("r", encoding="utf-8") as fp:
        production_meta = json.load(fp)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_bytes = converter.convert()

    tflite_path = output_path / "solar_ann_offline.tflite"
    tflite_path.write_bytes(tflite_bytes)

    offline_meta = {
        "model_type": "ANN (TFLite)",
        "input_feature_columns": production_meta.get("feature_columns", []),
        "x_scaler": _build_scaler_snapshot(x_scaler),
        "y_scaler": _build_scaler_snapshot(y_scaler),
        "feature_medians": production_meta.get("feature_medians", {}),
        "notes": [
            "Use this model for offline inference on Android.",
            "Apply x_scaler min-max normalization before inference.",
            "Apply inverse y_scaler transform after inference.",
            "ANN is used for offline deployment; XGBoost remains strongest online model.",
        ],
    }

    offline_meta_path = output_path / "offline_metadata.json"
    with offline_meta_path.open("w", encoding="utf-8") as fp:
        json.dump(offline_meta, fp, indent=2)

    return {
        "tflite_model": tflite_path.as_posix(),
        "offline_metadata": offline_meta_path.as_posix(),
    }


if __name__ == "__main__":
    artifacts = export_ann_offline_bundle()
    print(f"[SAVED] {artifacts['tflite_model']}")
    print(f"[SAVED] {artifacts['offline_metadata']}")

