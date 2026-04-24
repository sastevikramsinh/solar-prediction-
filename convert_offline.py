"""CLI entry point for offline model conversion."""

from src.offline.export_tflite import export_ann_offline_bundle


if __name__ == "__main__":
    paths = export_ann_offline_bundle(models_dir="./models", out_dir="./models/offline")
    print(f"[SAVED] {paths['tflite_model']}")
    print(f"[SAVED] {paths['offline_metadata']}")

