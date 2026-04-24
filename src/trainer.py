"""Data splitting and scaling helpers for model training."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

TARGET_COLUMN = "AC_POWER"


def chronological_split(
    df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Split dataframe into chronological train/validation/test partitions.

    Args:
        df: Feature-engineered dataframe including target column.
        val_size: Validation proportion.
        test_size: Test proportion.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test.
    """
    if "DATE_TIME" in df.columns:
        sorted_df = df.sort_values("DATE_TIME").reset_index(drop=True)
    else:
        sorted_df = df.sort_index().reset_index(drop=True)
    n_rows = len(sorted_df)
    train_end = int(n_rows * (1 - val_size - test_size))
    val_end = int(n_rows * (1 - test_size))

    train_df = sorted_df.iloc[:train_end]
    val_df = sorted_df.iloc[train_end:val_end]
    test_df = sorted_df.iloc[val_end:]

    feature_cols = [
        col for col in sorted_df.columns if col not in [TARGET_COLUMN, "DATE_TIME", "PLANT"]
    ]

    x_train = train_df[feature_cols]
    x_val = val_df[feature_cols]
    x_test = test_df[feature_cols]
    y_train = train_df[TARGET_COLUMN]
    y_val = val_df[TARGET_COLUMN]
    y_test = test_df[TARGET_COLUMN]

    print(
        "Split sizes -> "
        f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}"
    )
    print(
        "Index ranges -> "
        f"Train: [0, {train_end - 1}], "
        f"Val: [{train_end}, {val_end - 1}], "
        f"Test: [{val_end}, {n_rows - 1}]"
    )
    print("No overlap check:", train_end <= val_end <= n_rows)

    return x_train, x_val, x_test, y_train, y_val, y_test


def get_scaled_data(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> tuple:
    """Scale train/validation/test features and target using train-only fit.

    Args:
        x_train: Training features.
        x_val: Validation features.
        x_test: Test features.
        y_train: Training target.
        y_val: Validation target.
        y_test: Test target.

    Returns:
        Scaled arrays plus original unscaled feature/target splits.
    """
    model_dir = Path("./models")
    model_dir.mkdir(parents=True, exist_ok=True)

    scaler = MinMaxScaler()
    x_train_sc = scaler.fit_transform(x_train)
    x_val_sc = scaler.transform(x_val)
    x_test_sc = scaler.transform(x_test)
    scaler_path = model_dir / "scaler.pkl"
    joblib.dump(scaler, scaler_path)
    print(f"[SAVED] {scaler_path.as_posix()}")

    y_scaler = MinMaxScaler()
    y_train_sc = y_scaler.fit_transform(y_train.values.reshape(-1, 1)).ravel()
    y_val_sc = y_scaler.transform(y_val.values.reshape(-1, 1)).ravel()
    y_test_sc = y_scaler.transform(y_test.values.reshape(-1, 1)).ravel()
    y_scaler_path = model_dir / "y_scaler.pkl"
    joblib.dump(y_scaler, y_scaler_path)
    print(f"[SAVED] {y_scaler_path.as_posix()}")

    print("[OK] Scalers saved. Feature scaler fit on training data only (no leakage).")
    return (
        x_train_sc,
        x_val_sc,
        x_test_sc,
        y_train_sc,
        y_val_sc,
        y_test_sc,
        x_train,
        x_val,
        x_test,
        y_train,
        y_val,
        y_test,
    )

