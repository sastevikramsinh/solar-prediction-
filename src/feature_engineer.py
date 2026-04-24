"""Feature engineering for solar power prediction."""

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create temporal, interaction, lag, and rolling features.

    Args:
        df: Cleaned dataframe from preprocessing stage.

    Returns:
        Feature-engineered dataframe containing target AC_POWER.
    """
    featured_df = df.copy()
    featured_df["DATE_TIME"] = pd.to_datetime(featured_df["DATE_TIME"], errors="coerce")

    featured_df["hour"] = featured_df["DATE_TIME"].dt.hour
    featured_df["month"] = featured_df["DATE_TIME"].dt.month
    featured_df["day_of_year"] = featured_df["DATE_TIME"].dt.dayofyear

    featured_df["sin_hour"] = np.sin(2 * np.pi * featured_df["hour"] / 24)
    featured_df["cos_hour"] = np.cos(2 * np.pi * featured_df["hour"] / 24)
    featured_df["sin_month"] = np.sin(2 * np.pi * featured_df["month"] / 12)
    featured_df["cos_month"] = np.cos(2 * np.pi * featured_df["month"] / 12)

    featured_df["TEMP_DELTA"] = (
        featured_df["MODULE_TEMPERATURE"] - featured_df["AMBIENT_TEMPERATURE"]
    )
    featured_df["IRRAD_TEMP"] = (
        featured_df["IRRADIATION"] * featured_df["AMBIENT_TEMPERATURE"]
    )
    featured_df["IRRAD_SQUARED"] = featured_df["IRRADIATION"] ** 2
    featured_df["WIND_IRRAD_INTERACTION"] = (
        featured_df["WIND_SPEED_10M"] * featured_df["IRRADIATION"]
    )
    featured_df["WIND_TEMP_INTERACTION"] = (
        featured_df["WIND_SPEED_10M"] * featured_df["AMBIENT_TEMPERATURE"]
    )

    featured_df = featured_df.sort_values(["PLANT", "DATE_TIME"])
    for lag in [1, 2, 3]:
        featured_df[f"AC_POWER_lag_{lag}"] = featured_df.groupby("PLANT")["AC_POWER"].shift(
            lag
        )
    featured_df["IRRAD_lag_1"] = featured_df.groupby("PLANT")["IRRADIATION"].shift(1)

    featured_df["rolling_mean_irrad_4"] = featured_df.groupby("PLANT")[
        "IRRADIATION"
    ].transform(lambda x: x.rolling(window=4, min_periods=1).mean())
    featured_df["rolling_max_irrad_4"] = featured_df.groupby("PLANT")[
        "IRRADIATION"
    ].transform(lambda x: x.rolling(window=4, min_periods=1).max())
    featured_df["rolling_mean_wind_4"] = featured_df.groupby("PLANT")[
        "WIND_SPEED_10M"
    ].transform(lambda x: x.rolling(window=4, min_periods=1).mean())

    drop_cols = ["PLANT_ID", "SOURCE_KEY", "DAILY_YIELD"]
    existing_drop_cols = [col for col in drop_cols if col in featured_df.columns]
    featured_df = featured_df.drop(columns=existing_drop_cols)
    featured_df = featured_df.dropna().reset_index(drop=True)

    print(f"Feature engineering complete. Shape: {featured_df.shape}")
    print(f"Features: {[c for c in featured_df.columns if c != 'AC_POWER']}")

    return featured_df

