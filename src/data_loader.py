"""Data loading and merging utilities for solar datasets."""

from pathlib import Path

import pandas as pd
from src.wind_data import align_wind_to_generation_timestamps, download_wind_data_for_plant


def _parse_datetime(series: pd.Series, fmt: str) -> pd.Series:
    """Parse datetime series with coercion.

    Args:
        series: Raw datetime column.
        fmt: Datetime format string.

    Returns:
        Parsed datetime series.
    """
    return pd.to_datetime(series, format=fmt, errors="coerce")


def load_and_merge_data(data_dir: str) -> pd.DataFrame:
    """Load all plant CSVs, aggregate inverter generation, and merge with weather.

    Args:
        data_dir: Directory containing the 4 raw CSV files.

    Returns:
        Merged plant-level dataframe across both plants.
    """
    data_path = Path(data_dir)
    p1_gen_path = data_path / "Plant_1_Generation_Data.csv"
    p1_weather_path = data_path / "Plant_1_Weather_Sensor_Data.csv"
    p2_gen_path = data_path / "Plant_2_Generation_Data.csv"
    p2_weather_path = data_path / "Plant_2_Weather_Sensor_Data.csv"

    required_files = [p1_gen_path, p1_weather_path, p2_gen_path, p2_weather_path]
    missing_files = [str(path) for path in required_files if not path.exists()]
    if missing_files:
        missing_joined = "\n".join(missing_files)
        raise FileNotFoundError(
            f"Missing required data files:\n{missing_joined}\n"
            f"Expected under: {data_path.resolve()}"
        )

    try:
        plant1_gen = pd.read_csv(p1_gen_path)
        plant1_weather = pd.read_csv(p1_weather_path)
        plant2_gen = pd.read_csv(p2_gen_path)
        plant2_weather = pd.read_csv(p2_weather_path)
    except Exception as exc:  # pylint: disable=broad-except
        raise RuntimeError(f"Failed reading CSV files from {data_path.resolve()}") from exc

    plant1_gen["DATE_TIME"] = _parse_datetime(
        plant1_gen["DATE_TIME"], fmt="%d-%m-%Y %H:%M"
    )
    plant1_weather["DATE_TIME"] = _parse_datetime(
        plant1_weather["DATE_TIME"], fmt="%Y-%m-%d %H:%M:%S"
    )
    plant2_gen["DATE_TIME"] = _parse_datetime(
        plant2_gen["DATE_TIME"], fmt="%Y-%m-%d %H:%M:%S"
    )
    plant2_weather["DATE_TIME"] = _parse_datetime(
        plant2_weather["DATE_TIME"], fmt="%Y-%m-%d %H:%M:%S"
    )

    plant1_gen_agg = (
        plant1_gen.groupby(["DATE_TIME", "PLANT_ID"])
        .agg(
            DC_POWER=("DC_POWER", "sum"),
            AC_POWER=("AC_POWER", "sum"),
            DAILY_YIELD=("DAILY_YIELD", "sum"),
        )
        .reset_index()
    )
    plant2_gen_agg = (
        plant2_gen.groupby(["DATE_TIME", "PLANT_ID"])
        .agg(
            DC_POWER=("DC_POWER", "sum"),
            AC_POWER=("AC_POWER", "sum"),
            DAILY_YIELD=("DAILY_YIELD", "sum"),
        )
        .reset_index()
    )

    weather_cols = [
        "DATE_TIME",
        "AMBIENT_TEMPERATURE",
        "MODULE_TEMPERATURE",
        "IRRADIATION",
    ]

    plant1_merged = pd.merge(
        plant1_gen_agg,
        plant1_weather[weather_cols],
        on="DATE_TIME",
        how="inner",
    )
    plant1_merged["PLANT"] = "Plant_1"

    plant2_merged = pd.merge(
        plant2_gen_agg,
        plant2_weather[weather_cols],
        on="DATE_TIME",
        how="inner",
    )
    plant2_merged["PLANT"] = "Plant_2"

    start_date = min(
        plant1_merged["DATE_TIME"].min(),
        plant2_merged["DATE_TIME"].min(),
    ).date()
    end_date = max(
        plant1_merged["DATE_TIME"].max(),
        plant2_merged["DATE_TIME"].max(),
    ).date()
    wind_cache_dir = Path("./data/external")
    p1_wind = download_wind_data_for_plant(
        "Plant_1", str(start_date), str(end_date), wind_cache_dir
    )
    p2_wind = download_wind_data_for_plant(
        "Plant_2", str(start_date), str(end_date), wind_cache_dir
    )

    p1_wind_aligned = align_wind_to_generation_timestamps(
        p1_wind, plant1_merged["DATE_TIME"]
    )
    p2_wind_aligned = align_wind_to_generation_timestamps(
        p2_wind, plant2_merged["DATE_TIME"]
    )

    plant1_merged = plant1_merged.merge(p1_wind_aligned, on="DATE_TIME", how="left")
    plant2_merged = plant2_merged.merge(p2_wind_aligned, on="DATE_TIME", how="left")

    df = pd.concat([plant1_merged, plant2_merged], ignore_index=True)
    df = df.sort_values("DATE_TIME").reset_index(drop=True)

    print(f"Total rows after merge: {len(df)}")
    print(f"Date range: {df['DATE_TIME'].min()} to {df['DATE_TIME'].max()}")
    print(f"Columns: {df.columns.tolist()}")
    print(
        df[
            [
                "AC_POWER",
                "IRRADIATION",
                "AMBIENT_TEMPERATURE",
                "WIND_SPEED_10M",
            ]
        ].describe()
    )

    return df

