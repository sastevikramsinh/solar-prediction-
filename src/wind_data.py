"""Wind data download and alignment utilities."""

from __future__ import annotations

import json
import urllib.parse
import urllib.request
from pathlib import Path

import pandas as pd


PLANT_COORDS: dict[str, tuple[float, float]] = {
    # Update these coordinates with exact plant coordinates if available.
    "Plant_1": (28.6139, 77.2090),
    "Plant_2": (23.0225, 72.5714),
}


def download_wind_data_for_plant(
    plant_name: str,
    start_date: str,
    end_date: str,
    out_dir: Path,
) -> pd.DataFrame:
    """Download archived hourly wind speed and cache to CSV.

    Args:
        plant_name: Plant key in `PLANT_COORDS`.
        start_date: Inclusive start date (YYYY-MM-DD).
        end_date: Inclusive end date (YYYY-MM-DD).
        out_dir: Directory for cached CSV output.

    Returns:
        DataFrame with `DATE_TIME` and `WIND_SPEED_10M`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{plant_name}_wind_openmeteo.csv"
    if out_path.exists():
        cached = pd.read_csv(out_path)
        cached["DATE_TIME"] = pd.to_datetime(cached["DATE_TIME"], errors="coerce")
        return cached

    lat, lon = PLANT_COORDS[plant_name]
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "wind_speed_10m",
        "timezone": "auto",
    }
    url = f"https://archive-api.open-meteo.com/v1/archive?{urllib.parse.urlencode(params)}"

    with urllib.request.urlopen(url, timeout=60) as response:  # nosec B310
        payload = json.loads(response.read().decode("utf-8"))

    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    wind = hourly.get("wind_speed_10m", [])
    if not times or not wind:
        raise RuntimeError(f"Open-Meteo returned empty wind data for {plant_name}.")

    wind_df = pd.DataFrame(
        {
            "DATE_TIME": pd.to_datetime(times, errors="coerce"),
            "WIND_SPEED_10M": pd.to_numeric(wind, errors="coerce"),
        }
    ).dropna()
    wind_df.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path.as_posix()}")
    return wind_df


def align_wind_to_generation_timestamps(
    wind_df: pd.DataFrame, target_timestamps: pd.Series
) -> pd.DataFrame:
    """Resample/interpolate hourly wind to generation timestamps.

    Args:
        wind_df: Hourly wind data.
        target_timestamps: Target 15-min timestamps for merge.

    Returns:
        DataFrame with `DATE_TIME` and aligned `WIND_SPEED_10M`.
    """
    wind_df = wind_df.copy()
    wind_df = wind_df.sort_values("DATE_TIME")
    target_df = pd.DataFrame(
        {"DATE_TIME": pd.to_datetime(target_timestamps.unique(), errors="coerce")}
    ).sort_values("DATE_TIME")

    full = pd.concat([wind_df[["DATE_TIME"]], target_df], ignore_index=True).drop_duplicates()
    full = full.sort_values("DATE_TIME")
    full = full.merge(wind_df, on="DATE_TIME", how="left")
    full = full.set_index("DATE_TIME")
    full["WIND_SPEED_10M"] = full["WIND_SPEED_10M"].interpolate(
        method="time", limit_direction="both"
    )
    full = full.reset_index()
    full = full[full["DATE_TIME"].isin(target_df["DATE_TIME"])]
    return full[["DATE_TIME", "WIND_SPEED_10M"]]

