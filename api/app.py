"""FastAPI inference service and animated web UI host."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
import urllib.parse
import urllib.request
import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi import HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field


MODELS_DIR = Path("./models")
WEB_DIR = Path("./web")
MODEL_PATH = MODELS_DIR / "xgboost_model.pkl"
META_PATH = MODELS_DIR / "production_metadata.json"


def _load_artifacts() -> tuple[object, dict]:
    """Load production model and metadata from disk."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Missing model artifact. Train first with `python train.py`.")
    if not META_PATH.exists():
        raise FileNotFoundError("Missing metadata artifact. Train first with `python train.py`.")
    model = joblib.load(MODEL_PATH)
    with META_PATH.open("r", encoding="utf-8") as fp:
        metadata = json.load(fp)
    return model, metadata


model, metadata = _load_artifacts()
feature_columns = metadata["feature_columns"]
feature_defaults = metadata["feature_medians"]

app = FastAPI(title="Solar + Wind Prediction API", version="1.0.0")
app.mount("/static", StaticFiles(directory=WEB_DIR / "static"), name="static")

cors_origins_env = os.getenv("CORS_ALLOW_ORIGINS", "").strip()
if cors_origins_env:
    allow_origins = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]
else:
    # Safe default for quick setup; tighten via CORS_ALLOW_ORIGINS in production.
    allow_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PredictionRequest(BaseModel):
    """Input payload for prediction endpoint."""

    datetime_iso: str = Field(..., description="ISO datetime, e.g. 2020-06-01T12:15:00")
    dc_power: float = Field(..., ge=0.0, le=50000.0)
    ambient_temperature: float = Field(..., ge=-30.0, le=80.0)
    module_temperature: float = Field(..., ge=-30.0, le=100.0)
    irradiation: float = Field(..., ge=0.0, le=1.5)
    wind_speed_10m: float = Field(..., ge=0.0, le=80.0)
    ac_power_lag_1: float | None = None
    ac_power_lag_2: float | None = None
    ac_power_lag_3: float | None = None
    irrad_lag_1: float | None = None


def _build_feature_row(req: PredictionRequest) -> pd.DataFrame:
    """Build a model-ready feature row from request values."""
    dt_parsed = pd.to_datetime(req.datetime_iso, errors="coerce")
    if pd.isna(dt_parsed):
        raise ValueError(
            "Invalid datetime format. Use ISO like 2026-04-24T11:57:00 or similar valid datetime."
        )
    dt = dt_parsed.to_pydatetime()
    row = {k: feature_defaults.get(k, 0.0) for k in feature_columns}

    row["DC_POWER"] = req.dc_power
    row["AMBIENT_TEMPERATURE"] = req.ambient_temperature
    row["MODULE_TEMPERATURE"] = req.module_temperature
    row["IRRADIATION"] = req.irradiation
    row["WIND_SPEED_10M"] = req.wind_speed_10m

    row["hour"] = dt.hour
    row["month"] = dt.month
    row["day_of_year"] = dt.timetuple().tm_yday
    row["sin_hour"] = float(np.sin(2 * np.pi * row["hour"] / 24))
    row["cos_hour"] = float(np.cos(2 * np.pi * row["hour"] / 24))
    row["sin_month"] = float(np.sin(2 * np.pi * row["month"] / 12))
    row["cos_month"] = float(np.cos(2 * np.pi * row["month"] / 12))

    row["TEMP_DELTA"] = row["MODULE_TEMPERATURE"] - row["AMBIENT_TEMPERATURE"]
    row["IRRAD_TEMP"] = row["IRRADIATION"] * row["AMBIENT_TEMPERATURE"]
    row["IRRAD_SQUARED"] = row["IRRADIATION"] ** 2
    row["WIND_IRRAD_INTERACTION"] = row["WIND_SPEED_10M"] * row["IRRADIATION"]
    row["WIND_TEMP_INTERACTION"] = row["WIND_SPEED_10M"] * row["AMBIENT_TEMPERATURE"]

    if "AC_POWER_lag_1" in row and req.ac_power_lag_1 is not None:
        row["AC_POWER_lag_1"] = req.ac_power_lag_1
    if "AC_POWER_lag_2" in row and req.ac_power_lag_2 is not None:
        row["AC_POWER_lag_2"] = req.ac_power_lag_2
    if "AC_POWER_lag_3" in row and req.ac_power_lag_3 is not None:
        row["AC_POWER_lag_3"] = req.ac_power_lag_3
    if "IRRAD_lag_1" in row and req.irrad_lag_1 is not None:
        row["IRRAD_lag_1"] = req.irrad_lag_1

    df = pd.DataFrame([row], columns=feature_columns)
    return df


@app.get("/")
def serve_home() -> FileResponse:
    """Serve the animated dashboard."""
    return FileResponse(WEB_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "model": metadata["best_model"]}


@app.get("/model-info")
def model_info() -> dict:
    """Return production model metadata."""
    return metadata


@app.post("/predict")
def predict(payload: PredictionRequest) -> dict:
    """Predict AC power (kW) for a single feature row."""
    try:
        # Physics guard: no sunlight implies no solar generation.
        if payload.irradiation <= 0.01:
            return {"predicted_ac_power_kw": 0.0}
        x = _build_feature_row(payload)
        pred = float(model.predict(x)[0])
        # Fallback for environments/models that can emit non-physical negatives.
        if pred <= 0.0:
            wind_cooling_factor = max(0.65, 1.0 - 0.03 * (payload.wind_speed_10m / 12.0))
            pred = float(np.clip(payload.irradiation * 13000.0 * wind_cooling_factor, 0.0, 30000.0))
        return {"predicted_ac_power_kw": round(pred, 3)}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}") from exc


@app.post("/simulate-wind")
def simulate_wind(payload: PredictionRequest) -> dict:
    """Generate a wind sweep to visualize impact on predicted AC power."""
    try:
        if payload.irradiation <= 0.01:
            return {
                "curve": [
                    {"wind_speed_10m": round(float(wind), 2), "predicted_kw": 0.0}
                    for wind in np.linspace(0, 12, 49)
                ]
            }
        sweep = []
        for wind in np.linspace(0, 12, 49):
            mutable = payload.model_copy(update={"wind_speed_10m": float(wind)})
            x = _build_feature_row(mutable)
            pred = float(model.predict(x)[0])
            sweep.append(
                {"wind_speed_10m": round(float(wind), 2), "predicted_kw": round(pred, 3)}
            )
        return {"curve": sweep}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=500, detail=f"Wind simulation failed: {exc}") from exc


@app.get("/factors")
def factors_page() -> FileResponse:
    """Serve interactive factors animation page."""
    return FileResponse(WEB_DIR / "factors.html")


@app.get("/live-context")
def live_context(lat: float, lon: float) -> dict:
    """Fetch live weather for coordinates and map it to model-ready defaults."""
    try:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": ",".join(
                [
                    "temperature_2m",
                    "wind_speed_10m",
                    "cloud_cover",
                    "shortwave_radiation",
                    "is_day",
                ]
            ),
            "timezone": "auto",
        }
        url = f"https://api.open-meteo.com/v1/forecast?{urllib.parse.urlencode(params)}"
        with urllib.request.urlopen(url, timeout=30) as response:  # nosec B310
            payload = json.loads(response.read().decode("utf-8"))
    except Exception as exc:  # pylint: disable=broad-except
        raise HTTPException(status_code=502, detail=f"Weather provider error: {exc}") from exc

    current = payload.get("current", {})
    temp = float(current.get("temperature_2m", 30.0))
    wind = float(current.get("wind_speed_10m", 3.0))
    cloud = float(current.get("cloud_cover", 20.0))
    sw = float(current.get("shortwave_radiation", 650.0))
    dt_iso = str(current.get("time", datetime.utcnow().isoformat()))

    irradiation = float(np.clip(sw / 1000.0, 0.0, 1.2))
    # Plant-scale DC proxy from irradiance for first prediction seed.
    dc_power = float(max(0.0, irradiation * 13000.0))
    if irradiation <= 0.01:
        dc_power = 0.0
    module_temp = float(np.clip(temp + 4.0 + irradiation * 10.0 - 0.08 * wind, 20.0, 65.0))

    return {
        "datetime_iso": dt_iso,
        "dc_power": round(dc_power, 3),
        "ambient_temperature": round(temp, 3),
        "module_temperature": round(module_temp, 3),
        "irradiation": round(irradiation, 4),
        "wind_speed_10m": round(wind, 3),
        "cloud_cover": round(cloud, 2),
        "source": "open-meteo",
        "provider_time_local": dt_iso,
        "raw_shortwave_radiation_wm2": round(sw, 2),
    }

