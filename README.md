# Solar Prediction Project

Solar power forecasting application with:

- a FastAPI backend for inference
- a modern responsive web UI for manual and live-weather predictions
- an interactive sun/cloud factors simulation page
- model training pipeline (XGBoost, ANN, CNN-LSTM, baseline regression)
- optional Android app scaffold and offline TFLite export tooling

## Tech Stack

- Python, FastAPI, Uvicorn
- Pandas, NumPy, scikit-learn, XGBoost, TensorFlow
- Vanilla HTML/CSS/JavaScript frontend

## Project Structure

- `api/` - FastAPI app (`api/app.py`)
- `web/` - frontend pages and static assets
- `src/` - data, preprocessing, feature engineering, model and training code
- `models/` - trained artifacts and metadata used by inference
- `data/` - raw and external datasets
- `outputs/` - plots and evaluation outputs
- `android_app/` - Android client scaffold

## Local Setup

1. Create and activate a virtual environment (recommended).
2. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

3. Run the API + web host:

```bash
python -m uvicorn api.app:app --reload
```

4. Open in browser:

- Home: `http://127.0.0.1:8000/`
- Factors page: `http://127.0.0.1:8000/factors`
- API docs: `http://127.0.0.1:8000/docs`

## Training

To retrain and regenerate model artifacts:

```bash
python train.py
```

This updates model files in `models/` and reports/plots in `outputs/`.

## API Endpoints

- `GET /health` - service health and active model info
- `POST /predict` - single prediction request
- `POST /simulate-wind` - wind sweep curve from a base payload
- `GET /live-context?lat=...&lon=...` - live weather mapped to model-ready fields
- `GET /model-info` - production metadata

## Input Notes

- `irradiation` is expected in approximately `0.0` to `1.2` range (not percent).
- If irradiation is near zero, prediction will be zero by design.
- UI and backend include fallback logic for non-physical model outputs so users still get useful estimates.

## Deployment Notes

- Configure `window.API_BASE_URL` in `web/static/config.js` for separate frontend/backend deployment.
- If using a tunnel URL, set backend URL from the UI via **Set Backend URL**.

## License

No license file is currently included. Add one if this project will be shared publicly.