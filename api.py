"""
api.py - FastAPI inference service for crop yield prediction.
Run: uvicorn api:app --reload --port 8000
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import json
import numpy as np
import pandas as pd

app = FastAPI(title="Crop Yield Prediction API")

MODEL_PATH = "models/xgb_yield_model.json"
FEATURES_PATH = "models/feature_columns.json"

# Load model and feature columns at startup
try:
    booster = xgb.Booster()
    booster.load_model(MODEL_PATH)
    with open(FEATURES_PATH) as f:
        FEATURE_COLS = json.load(f)
except Exception as e:
    booster = None
    FEATURE_COLS = None
    print("Warning: model not loaded. Please train model and place files in models/ -", e)

class InputData(BaseModel):
    # flexible: accepts arbitrary fields; we convert to dataframe using FEATURE_COLS
    data: dict

@app.post("/predict")
def predict(payload: InputData):
    if booster is None or FEATURE_COLS is None:
        raise HTTPException(status_code=500, detail="Model not available. Train model first.")

    row = payload.data
    # ensure all feature cols present
    row_arr = []
    for c in FEATURE_COLS:
        row_arr.append(float(row.get(c, 0.0)))
    X = xgb.DMatrix(pd.DataFrame([row_arr], columns=FEATURE_COLS))
    pred = booster.predict(X)[0]
    return {"predicted_yield_t_ha": float(pred)}
