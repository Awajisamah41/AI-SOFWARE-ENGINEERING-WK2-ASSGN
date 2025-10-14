"""
model_training.py
Train a crop yield prediction model from a CSV dataset.

Usage:
    python model_training.py
Outputs:
    - models/xgb_yield_model.json  (XGBoost model)
    - models/feature_columns.json  (feature column order)
"""
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import joblib

DATA_PATH = "data/crop_data.csv"
os.makedirs("models", exist_ok=True)

def load_data(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected dataset at {path}. Please place your merged dataset there.")
    df = pd.read_csv(path)
    return df

def prepare_features(df):
    # Basic cleaning
    df = df.dropna(subset=["yield_t_ha"])
    num_cols = ["precip_mm_growing","mean_temp_growing","fertilizer_kg_ha","prev_yield","planting_day_of_year"]
    for c in num_cols:
        if c in df.columns:
            df[c] = df[c].fillna(df[c].median())
        else:
            df[c] = 0.0

    # Simple soil encoding: if soil_type present, one-hot
    if "soil_type" in df.columns:
        df["soil_type"] = df["soil_type"].fillna("unknown")
        soil_dummies = pd.get_dummies(df["soil_type"], prefix="soil")
        df = pd.concat([df, soil_dummies], axis=1)

    feature_cols = [
        "precip_mm_growing","mean_temp_growing","fertilizer_kg_ha","prev_yield","planting_day_of_year"
    ]
    feature_cols += [c for c in df.columns if c.startswith("soil_")]

    X = df[feature_cols].astype(float)
    y = df["yield_t_ha"].astype(float)
    return X, y, feature_cols

def train_and_save(X_train, y_train, X_val, y_val, feature_cols):
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    params = {
        "objective":"reg:squarederror",
        "eval_metric":"rmse",
        "eta":0.05,
        "max_depth":6,
        "subsample":0.8,
        "colsample_bytree":0.8,
        "seed":42
    }
    model = xgb.train(params, dtrain, num_boost_round=1000, evals=[(dval,"val")], early_stopping_rounds=20, verbose_eval=50)
    # Save model and feature order
    model.save_model("models/xgb_yield_model.json")
    with open("models/feature_columns.json","w") as f:
        json.dump(feature_cols, f)
    return model

def evaluate(model, X_test, y_test):
    dpred = xgb.DMatrix(X_test)
    preds = model.predict(dpred)
    print("MAE:", mean_absolute_error(y_test, preds))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, preds)))
    print("R2:", r2_score(y_test, preds))

def main():
    df = load_data()
    X, y, feature_cols = prepare_features(df)
    # time-aware split: hold out last year if 'year' present
    if "year" in df.columns:
        last_year = df["year"].max()
        train_mask = df["year"] < last_year
        test_mask = df["year"] == last_year
        if test_mask.sum() >= 30:
            X_train, X_test = X[train_mask], X[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_and_save(X_train, y_train, X_test, y_test, feature_cols)
    evaluate(model, X_test, y_test)

if __name__ == '__main__':
    main()
