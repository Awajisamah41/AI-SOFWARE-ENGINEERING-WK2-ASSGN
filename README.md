Predictive Crop Yield Forecasting for Smallholder Farms
=====================================

This project implements an end-to-end pipeline to predict crop yield (tons/ha)
from environmental and farm-management features. It uses RandomForest and XGBoost,
includes SHAP explainability, and provides a FastAPI inference service.

Files included:
- model_training.py : Data loading, cleaning, feature engineering, training, evaluation, SHAP.
- api.py            : FastAPI app exposing /predict endpoint (loads trained model file).
- Dockerfile        : Simple Dockerfile to containerize the API.
- requirements.txt  : Python dependencies.
- example_input.json: Example JSON payload for the API.
- README.md         : This file.
- NOTE.txt          : Notes on using your own data and where to merge FAOSTAT / climate data.
- license.txt       : MIT License.

How to use (local):
1. Place your merged dataset as `data/crop_data.csv` relative to the project root.
   Expected minimal columns: country, year, crop, yield_t_ha, precip_mm_growing,
   mean_temp_growing, fertilizer_kg_ha, soil_type, planting_day_of_year, prev_yield.
2. Run training: python model_training.py
   This will train an XGBoost model and save 'models/xgb_yield_model.json' and 'models/feature_columns.json'.
3. Run the API: uvicorn api:app --reload --port 8000
4. POST JSON to http://localhost:8000/predict with the example_input.json schema.

Notes:
- The training script uses a time-based holdout (last year) when available.
- Adapt column names in model_training.py to match your dataset if needed.
- For production use, consider LightGBM, robust hyperparameter tuning, secure deployment and monitoring.
