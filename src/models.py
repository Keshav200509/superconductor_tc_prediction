from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor


def get_models():
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(),
        "Random Forest (Base)": RandomForestRegressor(
            n_estimators=100, random_state=42
        ),
        "Random Forest (Tuned)": RandomForestRegressor(
            n_estimators=300,
            max_depth=20,
            random_state=42,
        ),
        "XGBoost (Base)": XGBRegressor(
            n_estimators=200,
            random_state=42,
        ),
        "XGBoost (Tuned)": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
    }
    return models
