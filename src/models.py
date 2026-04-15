from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from xgboost import XGBRegressor



# from sklearn.linear_model import LinearRegression, Ridge
# from sklearn.ensemble import RandomForestRegressor
# from xgboost import XGBRegressor

# XGBRegressor(
#     n_estimators=500,
#     learning_rate=0.05,
#     max_depth=6,
#     subsample=0.8,
#     colsample_bytree=0.8,
#     random_state=42
# )

# def get_models():
#     models = {
#         "Linear Regression": LinearRegression(),
#         "Ridge": Ridge(),
#         "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
#         "XGBoost": XGBRegressor(n_estimators=200, learning_rate=0.1)
#     }
#     return models

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
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
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),

        # 🔴 UPDATED XGBoost (tuned)
        "XGBoost": XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        ),
    }
    return models
            random_state=42
        )
    }
    return models
