from __future__ import annotations
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Trains the model and returns performance metrics."""
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return rmse, mae, r2, preds

def cross_validate_model(model, X, y, cv: int = 5):
    """Performs cross-validation and returns the average RMSE."""
    scores = cross_val_score(
        model, X, y, 
        cv=cv, 
        scoring='neg_root_mean_squared_error'
    )
    return -scores.mean()

def segment_error(y_test, preds):
    """Calculates RMSE for different temperature ranges."""
    y_test = np.array(y_test)
    preds = np.array(preds)
    
    low = y_test < 20
    mid = (y_test >= 20) & (y_test <= 80)
    high = y_test > 80

    def get_rmse(mask):
        if not np.any(mask): return 0.0
        return np.sqrt(np.mean((y_test[mask] - preds[mask]) ** 2))

    return {
        "Low Tc RMSE": get_rmse(low),
        "Mid Tc RMSE": get_rmse(mid),
        "High Tc RMSE": get_rmse(high),
    }

def stability_test(model, X_train, y_train, X_test, y_test):
    """Checks model stability across different random seeds."""
    rmses = []
    for seed in [0, 42, 100]:
        # Check if the model supports random_state (like RF or XGBoost)
        if hasattr(model, "random_state"):
            model.random_state = seed
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        rmses.append(rmse)

    return rmses
