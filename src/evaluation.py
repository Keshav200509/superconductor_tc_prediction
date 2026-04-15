
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# import numpy as np

# def evaluate_model(model, X_train, X_test, y_train, y_test):
#     model.fit(X_train, y_train)
#     preds = model.predict(X_test)

#     rmse = np.sqrt(mean_squared_error(y_test, preds))
#     mae = mean_absolute_error(y_test, preds)
#     r2 = r2_score(y_test, preds)

#     return rmse, mae, r2


# from sklearn.model_selection import cross_val_score
# import numpy as np

# scores = cross_val_score(model, X_train, y_train,
#                          scoring='neg_root_mean_squared_error',
#                          cv=5)

# rmse_cv = np.mean(-scores)

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import numpy as np

def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    return rmse, mae, r2, preds


def cross_validate_model(model, X_train, y_train, cv: int = 5):

    return rmse, mae, r2, preds


# 🔴 NEW: Cross-validation function
def cross_validate_model(model, X_train, y_train):
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring="neg_mean_squared_error",
    )
    return np.sqrt(-scores.mean())


def segment_error(y_test, preds):
    import numpy as np

    low = y_test < 20
    mid = (y_test >= 20) & (y_test <= 80)
    high = y_test > 80

    def rmse(mask):
        return np.sqrt(np.mean((y_test[mask] - preds[mask]) ** 2))

    return {
        "Low Tc RMSE": rmse(low),
        "Mid Tc RMSE": rmse(mid),
        "High Tc RMSE": rmse(high),
    }


# Stability check (multiple seeds)
def stability_test(model, X_train, y_train, X_test, y_test):
    import numpy as np

    rmses = []

    for seed in [0, 42, 100]:
        if "random_state" in model.get_params():
            model.set_params(random_state=seed)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, preds))
        rmses.append(rmse)

    return rmses
        scoring='neg_root_mean_squared_error',
        cv=5
    )
    return np.mean(-scores)

