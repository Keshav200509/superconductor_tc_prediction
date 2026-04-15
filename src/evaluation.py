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


# 🔴 NEW: Cross-validation function
def cross_validate_model(model, X_train, y_train):
    scores = cross_val_score(
        model,
        X_train,
        y_train,
        scoring='neg_root_mean_squared_error',
        cv=5
    )
    return np.mean(-scores)