from __future__ import annotations
import numpy as np
from sklearn.model_selection import train_test_split

def split_data(df, log_transform: bool = False):
    """Splits data into training and testing sets with optional log transformation."""
    X = df.drop("critical_temp", axis=1)
    y = df["critical_temp"]

    if log_transform:
        # We use log1p to handle 0 values safely: log(1 + x)
        y = np.log1p(y)

    return train_test_split(X, y, test_size=0.2, random_state=42)


