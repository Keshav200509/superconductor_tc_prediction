
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def split_data(df, log_transform: bool = False):
    X = df.drop("critical_temp", axis=1)
    y = df["critical_temp"]

    if log_transform:
        y = np.log1p(y)  # log(1 + Tc)

    return train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.model_selection import train_test_split

def split_data(df):
    X = df.drop("critical_temp", axis=1)
    y = df["critical_temp"]

    return train_test_split(X, y, test_size=0.2, random_state=42)
