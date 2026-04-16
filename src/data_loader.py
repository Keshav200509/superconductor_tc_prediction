from __future__ import annotations
import pandas as pd

def load_data(path: str):
    """Loads a CSV file into a DataFrame."""
    return pd.read_csv(path)

if __name__ == "__main__":
    # This will now work because the function is defined above
    df = load_data("data/train.csv")
    print(df.head())




