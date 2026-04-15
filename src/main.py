from __future__ import annotations

from src.data_loader import load_data
from src.evaluation import (
    cross_validate_model,
    evaluate_model,
    segment_error,
    stability_test,
)
from src.models import get_models
from src.preprocessing import split_data
from src.visualization import plot_error_by_range


def main():
    df = load_data("data/train.csv")

    # Experiment 1: Normal target
    X_train, X_test, y_train, y_test = split_data(df, log_transform=False)

    models = get_models()

    for name, model in models.items():
        print(f"\n🔹 Running Model: {name}")

        rmse, mae, r2, preds = evaluate_model(
            model, X_train, X_test, y_train, y_test
        )

        cv_rmse = cross_validate_model(model, X_train, y_train)

        print(f"{name} -> RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")
        print(f"{name} -> CV RMSE: {cv_rmse:.3f}")

        # Segmented Error
        seg = segment_error(y_test, preds)
        print("Segmented Error:", seg)

        # Stability Test
        stability = stability_test(model, X_train, y_train, X_test, y_test)
        print("Stability RMSEs:", stability)

        # Extra plot only for tuned models
        if "Tuned" in name:
            plot_error_by_range(y_test, preds)


if __name__ == "__main__":
    main()
