# from src.data_loader import load_data
# from src.preprocessing import split_data
# from src.models import get_models
# from src.evaluation import evaluate_model

# def main():
#     df = load_data("data/train.csv")

#     X_train, X_test, y_train, y_test = split_data(df)

#     models = get_models()

#     results = {}

#     for name, model in models.items():
#         rmse, mae, r2 = evaluate_model(model, X_train, X_test, y_train, y_test)
#         results[name] = {"RMSE": rmse, "MAE": mae, "R2": r2}
#         print(f"{name} -> RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")

# if __name__ == "__main__":
#     main()

from src.data_loader import load_data
from src.preprocessing import split_data
from src.models import get_models
from src.evaluation import evaluate_model, cross_validate_model
from src.visualization import plot_predictions, plot_residuals, plot_feature_importance


def main():
    df = load_data("data/train.csv")

    X_train, X_test, y_train, y_test = split_data(df)

    models = get_models()

    for name, model in models.items():
        print(f"\n🔹 Running Model: {name}")

        rmse, mae, r2, preds = evaluate_model(
            model, X_train, X_test, y_train, y_test
        )

        cv_rmse = cross_validate_model(model, X_train, y_train)

        print(f"{name} -> RMSE: {rmse:.3f}, MAE: {mae:.3f}, R2: {r2:.3f}")
        print(f"{name} -> CV RMSE: {cv_rmse:.3f}")

        # 🔴 Only for best model (XGBoost)
        if name == "XGBoost":
            plot_predictions(y_test, preds)
            plot_residuals(y_test, preds)
            plot_feature_importance(model)


if __name__ == "__main__":
    main()