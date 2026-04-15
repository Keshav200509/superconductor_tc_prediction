import matplotlib.pyplot as plt
from xgboost import plot_importance
import os

# Ensure output directory exists
os.makedirs("outputs/plots", exist_ok=True)


def plot_predictions(y_test, preds):
    plt.figure()
    plt.scatter(y_test, preds, alpha=0.5)
    plt.xlabel("Actual Tc")
    plt.ylabel("Predicted Tc")
    plt.title("Actual vs Predicted Tc")
    plt.savefig("outputs/plots/pred_vs_actual.png")
    plt.close()


def plot_residuals(y_test, preds):
    residuals = y_test - preds

    plt.figure()
    plt.hist(residuals, bins=50)
    plt.title("Residual Distribution")
    plt.savefig("outputs/plots/residuals.png")
    plt.close()


def plot_feature_importance(model):
    plt.figure()
    plot_importance(model, max_num_features=10)
    plt.title("Top 10 Important Features")
    plt.savefig("outputs/plots/feature_importance.png")
    plt.close()