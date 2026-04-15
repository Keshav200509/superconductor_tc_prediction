from __future__ import annotations

import matplotlib.pyplot as plt


def plot_error_by_range(y_test, preds):
    ranges = ["Low (<20)", "Mid (20-80)", "High (>80)"]
    masks = [
        y_test < 20,
        (y_test >= 20) & (y_test <= 80),
        y_test > 80,
    ]

    errors = []
    for mask in masks:
        rmse = ((y_test[mask] - preds[mask]) ** 2).mean() ** 0.5
        errors.append(rmse)

    plt.figure()
    plt.bar(ranges, errors)
    plt.title("RMSE by Tc Range")
    plt.savefig("outputs/plots/error_by_range.png")
    plt.close()
