# Superconducting Critical Temperature (Tc) Prediction using Machine Learning

##  Overview

This project focuses on predicting the **superconducting critical temperature (Tc)** of materials using machine learning techniques. The study is based on composition-derived features and explores how different models capture the complex relationships between material properties and superconductivity.

The project not only reproduces a known research approach but also extends it through **experimental analysis, model comparison, and interpretability techniques**.

---

##  Objective

* Predict the critical temperature (Tc) of known superconductors
* Compare performance of linear and ensemble models
* Reproduce research-level results using a structured ML pipeline
* Perform additional analysis such as:

  * Cross-validation
  * Model ablation
  * Error segmentation
  * Stability testing

---

##  Dataset

* **Source:** UCI Machine Learning Repository
* **Name:** Superconductivity Dataset
* **Samples:** 21,263
* **Features:** 81 numerical features
* **Target:** Critical Temperature (Tc in Kelvin)

The dataset consists of features derived from elemental properties such as atomic mass, valence, electron affinity, and thermal conductivity.

---

##  Methodology

### 1. Data Preprocessing

* Separation of features and target variable
* Train-test split (80:20)
* Optional target transformation (log scaling)

### 2. Models Implemented

* Linear Regression
* Ridge Regression
* Random Forest (Base & Tuned)
* XGBoost (Base & Tuned)

### 3. Evaluation Metrics

* Root Mean Squared Error (RMSE)
* Mean Absolute Error (MAE)
* R² Score
* 5-Fold Cross-Validation

---

##  Experimental Enhancements

This project goes beyond standard implementation by including:

* **Model Ablation Study**

  * Comparison between base and tuned models

* **Cross-Validation**

  * Ensures robustness and generalization

* **Error Segmentation**

  * Performance analysis across Tc ranges:

    * Low (< 20 K)
    * Medium (20–80 K)
    * High (> 80 K)

* **Stability Testing**

  * Model consistency checked across different random seeds

* **Visualization**

  * Predicted vs Actual plots
  * Residual analysis
  * Feature importance
  * Error distribution

---

##  Results Summary

| Model                 | RMSE     | CV RMSE  | R²    |
| --------------------- | -------- | -------- | ----- |
| Linear Regression     | 17.38    | 17.70    | 0.738 |
| Ridge                 | 17.40    | 17.70    | 0.737 |
| Random Forest (Tuned) | **8.95** | 9.75     | 0.930 |
| XGBoost (Tuned)       | 9.00     | **9.72** | 0.930 |

###  Key Findings

* Linear models underperform due to non-linear relationships
* Ensemble models significantly improve prediction accuracy
* Random Forest slightly outperforms XGBoost in this implementation
* Results align closely with published research benchmarks (~9–10 RMSE)

---

##  Project Structure

```
superconductor_tc_prediction/
│
├── data/                  # Dataset
├── notebooks/             # EDA notebook
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   ├── visualization.py
│
├── outputs/
│   ├── plots/             # Generated graphs
│   └── results/           # Model results
│
├── main.py                # Main execution script
└── README.md
```

---

##  Outputs

The project generates:

* Model performance metrics
* Prediction vs Actual plots
* Residual distribution
* Feature importance visualization
* Error analysis across Tc ranges

---

##  Reference

Hamidieh, K. (2018)
*A data-driven statistical model for predicting the critical temperature of a superconductor*
Computational Materials Science
DOI: 10.1016/j.commatsci.2018.07.052

---

##  Key Contributions

* Reproduced research-level results using a structured ML pipeline
* Conducted additional experiments for robustness and interpretability
* Analyzed model performance across different temperature regimes
* Provided insights into feature influence on superconducting properties

---

##  Limitations

* Model predicts Tc only for known superconductors
* No structural (crystal-level) information used
* Higher error observed for high Tc materials

---

##  Future Work

* Incorporate structural and graph-based features
* Explore deep learning models (GNNs)
* Extend to superconductivity classification tasks

---

## Author

Keshav Yadav

---

##  Note

All experiments, model implementations, and analysis in this project were independently developed and executed, with reference to existing literature for methodological guidance.
