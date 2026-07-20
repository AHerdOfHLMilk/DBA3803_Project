# DBA3803 — Predicting Hospital Readmissions

Group **predictive-analytics** project for NUS **DBA3803 (Predictive Analytics in Business)**. We build and compare machine-learning models that predict whether a patient will be **readmitted to hospital**, using the `hospital_readmissions` dataset — including a synthetically-augmented version to tackle class imbalance.

## The problem
Hospital readmissions are costly and often preventable. We frame readmission as a **binary classification** task and compare models on both predictive performance and interpretability — so a hospital could flag high-risk patients for follow-up.

## What we did
- Explored and cleaned the readmissions dataset and engineered features.
- Addressed class imbalance with a **synthetically-augmented dataset**.
- Trained, tuned (cross-validation), and compared a range of models.
- Analysed **feature importance** to understand what drives readmission risk.

## Models compared
- Logistic Regression — baseline
- Support Vector Machine (SVM)
- Random Forest
- XGBoost (gradient boosting)
- Neural networks — linear and polynomial-feature variants

## Notebooks
| File | Contents |
|------|----------|
| `finalfinalpls.ipynb` | Consolidated final analysis |
| `RF_XGB.ipynb` | Random Forest & XGBoost |
| `rf_original.ipynb` | Earlier Random Forest baseline |
| `boosting.ipynb` | Boosting experiments |
| `crossval.ipynb` | Cross-validation |
| `dba3803proj.py` | Supporting script |

*(HTML exports of the notebooks are included for quick viewing without running them.)*

## Tech
Python · pandas · scikit-learn · XGBoost · NumPy / PyTorch (neural nets) · Jupyter
