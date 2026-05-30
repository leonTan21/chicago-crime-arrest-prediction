"""
Logistic Regression — Chicago Crime Arrest Prediction
Hyperparameters sourced from GridSearchCV in chicago_crime_prediction.ipynb.
Run from project root: python models/logistic_regression.py
"""

import os
import sys
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.preprocessing import get_preprocessed_data

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

# Best hyperparameters from GridSearchCV (cv=5) in the notebook
BEST_PARAMS = {
    "C":          0.1,
    "solver":     "saga",
    "max_iter":   1000,
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs":     -1,
}


def print_metrics(label: str, y_true, y_pred, y_proba) -> dict:
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_true, y_proba),
    }
    print(f"\n{label}:")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    return metrics


def main():
    X_train, X_test, y_train, y_test, features, _ = get_preprocessed_data()

    print("\nTraining Logistic Regression...")
    model = LogisticRegression(**BEST_PARAMS)
    model.fit(X_train, y_train)

    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba  = model.predict_proba(X_test)[:, 1]

    print_metrics("Train Set", y_train, model.predict(X_train), train_proba)
    print_metrics("Test Set",  y_test,  model.predict(X_test),  test_proba)

    out = os.path.join(MODELS_DIR, "logistic_regression.pkl")
    joblib.dump({"model": model, "features": features}, out)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
