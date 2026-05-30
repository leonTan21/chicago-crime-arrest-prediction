"""
XGBoost Classifier — Chicago Crime Arrest Prediction
Hyperparameters sourced from GridSearchCV in chicago_crime_prediction.ipynb.
Run from project root: python models/xgboost_classifier.py

Note on class imbalance: XGBoost does not accept class_weight='balanced'.
Instead, scale_pos_weight = n_negative / n_positive is used, which achieves
the same effect as balanced weighting for binary classification.
"""

import os
import sys
import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.preprocessing import get_preprocessed_data

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

BEST_PARAMS = {
    "n_estimators":   300,
    "learning_rate":  0.05,
    "max_depth":      6,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":    "logloss",
    "random_state":   42,
    "n_jobs":         -1,
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

    # Derive scale_pos_weight from training label distribution
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    spw   = round(n_neg / n_pos, 2)
    print(f"\nscale_pos_weight = {spw} (n_neg={n_neg:,} / n_pos={n_pos:,})")

    model = XGBClassifier(**BEST_PARAMS, scale_pos_weight=spw)
    model.fit(X_train, y_train)

    train_proba = model.predict_proba(X_train)[:, 1]
    test_proba  = model.predict_proba(X_test)[:, 1]

    print_metrics("Train Set", y_train, model.predict(X_train), train_proba)
    print_metrics("Test Set",  y_test,  model.predict(X_test),  test_proba)

    out = os.path.join(MODELS_DIR, "xgboost_classifier.pkl")
    joblib.dump({"model": model, "features": features}, out)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
