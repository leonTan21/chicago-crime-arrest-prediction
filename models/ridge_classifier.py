"""
Ridge Classifier — Chicago Crime Arrest Prediction
Hyperparameters sourced from GridSearchCV in chicago_crime_prediction.ipynb.
Run from project root: python models/ridge_classifier.py

Note: RidgeClassifier does not support predict_proba(). ROC-AUC is computed
from decision_function() scores, which are real-valued and monotone with
respect to class probability — valid for ranking-based metrics like AUC.
"""

import os
import sys
import joblib
import numpy as np
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.preprocessing import get_preprocessed_data

MODELS_DIR = os.path.dirname(os.path.abspath(__file__))

BEST_PARAMS = {
    "alpha":        10.0,
    "class_weight": "balanced",
}


def print_metrics(label: str, y_true, y_pred, y_score) -> dict:
    metrics = {
        "Accuracy":  accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall":    recall_score(y_true, y_pred, zero_division=0),
        "F1":        f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC":   roc_auc_score(y_true, y_score),
    }
    print(f"\n{label}:")
    for k, v in metrics.items():
        print(f"  {k:<12}: {v:.4f}")
    return metrics


def main():
    X_train, X_test, y_train, y_test, features, _ = get_preprocessed_data()

    print("\nTraining Ridge Classifier...")
    model = RidgeClassifier(**BEST_PARAMS)
    model.fit(X_train, y_train)

    train_score = model.decision_function(X_train)
    test_score  = model.decision_function(X_test)

    print_metrics("Train Set", y_train, model.predict(X_train), train_score)
    print_metrics("Test Set",  y_test,  model.predict(X_test),  test_score)

    out = os.path.join(MODELS_DIR, "ridge_classifier.pkl")
    joblib.dump({"model": model, "features": features}, out)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
