"""
evaluate.py
===========
Model evaluation utilities for the CORDIS Heart Disease Prediction project.

Provides:
    evaluate_model()    — Per-model metric computation (Accuracy, Precision,
                          Recall, F1, ROC-AUC).
    generate_report()   — Aggregated comparison table across all models.
    print_conf_matrix() — Formatted confusion matrix display.

All metrics are computed on the independent held-out test set.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
)
from typing import Any, Dict


# ---------------------------------------------------------------------------
# Per-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Compute classification metrics for a single fitted model.

    Metrics returned:
        Accuracy  — overall correct prediction rate
        Precision — positive predictive value (TP / (TP + FP))
        Recall    — sensitivity / true positive rate (TP / (TP + FN))
        F1-Score  — harmonic mean of Precision and Recall
        ROC-AUC   — area under the receiver operating characteristic curve

    Args:
        model  : Fitted sklearn-compatible classifier.
        X_test : Transformed test features.
        y_test : True binary labels.

    Returns:
        dict mapping metric name → float value.
    """
    y_pred = model.predict(X_test)

    # Obtain probability scores for ROC-AUC
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X_test)
        # Sigmoid normalisation → [0, 1]
        y_prob = 1.0 / (1.0 + np.exp(-scores))
    else:
        y_prob = y_pred.astype(float)

    return {
        'Accuracy':  accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall':    recall_score(y_test, y_pred, zero_division=0),
        'F1-Score':  f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC':   roc_auc_score(y_test, y_prob),
    }


# ---------------------------------------------------------------------------
# Confusion matrix display
# ---------------------------------------------------------------------------

def print_conf_matrix(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = '',
) -> None:
    """
    Print a formatted confusion matrix with TP/FP/FN/TN labels.

    Args:
        model     : Fitted classifier.
        X_test    : Transformed test features.
        y_test    : True binary labels.
        model_name: Display name for the model heading.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)

    tn, fp, fn, tp = cm.ravel()

    print(f"\n  {'Confusion Matrix':^40}")
    print(f"  {'─' * 40}")
    print(f"  {'':20} {'Predicted NO':>10} {'Predicted YES':>12}")
    print(f"  {'Actual NO':20} {tn:>10} {fp:>12}")
    print(f"  {'Actual YES':20} {fn:>10} {tp:>12}")
    print(f"  {'─' * 40}")
    print(f"  Sensitivity (Recall)  : {tp / (tp + fn):.4f}  [TP / (TP + FN)]")
    print(f"  Specificity           : {tn / (tn + fp):.4f}  [TN / (TN + FP)]")
    print(f"  Positive Pred. Value  : {tp / (tp + fp):.4f}  [TP / (TP + FP)]")
    print(f"  Negative Pred. Value  : {tn / (tn + fn):.4f}  [TN / (TN + FN)]")


# ---------------------------------------------------------------------------
# Aggregated report
# ---------------------------------------------------------------------------

def generate_report(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """
    Generate a formatted performance comparison table for all models.

    For each model:
        1. Computes classification metrics.
        2. Prints the sklearn classification_report.
        3. Prints the formatted confusion matrix.

    Args:
        trained_models: Name → fitted estimator mapping.
        X_test        : Transformed test features.
        y_test        : True binary labels.

    Returns:
        pd.DataFrame with one row per model and columns:
        [Model, Accuracy, Precision, Recall, F1-Score, ROC-AUC]
    """
    records = []

    for name, model in trained_models.items():
        print(f"\n{'=' * 55}")
        print(f"  Model: {name}")
        print(f"{'=' * 55}")

        # --- sklearn classification report ---
        y_pred = model.predict(X_test)
        print(classification_report(
            y_test, y_pred,
            target_names=['No Disease (0)', 'Disease (1)'],
            zero_division=0,
        ))

        # --- confusion matrix ---
        print_conf_matrix(model, X_test, y_test, name)

        # --- aggregate metrics ---
        metrics = evaluate_model(model, X_test, y_test)
        metrics['Model'] = name
        records.append(metrics)

    report_df = pd.DataFrame(records)
    # Move 'Model' column to front
    cols = ['Model'] + [c for c in report_df.columns if c != 'Model']
    return report_df[cols]
