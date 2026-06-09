import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from typing import Dict, Any

def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Evaluate a model on the test dataset.
    
    Args:
        model (Any): Classifier model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        
    Returns:
        dict: Metric names and their computed values.
    """
    y_pred = model.predict(X_test)
    
    # Check if model has predict_proba
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_prob = model.decision_function(X_test)
        # Normalize decision function values to range [0, 1] using standard logistic sigmoid
        y_prob = 1 / (1 + np.exp(-y_prob))
    else:
        y_prob = y_pred.astype(float)
        
    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, zero_division=0),
        'Recall': recall_score(y_test, y_pred, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred, zero_division=0),
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    }
    
    return metrics

def generate_report(trained_models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """
    Generate an aggregated evaluation report comparing all trained models.
    
    Args:
        trained_models (dict): Trained model name to estimator mapping.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): Test labels.
        
    Returns:
        pd.DataFrame: Metric comparison table.
    """
    report_data = []
    
    for name, model in trained_models.items():
        metrics = evaluate_model(model, X_test, y_test)
        metrics['Model'] = name
        report_data.append(metrics)
        
        # Print classification report
        y_pred = model.predict(X_test)
        print(f"--- Classification Report for {name} ---")
        print(classification_report(y_test, y_pred, zero_division=0))
        print("="*50)
        
    report_df = pd.DataFrame(report_data)
    # Reorder columns to make Model the first column
    cols = ['Model'] + [col for col in report_df.columns if col != 'Model']
    report_df = report_df[cols]
    
    return report_df
