"""
train.py
========
Model definition, hyperparameter grids, and cross-validated GridSearch
training for the CORDIS Heart Disease Prediction project.

Models trained:
    - Logistic Regression  (L₂ regularised)
    - Decision Tree        (CART)
    - K-Nearest Neighbors
    - Random Forest        (bagging ensemble)
    - XGBoost              (gradient-boosted trees)

All models are tuned via StratifiedKFold GridSearchCV optimising F1-score,
which balances Precision and Recall — important for clinical false-negative
minimisation.
"""

import pandas as pd
from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.neighbors        import KNeighborsClassifier
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import GridSearchCV, StratifiedKFold
from xgboost                  import XGBClassifier
from typing import Any, Dict, Tuple


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

def get_models_and_params() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Return a registry of (estimator, hyperparameter_grid) pairs.

    Returns:
        dict: Keys are human-readable model names; values are
              (unfitted estimator, param_grid dict) tuples.
    """
    return {
        # ----------------------------------------------------------------
        # Logistic Regression — interpretable linear baseline
        # ----------------------------------------------------------------
        'LogisticRegression': (
            LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs'),
            {
                'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                # Note: 'penalty' default is L2 in sklearn 1.8+; no need to pass explicitly
            },
        ),

        # ----------------------------------------------------------------
        # Decision Tree — axis-aligned non-linear, fully explainable
        # ----------------------------------------------------------------
        'DecisionTree': (
            DecisionTreeClassifier(random_state=42),
            {
                'criterion':        ['gini', 'entropy'],
                'max_depth':        [None, 3, 5, 8, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
            },
        ),

        # ----------------------------------------------------------------
        # K-Nearest Neighbors — non-parametric, instance-based
        # ----------------------------------------------------------------
        'KNN': (
            KNeighborsClassifier(n_jobs=-1),
            {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights':     ['uniform', 'distance'],
                'metric':      ['euclidean', 'manhattan'],
            },
        ),

        # ----------------------------------------------------------------
        # Random Forest — bagging ensemble; robust to overfitting
        # ----------------------------------------------------------------
        'RandomForest': (
            RandomForestClassifier(random_state=42, n_jobs=-1),
            {
                'n_estimators':     [100, 200, 300],
                'max_depth':        [None, 5, 10, 15],
                'min_samples_split': [2, 5],
                'max_features':     ['sqrt', 'log2'],
            },
        ),

        # ----------------------------------------------------------------
        # XGBoost — gradient-boosted trees; state-of-art tabular benchmark
        # ----------------------------------------------------------------
        'XGBoost': (
            XGBClassifier(
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False,
                verbosity=0,
            ),
            {
                'n_estimators':  [100, 200, 300],
                'max_depth':     [3, 5, 7],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample':     [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0],
            },
        ),
    }


# ---------------------------------------------------------------------------
# Training orchestration
# ---------------------------------------------------------------------------

def train_and_tune(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv_folds: int = 5,
) -> Dict[str, Any]:
    """
    Train and hyperparameter-tune all registered models via StratifiedKFold
    GridSearchCV.

    Scoring: F1 (macro) — balances Precision and Recall, appropriate
    for clinical binary classification where false negatives (missed
    disease) carry high cost.

    Args:
        X_train  : Transformed training features.
        y_train  : Binary training labels (0 = no disease, 1 = disease).
        cv_folds : Number of stratified cross-validation folds (default 5).

    Returns:
        dict: Model name → best fitted estimator.
    """
    models_spec   = get_models_and_params()
    trained_models: Dict[str, Any] = {}

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    print(f"\n[TRAIN] Starting GridSearchCV across {len(models_spec)} models "
          f"with {cv_folds}-fold stratified CV...\n")

    for name, (model, param_grid) in models_spec.items():
        print(f"  >> Tuning: {name} ...")
        gs = GridSearchCV(
            estimator  = model,
            param_grid = param_grid,
            cv         = cv,
            scoring    = 'f1',
            n_jobs     = -1,
            verbose    = 0,
            refit      = True,
        )
        gs.fit(X_train, y_train)

        trained_models[name] = gs.best_estimator_

        print(f"     Best params : {gs.best_params_}")
        print(f"     Best CV F1  : {gs.best_score_:.4f}\n")

    print("[TRAIN] All models trained and tuned successfully.")
    return trained_models
