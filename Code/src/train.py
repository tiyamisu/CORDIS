import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from typing import Dict, Any, Tuple

def get_models_and_params() -> Dict[str, Tuple[Any, Dict[str, Any]]]:
    """
    Returns a dictionary of models and their hyperparameter grids for tuning.
    
    Returns:
        dict: Keys are model names, values are tuples of (estimator, parameter_grid).
    """
    models = {
        'LogisticRegression': (
            LogisticRegression(max_iter=1000, random_state=42),
            {
                'C': [0.01, 0.1, 1.0, 10.0],
                'penalty': ['l2']
            }
        ),
        'DecisionTree': (
            DecisionTreeClassifier(random_state=42),
            {
                'criterion': ['gini', 'entropy'],
                'max_depth': [None, 3, 5, 8, 10],
                'min_samples_split': [2, 5, 10]
            }
        ),
        'KNN': (
            KNeighborsClassifier(),
            {
                'n_neighbors': [3, 5, 7, 9, 11],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            }
        ),
        'RandomForest': (
            RandomForestClassifier(random_state=42),
            {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 5, 10],
                'min_samples_split': [2, 5]
            }
        )
    }
    return models

def train_and_tune(X_train: pd.DataFrame, y_train: pd.Series, cv_folds: int = 5) -> Dict[str, Any]:
    """
    Train and hyperparameter tune multiple models using StratifiedKFold Cross-Validation.
    
    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training labels.
        cv_folds (int): Number of folds for cross validation.
        
    Returns:
        dict: Trained and optimized model objects.
    """
    models_spec = get_models_and_params()
    trained_models = {}
    
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    for name, (model, param_grid) in models_spec.items():
        print(f"Tuning hyper-parameters for {name}...")
        
        # Optimize using F1-score as clinical targets benefit from balancing Precision and Recall
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for {name}: {grid_search.best_params_}")
        print(f"Best CV F1-score: {grid_search.best_score_:.4f}\n")
        
        trained_models[name] = grid_search.best_estimator_
        
    return trained_models
