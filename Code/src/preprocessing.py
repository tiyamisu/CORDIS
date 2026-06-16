"""
preprocessing.py
================
Data loading, cleaning, column normalisation, outlier winsorization,
missing-value imputation, and stratified train/test splitting for the
CORDIS Heart Disease Prediction project.

The CSV header uses verbose column names; this module maps them to
short, consistent aliases used throughout the rest of the pipeline.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Tuple

# ---------------------------------------------------------------------------
# Column rename map: CSV long name → short alias
# ---------------------------------------------------------------------------
COLUMN_RENAME_MAP: dict = {
    'Age':                            'age',
    'Sex':                            'sex',
    'Chest Pain Type':                'cp',
    'Resting Blood Pressure':         'trestbps',
    'Cholestrol':                     'chol',
    'Fasting Blood Sugar':            'fbs',
    'Rest ECG':                       'restecg',
    'Max Heart Rate':                 'thalach',
    'Exercise Angina':                'exang',
    'Old Peak':                       'oldpeak',
    'Slope of Peak':                  'slope',
    'Vessles by Fluoroscopy':         'ca',
    'Thalassemia':                    'thal',
    'Target (Presence of Heart Disease)': 'target',
}

# Numerical features subject to scaling
NUMERIC_FEATURES: list = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']

# Categorical features (used for reference in outlier handling)
ORDINAL_FEATURES: list = ['cp', 'restecg', 'slope', 'ca', 'thal']
BINARY_FEATURES: list  = ['sex', 'fbs', 'exang']


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the CSV dataset and normalise column names.

    Args:
        file_path: Absolute or relative path to the CSV file.

    Returns:
        pd.DataFrame with short, snake_case column aliases.

    Raises:
        FileNotFoundError: If the file does not exist at the given path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"[PREPROCESS] Dataset not found at: {file_path}\n"
            "Please verify the path and re-run."
        )

    df = pd.read_csv(file_path)
    print(f"[PREPROCESS] Loaded raw dataset: {df.shape[0]} rows × {df.shape[1]} columns")

    # Rename long CSV headers to short aliases wherever they exist
    rename = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)

    # Confirm expected target column exists
    if 'target' not in df.columns:
        raise KeyError(
            "[PREPROCESS] Target column not found after renaming. "
            f"Available columns: {list(df.columns)}"
        )

    return df


def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Remove duplicates, handle missing values, and winsorize outliers.

    Strategy:
        - Duplicates    : Drop entirely (dataset has heavy duplication).
        - Missing values: Median for continuous; mode for categorical/binary.
        - Outliers      : IQR-based winsorization on continuous features
                          (caps extreme values rather than dropping rows).

    Args:
        df             : Raw DataFrame (already column-renamed).
        drop_duplicates: If True, remove duplicate rows.

    Returns:
        Cleaned pd.DataFrame.
    """
    df_clean = df.copy()

    # ---- 1. Duplicate removal ----
    n_dups = df_clean.duplicated().sum()
    print(f"[PREPROCESS] Duplicate rows found : {n_dups}")
    if n_dups > 0 and drop_duplicates:
        df_clean = df_clean.drop_duplicates().reset_index(drop=True)
        print(f"[PREPROCESS] After deduplication  : {df_clean.shape[0]} rows remain")
    else:
        print("[PREPROCESS] Skipping deduplication (none found or flag=False)")

    # ---- 2. Missing value imputation ----
    missing_total = df_clean.isnull().sum().sum()
    print(f"[PREPROCESS] Missing values found  : {missing_total}")
    if missing_total > 0:
        for col in df_clean.columns:
            if df_clean[col].isnull().any():
                n_miss = df_clean[col].isnull().sum()
                if col in NUMERIC_FEATURES:
                    fill = df_clean[col].median()
                    strategy = "median"
                else:
                    fill = df_clean[col].mode()[0]
                    strategy = "mode"
                df_clean[col] = df_clean[col].fillna(fill)
                print(f"  [IMPUTE] '{col}': {n_miss} values filled with {strategy} ({fill:.4g})")
    else:
        print("[PREPROCESS] No missing values — imputation skipped")

    # ---- 3. IQR Winsorization (outlier capping) ----
    print("[PREPROCESS] Applying IQR winsorization to continuous features...")
    for col in NUMERIC_FEATURES:
        if col not in df_clean.columns:
            continue
        q1  = df_clean[col].quantile(0.25)
        q3  = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        lo  = q1 - 1.5 * iqr
        hi  = q3 + 1.5 * iqr
        n_capped = ((df_clean[col] < lo) | (df_clean[col] > hi)).sum()
        if n_capped > 0:
            df_clean[col] = df_clean[col].clip(lower=lo, upper=hi)
            print(f"  [WINSOR] '{col}': {n_capped} values capped to [{lo:.2f}, {hi:.2f}]")

    print(f"[PREPROCESS] Cleaned dataset shape : {df_clean.shape}")
    return df_clean


def split_data(
    df: pd.DataFrame,
    target_col: str = 'target',
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Stratified 80/20 train–test split.

    Args:
        df          : Cleaned DataFrame.
        target_col  : Name of the target column.
        test_size   : Fraction reserved for testing (default 0.2).
        random_state: Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test
    """
    if target_col not in df.columns:
        raise KeyError(f"[SPLIT] Target column '{target_col}' not in DataFrame.")

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"[PREPROCESS] Stratified split -> Train: {len(X_train)} | Test: {len(X_test)}")
    print(f"[PREPROCESS] Class balance (train) -> {y_train.value_counts().to_dict()}")

    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Legacy compatibility shim (keeps old call sites working)
# ---------------------------------------------------------------------------
class HeartDiseasePreprocessor:
    """Backward-compat wrapper. Use module-level functions for new code."""

    def __init__(self, target_col='target', test_size=0.2, random_state=42):
        self.target_col   = target_col
        self.test_size    = test_size
        self.random_state = random_state

    def process_pipeline(self, file_path: str):
        df = load_data(file_path)
        df = clean_data(df)
        return split_data(df, self.target_col, self.test_size, self.random_state)
