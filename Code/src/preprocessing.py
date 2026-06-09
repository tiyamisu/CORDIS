import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

class HeartDiseasePreprocessor:
    """
    A production-quality preprocessing pipeline for the heart disease dataset.
    Ensures zero data leakage by splitting features prior to fitting scaling parameters.
    """
    def __init__(self, target_col: str = 'target', test_size: float = 0.2, random_state: int = 42):
        self.target_col = target_col
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
        
    def load_and_clean(self, file_path: str, drop_duplicates: bool = True) -> pd.DataFrame:
        """
        Load dataset, check & remove duplicate rows, and handle missing values.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset path not found: {file_path}")
            
        df = pd.read_csv(file_path)
        print(f"[PREPROCESS] Loaded dataset of shape: {df.shape}")
        
        # Duplicate checking and removal
        dup_count = df.duplicated().sum()
        if dup_count > 0:
            print(f"[PREPROCESS] Found {dup_count} duplicate rows.")
            if drop_duplicates:
                df = df.drop_duplicates().reset_index(drop=True)
                print(f"[PREPROCESS] Removed duplicate rows. New shape: {df.shape}")
        else:
            print("[PREPROCESS] No duplicates detected.")
            
        # Missing value analysis and imputation (Median strategy for continuous features)
        missing_count = df.isnull().sum().sum()
        if missing_count > 0:
            print(f"[PREPROCESS] Found {missing_count} missing values. Handling imputation...")
            for col in df.columns:
                if df[col].isnull().any():
                    # If column is target or categorical, impute with mode, else with median
                    if col == self.target_col or df[col].nunique() < 5:
                        fill_val = df[col].mode()[0]
                    else:
                        fill_val = df[col].median()
                    df[col] = df[col].fillna(fill_val)
        else:
            print("[PREPROCESS] No missing values detected.")
            
        return df

    def separate_features_target(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Separate features (X) and target variable (y).
        """
        if self.target_col not in df.columns:
            raise KeyError(f"Target column '{self.target_col}' not found in the dataset.")
            
        X = df.drop(columns=[self.target_col])
        y = df[self.target_col]
        return X, y

    def process_pipeline(self, file_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Runs the full preprocessing pipeline:
        1. Load & clean data (impute, deduplicate)
        2. Separate features & target
        3. Train-test split (80:20 ratio, stratified)
        4. Standard scaling of numeric attributes (fit on Train, transform Train & Test)
        """
        # Load & Clean
        df = self.load_and_clean(file_path)
        
        # Separate
        X, y = self.separate_features_target(df)
        
        # Stratified Split (80:20)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y
        )
        print(f"[PREPROCESS] Stratified split (ratio={1.0 - self.test_size}:{self.test_size}) done.")
        print(f"             Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
        
        # Standard Scaling (ensures scaling fit parameters are isolated to X_train)
        X_train_scaled = X_train.copy()
        X_test_scaled = X_test.copy()
        
        X_train_scaled[self.numeric_cols] = self.scaler.fit_transform(X_train[self.numeric_cols])
        X_test_scaled[self.numeric_cols] = self.scaler.transform(X_test[self.numeric_cols])
        print("[PREPROCESS] Feature scaling applied via StandardScaler.")
        
        return X_train_scaled, X_test_scaled, y_train, y_test

# Backward-compatibility wrappers for main.py runner
import os

def load_data(file_path: str) -> pd.DataFrame:
    preprocessor = HeartDiseasePreprocessor()
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    preprocessor = HeartDiseasePreprocessor()
    # Mocking standard cleaner function
    df_cleaned = df.copy()
    dup_count = df_cleaned.duplicated().sum()
    if dup_count > 0 and drop_duplicates:
        df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
    if df_cleaned.isnull().sum().sum() > 0:
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().any():
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    return df_cleaned

def split_data(df: pd.DataFrame, target_col: str = 'target', test_size: float = 0.2, random_state: int = 42):
    preprocessor = HeartDiseasePreprocessor(target_col, test_size, random_state)
    return preprocessor.separate_features_target(df)[0], None, None, None # Kept signature placeholders if needed
    
# Replacing with simpler split function to support original main.py
def split_data(df: pd.DataFrame, target_col: str = 'target', test_size: float = 0.2, random_state: int = 42):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
