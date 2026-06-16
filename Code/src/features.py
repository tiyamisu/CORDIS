"""
features.py
===========
Feature engineering and transformation pipeline for the CORDIS
Heart Disease Prediction project.

Responsibilities:
    - Derive clinically motivated interaction / ratio features.
    - One-hot encode ordinal/nominal categorical features.
    - Standard-scale continuous numerical features.
    - Preserve binary pass-through features unchanged.
    - Expose `fit_transform` / `transform` API so the same object can
      be serialised with joblib for production inference.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import List

# ---------------------------------------------------------------------------
# Column definitions (short alias names — post-rename)
# ---------------------------------------------------------------------------

# Continuous features to scale
NUM_COLS: List[str] = [
    'age', 'trestbps', 'chol', 'thalach', 'oldpeak',
    # Engineered features (added in _engineer_features)
    'bp_chol_ratio', 'thalach_age_ratio', 'oldpeak_slope_interaction',
    'hr_reserve',
]

# Ordinal/nominal features to one-hot encode
CAT_COLS: List[str] = ['cp', 'restecg', 'slope', 'thal']

# Binary / already-encoded features — pass through unchanged
PASS_COLS: List[str] = ['sex', 'fbs', 'exang', 'ca', 'age_risk_flag']


class FeaturePipeline:
    """
    End-to-end feature transformation pipeline.

    Usage::

        pipeline = FeaturePipeline()
        X_train_t = pipeline.fit_transform(X_train)   # fit + transform
        X_test_t  = pipeline.transform(X_test)         # transform only

    Attributes:
        feature_names (List[str]): Ordered list of output feature names,
                                   populated after fit_transform.
    """

    def __init__(self) -> None:
        self.scaler:  StandardScaler  = StandardScaler()
        self.encoder: OneHotEncoder   = OneHotEncoder(
            handle_unknown='ignore', sparse_output=False
        )
        self.fitted: bool = False
        self.feature_names: List[str] = []

    # ------------------------------------------------------------------
    # Internal: feature engineering
    # ------------------------------------------------------------------

    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific interaction terms and clinical risk indices.

        Derived features:
            bp_chol_ratio          : Resting BP / (cholesterol + ε)
                                     — cardiovascular risk index
            thalach_age_ratio      : Max heart rate / (age + ε)
                                     — age-normalised cardiac output
            oldpeak_slope_interaction : ST depression × slope of peak
                                     — combined ST segment risk signal
            age_risk_flag          : Binary flag (1 if age > 55)
                                     — high-risk age group indicator
            hr_reserve             : 220 − age − thalach
                                     — heart rate reserve (fitness proxy)

        Args:
            X: DataFrame with short-alias columns (pre-transform).

        Returns:
            DataFrame with original + engineered columns.
        """
        Xe = X.copy()

        # 1. Blood pressure to cholesterol ratio
        Xe['bp_chol_ratio'] = Xe['trestbps'] / (Xe['chol'] + 1e-5)

        # 2. Age-normalised maximum heart rate
        Xe['thalach_age_ratio'] = Xe['thalach'] / (Xe['age'] + 1e-5)

        # 3. ST-depression × slope interaction
        Xe['oldpeak_slope_interaction'] = Xe['oldpeak'] * Xe['slope']

        # 4. High-risk age flag (age > 55 clinically meaningful threshold)
        Xe['age_risk_flag'] = (Xe['age'] > 55).astype(int)

        # 5. Heart rate reserve  (proxy for cardiovascular fitness)
        Xe['hr_reserve'] = 220 - Xe['age'] - Xe['thalach']

        return Xe

    # ------------------------------------------------------------------
    # Public: fit + transform (training only)
    # ------------------------------------------------------------------

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit all transformers on training data and return transformed array.

        Args:
            X_train: Raw training features (short-alias columns).

        Returns:
            Transformed pd.DataFrame ready for model training.
        """
        # Step 1 — engineer new features
        Xe = self._engineer_features(X_train)

        # Step 2 — scale continuous features
        X_num_scaled = self.scaler.fit_transform(Xe[NUM_COLS])
        df_num = pd.DataFrame(X_num_scaled, columns=NUM_COLS, index=X_train.index)

        # Step 3 — one-hot encode categorical features
        X_cat_enc = self.encoder.fit_transform(Xe[CAT_COLS].astype(str))
        cat_names = list(self.encoder.get_feature_names_out(CAT_COLS))
        df_cat = pd.DataFrame(X_cat_enc, columns=cat_names, index=X_train.index)

        # Step 4 — concatenate all parts
        X_out = pd.concat([df_num, df_cat, Xe[PASS_COLS]], axis=1)

        self.feature_names = list(X_out.columns)
        self.fitted = True

        print(f"[FEATURES] Pipeline fitted. Output feature count: {len(self.feature_names)}")
        return X_out

    # ------------------------------------------------------------------
    # Public: transform (test / production)
    # ------------------------------------------------------------------

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply already-fitted transformers to new data (no re-fitting).

        Args:
            X: Raw feature DataFrame (short-alias columns).

        Returns:
            Transformed pd.DataFrame with the same column order as fit_transform.

        Raises:
            ValueError: If called before fit_transform.
        """
        if not self.fitted:
            raise ValueError(
                "[FEATURES] FeaturePipeline must be fitted via fit_transform() "
                "before calling transform()."
            )

        # Step 1 — engineer features
        Xe = self._engineer_features(X)

        # Step 2 — scale (using fitted parameters)
        X_num_scaled = self.scaler.transform(Xe[NUM_COLS])
        df_num = pd.DataFrame(X_num_scaled, columns=NUM_COLS, index=X.index)

        # Step 3 — encode (using fitted encoder)
        X_cat_enc = self.encoder.transform(Xe[CAT_COLS].astype(str))
        cat_names = list(self.encoder.get_feature_names_out(CAT_COLS))
        df_cat = pd.DataFrame(X_cat_enc, columns=cat_names, index=X.index)

        # Step 4 — concatenate
        X_out = pd.concat([df_num, df_cat, Xe[PASS_COLS]], axis=1)

        # Enforce identical column order as training output
        X_out = X_out[self.feature_names]
        return X_out
