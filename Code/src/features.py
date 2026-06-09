import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class FeaturePipeline:
    def __init__(self):
        self.scaler = StandardScaler()
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        self.fitted = False
        
        # Define continuous and categorical column sets
        self.num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'bp_chol_ratio', 'thalach_age_ratio', 'oldpeak_slope_interaction']
        self.cat_cols = ['cp', 'restecg', 'slope', 'thal']
        self.pass_cols = ['sex', 'fbs', 'exang', 'ca'] # Already binary/encoded or ordinal
        
    def _engineer_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create domain-specific interaction terms and clinical risk indices.
        """
        X_eng = X.copy()
        
        # 1. Blood pressure to cholesterol ratio (cardiovascular risk index)
        # Avoid division by zero
        X_eng['bp_chol_ratio'] = X_eng['trestbps'] / (X_eng['chol'] + 1e-5)
        
        # 2. Maximum heart rate relative to age
        X_eng['thalach_age_ratio'] = X_eng['thalach'] / (X_eng['age'] + 1e-5)
        
        # 3. ST depression relative to the slope of peak exercise
        X_eng['oldpeak_slope_interaction'] = X_eng['oldpeak'] * X_eng['slope']
        
        return X_eng

    def fit_transform(self, X_train: pd.DataFrame) -> pd.DataFrame:
        """
        Fit transformers on the training data and transform it.
        """
        # Step 1: Engineer features
        X_eng = self._engineer_features(X_train)
        
        # Step 2: Fit and transform scaling for numeric columns
        X_num_scaled = self.scaler.fit_transform(X_eng[self.num_cols])
        df_num_scaled = pd.DataFrame(X_num_scaled, columns=self.num_cols, index=X_train.index)
        
        # Step 3: Fit and transform one-hot encoding for categorical columns
        X_cat_encoded = self.encoder.fit_transform(X_eng[self.cat_cols].astype(str))
        cat_feature_names = self.encoder.get_feature_names_out(self.cat_cols)
        df_cat_encoded = pd.DataFrame(X_cat_encoded, columns=cat_feature_names, index=X_train.index)
        
        # Step 4: Concatenate everything
        X_transformed = pd.concat([
            df_num_scaled,
            df_cat_encoded,
            X_eng[self.pass_cols]
        ], axis=1)
        
        self.feature_names = list(X_transformed.columns)
        self.fitted = True
        return X_transformed

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transform test or production data using the fitted transformers.
        """
        if not self.fitted:
            raise ValueError("FeaturePipeline must be fitted before calling transform.")
            
        # Step 1: Engineer features
        X_eng = self._engineer_features(X)
        
        # Step 2: Transform numeric columns
        X_num_scaled = self.scaler.transform(X_eng[self.num_cols])
        df_num_scaled = pd.DataFrame(X_num_scaled, columns=self.num_cols, index=X.index)
        
        # Step 3: Transform categorical columns
        X_cat_encoded = self.encoder.transform(X_eng[self.cat_cols].astype(str))
        cat_feature_names = self.encoder.get_feature_names_out(self.cat_cols)
        df_cat_encoded = pd.DataFrame(X_cat_encoded, columns=cat_feature_names, index=X.index)
        
        # Step 4: Concatenate everything
        X_transformed = pd.concat([
            df_num_scaled,
            df_cat_encoded,
            X_eng[self.pass_cols]
        ], axis=1)
        
        # Order columns identically
        X_transformed = X_transformed[self.feature_names]
        return X_transformed
