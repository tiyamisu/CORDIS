import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the heart disease dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Dataset loaded successfully with shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading dataset from {file_path}: {e}")
        raise e

def clean_data(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    """
    Clean the dataset by removing duplicate rows and handling missing values.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        drop_duplicates (bool): Whether to remove exact duplicate rows. Default is True.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df_cleaned = df.copy()
    
    # Check for duplicates
    duplicate_count = df_cleaned.duplicated().sum()
    if duplicate_count > 0:
        print(f"Found {duplicate_count} duplicate rows.")
        if drop_duplicates:
            df_cleaned = df_cleaned.drop_duplicates().reset_index(drop=True)
            print(f"Removed duplicates. New shape: {df_cleaned.shape}")
    else:
        print("No duplicate rows found.")
        
    # Check for missing values
    missing_values = df_cleaned.isnull().sum().sum()
    if missing_values > 0:
        print(f"Found {missing_values} missing values. Imputing with median for numeric columns...")
        # Simple imputation for numeric features if any nulls exist
        for col in df_cleaned.columns:
            if df_cleaned[col].isnull().any():
                df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    else:
        print("No missing values found.")
        
    return df_cleaned

def split_data(df: pd.DataFrame, target_col: str = 'target', test_size: float = 0.2, random_state: int = 42):
    """
    Split the dataset into training and testing sets, using stratification on the target.
    
    Args:
        df (pd.DataFrame): Dataframe to split.
        target_col (str): The label/target column name.
        test_size (float): Proportion of dataset to include in test split.
        random_state (int): Seed for reproducibility.
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    
    print(f"Data split completed:")
    print(f"  X_train shape: {X_train.shape}, y_train distribution:\n{y_train.value_counts(normalize=True)}")
    print(f"  X_test shape: {X_test.shape}, y_test distribution:\n{y_test.value_counts(normalize=True)}")
    
    return X_train, X_test, y_train, y_test
