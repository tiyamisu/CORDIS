import os
import pandas as pd
from src.preprocessing import load_data, clean_data, split_data
from src.features import FeaturePipeline
from src.train import train_and_tune
from src.evaluate import generate_report
from src.visualize import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_feature_importances,
    plot_model_comparison
)

def run_pipeline(data_path: str, output_images_dir: str, metrics_output_path: str):
    """
    Run the end-to-end CORDIS machine learning pipeline:
    1. Data Loading & Understanding
    2. Data Preprocessing & Cleaning (including Deduplication)
    3. Stratified Train-Test Splitting
    4. Feature Engineering & Scaling (handling categorical and continuous inputs)
    5. Model Training & Cross-Validated Hyperparameter Tuning
    6. Performance Evaluation
    7. Metric Visualization Output
    """
    print("="*60)
    print("                 CORDIS ML PIPELINE RUNNER                  ")
    print("="*60)
    
    # 1 & 2. Data Loading & Cleaning
    df_raw = load_data(data_path)
    df_cleaned = clean_data(df_raw, drop_duplicates=True)
    
    # 3. Stratified Splitting
    X_train, X_test, y_train, y_test = split_data(df_cleaned, target_col='target', test_size=0.2, random_state=42)
    
    # 4. Feature Pipeline
    print("\nStarting feature engineering and scaling...")
    pipeline = FeaturePipeline()
    X_train_trans = pipeline.fit_transform(X_train)
    X_test_trans = pipeline.transform(X_test)
    print(f"Feature transformation completed. Features count: {len(pipeline.feature_names)}")
    
    # 5. Model Training & Tuning
    print("\nStarting model training and cross-validated tuning...")
    trained_models = train_and_tune(X_train_trans, y_train, cv_folds=5)
    
    # 6. Evaluation
    print("\nEvaluating trained models on the independent test set...")
    report_df = generate_report(trained_models, X_test_trans, y_test)
    
    # Print metrics report summary
    print("\nAggregated Performance Report:")
    print(report_df.to_string(index=False))
    
    # Save metrics report to file
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    report_df.to_csv(metrics_output_path, index=False)
    print(f"\nSaved metrics comparison to: {metrics_output_path}")
    
    # 7. Visualization
    print("\nGenerating evaluation plots and figures...")
    os.makedirs(output_images_dir, exist_ok=True)
    
    plot_confusion_matrices(trained_models, X_test_trans, y_test, output_images_dir)
    plot_roc_curves(trained_models, X_test_trans, y_test, output_images_dir)
    plot_feature_importances(trained_models, pipeline.feature_names, output_images_dir)
    plot_model_comparison(report_df, output_images_dir)
    
    print("\n" + "="*60)
    print("             CORDIS PIPELINE EXECUTION SUCCESS              ")
    print("="*60)

if __name__ == "__main__":
    # Define paths relative to the script execution folder or root
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, '..', 'Datasets', 'heart.csv')
    OUTPUT_IMAGES_DIR = os.path.join(BASE_DIR, '..', 'Images')
    METRICS_OUTPUT_PATH = os.path.join(BASE_DIR, 'outputs', 'model_metrics.csv')
    
    run_pipeline(DATA_PATH, OUTPUT_IMAGES_DIR, METRICS_OUTPUT_PATH)
