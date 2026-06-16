import os
import joblib
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

# ---------------------------------------------------------------------------
# Constants: Feature names expected by the model (original 13 clinical inputs)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal'
]

# Model and pipeline artifact paths (relative to this script)
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH    = os.path.join(_BASE_DIR, 'outputs', 'best_model.pkl')
PIPELINE_SAVE_PATH = os.path.join(_BASE_DIR, 'outputs', 'feature_pipeline.pkl')


# ---------------------------------------------------------------------------
# Helper: Select the best model from the evaluated report
# ---------------------------------------------------------------------------
def _select_best_model(report_df: pd.DataFrame, trained_models: dict):
    """
    Choose the best model based on ROC-AUC score from the evaluation report.

    Args:
        report_df     : DataFrame with per-model evaluation metrics.
        trained_models: Dict mapping model name -> fitted estimator.

    Returns:
        tuple: (best_model_name, best_model_estimator)
    """
    best_row = report_df.loc[report_df['ROC-AUC'].idxmax()]
    best_name = best_row['Model']
    best_model = trained_models[best_name]
    print(f"\n[Best Model] {best_name}  |  ROC-AUC: {best_row['ROC-AUC']:.4f}")
    return best_name, best_model


# ---------------------------------------------------------------------------
# Helper: Save model artefacts with joblib
# ---------------------------------------------------------------------------
def _save_artifacts(model, pipeline: FeaturePipeline, model_path: str, pipeline_path: str):
    """
    Persist the best model and fitted FeaturePipeline to disk.

    Args:
        model        : Fitted sklearn estimator.
        pipeline     : Fitted FeaturePipeline object.
        model_path   : Destination path for the model pickle.
        pipeline_path: Destination path for the pipeline pickle.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(pipeline, pipeline_path)
    print(f"\nSaved best model      -> {model_path}")
    print(f"Saved feature pipeline -> {pipeline_path}")


# ---------------------------------------------------------------------------
# Core: User-facing prediction function
# ---------------------------------------------------------------------------
def predict_from_user_input(model, pipeline: FeaturePipeline) -> None:
    """
    Interactively prompt the user for all 13 clinical feature values,
    apply the fitted FeaturePipeline, and display the heart disease
    prediction with probability using the supplied model.

    Args:
        model   : Fitted sklearn classifier that supports predict_proba.
        pipeline: Fitted FeaturePipeline instance (already fit_transform-ed).
    """
    print("\n" + "=" * 60)
    print("          HEART DISEASE PREDICTION — PATIENT INPUT          ")
    print("=" * 60)
    print("Please enter the following clinical feature values.\n")

    # ------------------------------------------------------------------
    # Feature metadata: (prompt label, type, valid range / allowed set)
    # ------------------------------------------------------------------
    feature_prompts = {
        'age'     : ("Age (years)",                    float, None),
        'sex'     : ("Sex (1 = male, 0 = female)",     int,   {0, 1}),
        'cp'      : ("Chest Pain Type (0-3)",           int,   {0, 1, 2, 3}),
        'trestbps': ("Resting Blood Pressure (mm Hg)", float, None),
        'chol'    : ("Serum Cholesterol (mg/dl)",      float, None),
        'fbs'     : ("Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)", int, {0, 1}),
        'restecg' : ("Resting ECG Results (0-2)",      int,   {0, 1, 2}),
        'thalach' : ("Max Heart Rate Achieved",        float, None),
        'exang'   : ("Exercise Induced Angina (1=Yes, 0=No)", int, {0, 1}),
        'oldpeak' : ("ST Depression (oldpeak)",        float, None),
        'slope'   : ("Slope of Peak Exercise ST (0-2)", int,  {0, 1, 2}),
        'ca'      : ("Number of Major Vessels (0-3)",  int,   {0, 1, 2, 3}),
        'thal'    : ("Thalassemia (0=normal, 1=fixed defect, 2=reversable defect)", int, {0, 1, 2}),
    }

    values = {}

    for col in FEATURE_COLUMNS:
        label, dtype, allowed = feature_prompts[col]
        while True:
            try:
                raw = input(f"  {label}: ").strip()
                val = dtype(raw)

                # Validate against allowed set if defined
                if allowed is not None and val not in allowed:
                    print(f"    [!] Invalid value. Allowed values: {sorted(allowed)}")
                    continue

                # Basic sanity bounds
                if col in ('age', 'trestbps', 'chol', 'thalach') and val <= 0:
                    print(f"    [!] Value must be positive. Please re-enter.")
                    continue

                values[col] = val
                break

            except ValueError:
                print(f"    [!] Invalid input. Expected a {'whole number' if dtype is int else 'numeric value'}.")

    # ------------------------------------------------------------------
    # Build DataFrame, transform, and predict
    # ------------------------------------------------------------------
    try:
        input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        input_transformed = pipeline.transform(input_df)
        prediction = model.predict(input_transformed)[0]

        # Probability estimate
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(input_transformed)[0][1]
        else:
            # Fallback: use decision function and sigmoid normalisation
            import numpy as np
            score = model.decision_function(input_transformed)[0]
            prob = 1 / (1 + np.exp(-score))

    except Exception as exc:
        print(f"\n[ERROR] Prediction failed: {exc}")
        return

    # ------------------------------------------------------------------
    # Display result
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("                     PREDICTION RESULT                     ")
    print("=" * 60)
    print(f"\n  Heart Disease Risk : {prob * 100:.2f}%")

    if prediction == 1:
        print("  Prediction        : \033[91mYES\033[0m")   # Red text
        print("\n  Outcome: YES — Patient is likely to have heart disease.")
    else:
        print("  Prediction        : \033[92mNO\033[0m")    # Green text
        print("\n  Outcome: NO  — Patient is unlikely to have heart disease.")

    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# Sample test case (used by run_pipeline to demonstrate prediction)
# ---------------------------------------------------------------------------
def run_sample_test(model, pipeline: FeaturePipeline) -> None:
    """
    Run a hardcoded sample patient case through the prediction pipeline
    and display the result. Mirrors the example from the task specification.
    """
    sample = {
        'age'     : 63,
        'sex'     : 1,
        'cp'      : 3,
        'trestbps': 145,
        'chol'    : 233,
        'fbs'     : 1,
        'restecg' : 0,
        'thalach' : 150,
        'exang'   : 0,
        'oldpeak' : 2.3,
        'slope'   : 0,
        'ca'      : 0,
        'thal'    : 1,
    }

    print("\n" + "=" * 60)
    print("              SAMPLE TEST CASE (AUTO-RUN)                   ")
    print("=" * 60)
    print("  Input Features:")
    for k, v in sample.items():
        print(f"    {k:10s}: {v}")

    try:
        input_df = pd.DataFrame([sample], columns=FEATURE_COLUMNS)
        input_transformed = pipeline.transform(input_df)
        prediction = model.predict(input_transformed)[0]

        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(input_transformed)[0][1]
        else:
            import numpy as np
            score = model.decision_function(input_transformed)[0]
            prob = 1 / (1 + np.exp(-score))

    except Exception as exc:
        print(f"\n[ERROR] Sample test failed: {exc}")
        return

    print(f"\n  Heart Disease Risk : {prob * 100:.2f}%")
    label = "YES — Patient is likely to have heart disease." if prediction == 1 \
            else "NO  — Patient is unlikely to have heart disease."
    print(f"  Prediction        : {'YES' if prediction == 1 else 'NO'}")
    print(f"  Outcome: {label}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
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
    8. Model & Pipeline Serialisation (joblib)
    9. Sample test-case prediction demonstration
    10. Interactive user-input prediction
    """
    print("=" * 60)
    print("                 CORDIS ML PIPELINE RUNNER                  ")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 1 & 2. Data Loading & Cleaning
    # ------------------------------------------------------------------
    df_raw = load_data(data_path)
    df_cleaned = clean_data(df_raw, drop_duplicates=True)

    # ------------------------------------------------------------------
    # 3. Stratified Splitting
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(
        df_cleaned, target_col='target', test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------
    # 4. Feature Pipeline
    # ------------------------------------------------------------------
    print("\nStarting feature engineering and scaling...")
    pipeline = FeaturePipeline()
    X_train_trans = pipeline.fit_transform(X_train)
    X_test_trans  = pipeline.transform(X_test)
    print(f"Feature transformation completed. Features count: {len(pipeline.feature_names)}")

    # ------------------------------------------------------------------
    # 5. Model Training & Tuning
    # ------------------------------------------------------------------
    print("\nStarting model training and cross-validated tuning...")
    trained_models = train_and_tune(X_train_trans, y_train, cv_folds=5)

    # ------------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------------
    print("\nEvaluating trained models on the independent test set...")
    report_df = generate_report(trained_models, X_test_trans, y_test)

    print("\nAggregated Performance Report:")
    print(report_df.to_string(index=False))

    # Save metrics CSV
    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    report_df.to_csv(metrics_output_path, index=False)
    print(f"\nSaved metrics comparison to: {metrics_output_path}")

    # ------------------------------------------------------------------
    # 7. Visualization
    # ------------------------------------------------------------------
    print("\nGenerating evaluation plots and figures...")
    os.makedirs(output_images_dir, exist_ok=True)
    plot_confusion_matrices(trained_models, X_test_trans, y_test, output_images_dir)
    plot_roc_curves(trained_models, X_test_trans, y_test, output_images_dir)
    plot_feature_importances(trained_models, pipeline.feature_names, output_images_dir)
    plot_model_comparison(report_df, output_images_dir)

    # ------------------------------------------------------------------
    # 8. Select best model and save artefacts
    # ------------------------------------------------------------------
    best_name, best_model = _select_best_model(report_df, trained_models)
    _save_artifacts(best_model, pipeline, MODEL_SAVE_PATH, PIPELINE_SAVE_PATH)

    # ------------------------------------------------------------------
    # 9. Sample test-case demonstration (no user interaction)
    # ------------------------------------------------------------------
    run_sample_test(best_model, pipeline)

    # ------------------------------------------------------------------
    # 10. Interactive user prediction
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    predict_from_user_input(best_model, pipeline)

    print("\n" + "=" * 60)
    print("             CORDIS PIPELINE EXECUTION SUCCESS              ")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    BASE_DIR            = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH           = os.path.join(BASE_DIR, '..', 'Datasets', 'heart.csv')
    OUTPUT_IMAGES_DIR   = os.path.join(BASE_DIR, '..', 'Images')
    METRICS_OUTPUT_PATH = os.path.join(BASE_DIR, 'outputs', 'model_metrics.csv')

    run_pipeline(DATA_PATH, OUTPUT_IMAGES_DIR, METRICS_OUTPUT_PATH)
