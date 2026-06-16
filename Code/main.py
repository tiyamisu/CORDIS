"""
main.py
=======
CORDIS — Heart Disease Prediction ML Pipeline
=============================================
End-to-end orchestration script for the CORDIS project.

Pipeline stages:
    1.  Data loading & column normalisation
    2.  Data cleaning (deduplication, imputation, outlier capping)
    3.  Stratified 80/20 train–test split
    4.  Feature engineering & scaling (FeaturePipeline)
    5.  Multi-model training with cross-validated GridSearchCV
    6.  Independent test-set evaluation
    7.  Visualisation (confusion matrices, ROC curves, feature
        importances, metric comparison)
    8.  Best-model selection (by ROC-AUC) & serialisation
    9.  Sample test-case demonstration
    10. Interactive patient prediction

Usage:
    cd Code
    python main.py
"""

import os
import sys
import joblib
import pandas as pd

from src.preprocessing import load_data, clean_data, split_data
from src.features       import FeaturePipeline
from src.train          import train_and_tune
from src.evaluate       import generate_report
from src.visualize      import (
    plot_confusion_matrices,
    plot_roc_curves,
    plot_feature_importances,
    plot_model_comparison,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Short-alias feature columns (order matches training)
FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal',
]

_BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH    = os.path.join(_BASE_DIR, 'outputs', 'best_model.pkl')
PIPELINE_SAVE_PATH = os.path.join(_BASE_DIR, 'outputs', 'feature_pipeline.pkl')

# ---------------------------------------------------------------------------
# Per-feature input metadata for interactive prediction
#   key   → column alias
#   value → (display prompt, cast type, allowed value set or None)
# ---------------------------------------------------------------------------
FEATURE_PROMPTS = {
    'age'     : ("Age (years)",                              float, None),
    'sex'     : ("Sex (1 = Male, 0 = Female)",              int,   {0, 1}),
    'cp'      : ("Chest Pain Type (0=Asymptomatic, 1=Atypical Angina, "
                 "2=Non-Anginal, 3=Typical Angina)",        int,   {0, 1, 2, 3}),
    'trestbps': ("Resting Blood Pressure (mm Hg)",          float, None),
    'chol'    : ("Serum Cholesterol (mg/dl)",               float, None),
    'fbs'     : ("Fasting Blood Sugar > 120 mg/dl "
                 "(1 = True, 0 = False)",                   int,   {0, 1}),
    'restecg' : ("Resting ECG Results (0=Normal, "
                 "1=ST-T Abnormality, 2=LV Hypertrophy)",   int,   {0, 1, 2}),
    'thalach' : ("Maximum Heart Rate Achieved (bpm)",        float, None),
    'exang'   : ("Exercise-Induced Angina (1=Yes, 0=No)",   int,   {0, 1}),
    'oldpeak' : ("ST Depression Induced by Exercise "
                 "(oldpeak value)",                          float, None),
    'slope'   : ("Slope of Peak Exercise ST Segment "
                 "(0=Downsloping, 1=Flat, 2=Upsloping)",    int,   {0, 1, 2}),
    'ca'      : ("Number of Major Vessels (0–4) Coloured "
                 "by Fluoroscopy",                           int,   {0, 1, 2, 3, 4}),
    'thal'    : ("Thalassemia (0=Normal, 1=Fixed Defect, "
                 "2=Reversible Defect, 3=Unknown)",          int,   {0, 1, 2, 3}),
}

# Positive-valued continuous columns (guard for nonsensical entries)
_POSITIVE_COLS = {'age', 'trestbps', 'chol', 'thalach'}


# ---------------------------------------------------------------------------
# Helper: select best model
# ---------------------------------------------------------------------------

def _select_best_model(
    report_df: pd.DataFrame,
    trained_models: dict,
) -> tuple:
    """
    Choose the model with the highest ROC-AUC on the test set.

    Args:
        report_df     : Evaluation results DataFrame.
        trained_models: Name → fitted estimator mapping.

    Returns:
        (best_model_name, best_model_estimator)
    """
    best_row  = report_df.loc[report_df['ROC-AUC'].idxmax()]
    best_name = best_row['Model']
    best_est  = trained_models[best_name]
    print(f"\n[BEST MODEL] {best_name}  |  ROC-AUC = {best_row['ROC-AUC']:.4f}")
    return best_name, best_est


# ---------------------------------------------------------------------------
# Helper: save artefacts
# ---------------------------------------------------------------------------

def _save_artifacts(
    model,
    pipeline: FeaturePipeline,
    model_path: str,
    pipeline_path: str,
) -> None:
    """
    Serialise the best model and fitted FeaturePipeline with joblib.

    Args:
        model        : Fitted sklearn estimator.
        pipeline     : Fitted FeaturePipeline.
        model_path   : Destination .pkl path for the model.
        pipeline_path: Destination .pkl path for the pipeline.
    """
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model,    model_path)
    joblib.dump(pipeline, pipeline_path)
    print(f"\n[SAVE] Best model      → {model_path}")
    print(f"[SAVE] Feature pipeline → {pipeline_path}")


# ---------------------------------------------------------------------------
# Interactive prediction
# ---------------------------------------------------------------------------

def predict_from_user_input(model, pipeline: FeaturePipeline) -> None:
    """
    Prompt the clinician / operator for all 13 patient feature values,
    transform them through the fitted FeaturePipeline, and display the
    heart disease prediction with associated probability.

    Args:
        model   : Best fitted sklearn classifier.
        pipeline: Fitted FeaturePipeline.
    """
    import numpy as np

    print("\n" + "=" * 62)
    print("       CORDIS — HEART DISEASE PREDICTION: PATIENT INPUT       ")
    print("=" * 62)
    print("Enter the following clinical measurements for the patient.\n")

    values: dict = {}

    for col in FEATURE_COLUMNS:
        label, dtype, allowed = FEATURE_PROMPTS[col]
        while True:
            try:
                raw = input(f"  {label}: ").strip()
                val = dtype(raw)

                if allowed is not None and val not in allowed:
                    print(f"    [!] Invalid. Allowed values: {sorted(allowed)}")
                    continue

                if col in _POSITIVE_COLS and val <= 0:
                    print(f"    [!] Must be a positive number. Please re-enter.")
                    continue

                values[col] = val
                break

            except ValueError:
                kind = "whole number" if dtype is int else "numeric value"
                print(f"    [!] Expected a {kind}. Try again.")

    try:
        input_df = pd.DataFrame([values], columns=FEATURE_COLUMNS)
        input_t  = pipeline.transform(input_df)
        pred     = model.predict(input_t)[0]

        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(input_t)[0][1]
        elif hasattr(model, 'decision_function'):
            score = model.decision_function(input_t)[0]
            prob  = 1.0 / (1.0 + np.exp(-score))
        else:
            prob = float(pred)

    except Exception as exc:
        print(f"\n[ERROR] Prediction failed: {exc}")
        return

    _display_result(pred, prob)


def _display_result(prediction: int, probability: float) -> None:
    """Pretty-print the prediction result."""
    print("\n" + "=" * 62)
    print("                    PREDICTION RESULT                        ")
    print("=" * 62)
    print(f"\n  Heart Disease Risk  : {probability * 100:.2f}%")

    if prediction == 1:
        print("  Prediction         : \033[91mYES - Disease Likely\033[0m")
        print("\n  [!] Outcome: Patient shows clinical indicators associated")
        print("     with the PRESENCE of heart disease.")
        print("     Recommend further diagnostic evaluation.")
    else:
        print("  Prediction         : \033[92mNO  - Disease Unlikely\033[0m")
        print("\n  [OK] Outcome: Patient profile suggests LOW cardiovascular")
        print("     disease risk based on current clinical measurements.")

    print("=" * 62 + "\n")


# ---------------------------------------------------------------------------
# Sample test case
# ---------------------------------------------------------------------------

def run_sample_test(model, pipeline: FeaturePipeline) -> None:
    """
    Run a hardcoded sample patient through the pipeline to confirm
    the serialised model is working correctly after loading.

    Sample corresponds to the first row of the Cleveland dataset:
        63-year-old male with typical angina and high ST depression.
    """
    import numpy as np

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

    print("\n" + "=" * 62)
    print("              SAMPLE TEST CASE (Row 1 — Known Positive)       ")
    print("=" * 62)
    for k, v in sample.items():
        print(f"  {k:<12}: {v}")

    try:
        df   = pd.DataFrame([sample], columns=FEATURE_COLUMNS)
        t    = pipeline.transform(df)
        pred = model.predict(t)[0]

        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(t)[0][1]
        else:
            score = model.decision_function(t)[0]
            prob  = 1.0 / (1.0 + np.exp(-score))

        print(f"\n  Risk Probability : {prob * 100:.2f}%")
        print(f"  Prediction       : {'YES — Disease Likely' if pred == 1 else 'NO — Disease Unlikely'}")

    except Exception as exc:
        print(f"\n[ERROR] Sample test failed: {exc}")

    print("=" * 62)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    data_path: str,
    output_images_dir: str,
    metrics_output_path: str,
) -> None:
    """
    Execute the complete CORDIS ML pipeline from raw data to saved model.

    Args:
        data_path          : Path to `heart_dataset.csv`.
        output_images_dir  : Directory for saving visualisation figures.
        metrics_output_path: Path for saving the metrics CSV report.
    """
    print("=" * 62)
    print("            CORDIS ML PIPELINE — STARTING                     ")
    print("=" * 62)

    # ------------------------------------------------------------------
    # Stages 1 & 2: Load and Clean
    # ------------------------------------------------------------------
    df_raw     = load_data(data_path)
    df_cleaned = clean_data(df_raw, drop_duplicates=True)

    # ------------------------------------------------------------------
    # Stage 3: Split
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test = split_data(
        df_cleaned, target_col='target', test_size=0.2, random_state=42
    )

    # ------------------------------------------------------------------
    # Stage 4: Feature Pipeline
    # ------------------------------------------------------------------
    print("\n[PIPELINE] Building feature engineering pipeline...")
    pipeline    = FeaturePipeline()
    X_train_t   = pipeline.fit_transform(X_train)
    X_test_t    = pipeline.transform(X_test)

    # ------------------------------------------------------------------
    # Stage 5: Train
    # ------------------------------------------------------------------
    trained_models = train_and_tune(X_train_t, y_train, cv_folds=5)

    # ------------------------------------------------------------------
    # Stage 6: Evaluate
    # ------------------------------------------------------------------
    print("\n[PIPELINE] Evaluating models on independent test set...")
    report_df = generate_report(trained_models, X_test_t, y_test)

    print("\n[PIPELINE] ═══ Aggregated Performance Report ═══")
    print(report_df.to_string(index=False))

    os.makedirs(os.path.dirname(metrics_output_path), exist_ok=True)
    report_df.to_csv(metrics_output_path, index=False)
    print(f"\n[PIPELINE] Metrics saved to: {metrics_output_path}")

    # ------------------------------------------------------------------
    # Stage 7: Visualise
    # ------------------------------------------------------------------
    print("\n[PIPELINE] Generating evaluation visualisations...")
    os.makedirs(output_images_dir, exist_ok=True)
    plot_confusion_matrices(trained_models, X_test_t, y_test, output_images_dir)
    plot_roc_curves(trained_models, X_test_t, y_test, output_images_dir)
    plot_feature_importances(trained_models, pipeline.feature_names, output_images_dir)
    plot_model_comparison(report_df, output_images_dir)

    # ------------------------------------------------------------------
    # Stage 8: Select best & save
    # ------------------------------------------------------------------
    best_name, best_model = _select_best_model(report_df, trained_models)
    _save_artifacts(best_model, pipeline, MODEL_SAVE_PATH, PIPELINE_SAVE_PATH)

    # ------------------------------------------------------------------
    # Stage 9: Sample test
    # ------------------------------------------------------------------
    run_sample_test(best_model, pipeline)

    # ------------------------------------------------------------------
    # Stage 10: Interactive prediction
    # ------------------------------------------------------------------
    print("\n" + "=" * 62)
    predict_from_user_input(best_model, pipeline)

    print("\n" + "=" * 62)
    print("          CORDIS PIPELINE — COMPLETED SUCCESSFULLY            ")
    print("=" * 62)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _CODE_DIR           = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT       = os.path.dirname(_CODE_DIR)

    DATA_PATH           = os.path.join(_PROJECT_ROOT, 'Datasets', 'heart_dataset.csv')
    OUTPUT_IMAGES_DIR   = os.path.join(_PROJECT_ROOT, 'Images')
    METRICS_OUTPUT_PATH = os.path.join(_CODE_DIR, 'outputs', 'model_metrics.csv')

    # Ensure running from Code directory so src imports resolve
    sys.path.insert(0, _CODE_DIR)

    run_pipeline(DATA_PATH, OUTPUT_IMAGES_DIR, METRICS_OUTPUT_PATH)
