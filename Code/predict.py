"""
predict.py
==========
CORDIS — Standalone Heart Disease Prediction Tool
=================================================
Loads the pre-trained best model and fitted FeaturePipeline serialised
by main.py, then provides an interactive clinical prediction interface.

Prerequisites:
    Run `python main.py` at least once to generate:
        outputs/best_model.pkl
        outputs/feature_pipeline.pkl

Usage:
    cd Code
    python predict.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Artefact paths (resolved relative to this script, regardless of CWD)
# ---------------------------------------------------------------------------
_BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.path.join(_BASE_DIR, 'outputs', 'best_model.pkl')
PIPELINE_PATH = os.path.join(_BASE_DIR, 'outputs', 'feature_pipeline.pkl')

# ---------------------------------------------------------------------------
# Feature column definitions (short-alias order used during training)
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol',
    'fbs', 'restecg', 'thalach', 'exang',
    'oldpeak', 'slope', 'ca', 'thal',
]

# ---------------------------------------------------------------------------
# Per-feature prompt metadata
#   (display label, cast type, allowed value set or None)
# ---------------------------------------------------------------------------
FEATURE_PROMPTS = {
    'age'     : ("Age (years)",
                 float, None),
    'sex'     : ("Sex (1 = Male, 0 = Female)",
                 int,   {0, 1}),
    'cp'      : ("Chest Pain Type  "
                 "0=Asymptomatic | 1=Atypical Angina | "
                 "2=Non-Anginal | 3=Typical Angina",
                 int,   {0, 1, 2, 3}),
    'trestbps': ("Resting Blood Pressure (mm Hg)",
                 float, None),
    'chol'    : ("Serum Cholesterol (mg/dl)",
                 float, None),
    'fbs'     : ("Fasting Blood Sugar > 120 mg/dl  (1=Yes, 0=No)",
                 int,   {0, 1}),
    'restecg' : ("Resting ECG Results  "
                 "0=Normal | 1=ST-T Abnormality | 2=LV Hypertrophy",
                 int,   {0, 1, 2}),
    'thalach' : ("Maximum Heart Rate Achieved (bpm)",
                 float, None),
    'exang'   : ("Exercise-Induced Angina  (1=Yes, 0=No)",
                 int,   {0, 1}),
    'oldpeak' : ("ST Depression Induced by Exercise (oldpeak)",
                 float, None),
    'slope'   : ("Slope of Peak ST Segment  "
                 "0=Downsloping | 1=Flat | 2=Upsloping",
                 int,   {0, 1, 2}),
    'ca'      : ("Number of Major Vessels Coloured by Fluoroscopy (0–4)",
                 int,   {0, 1, 2, 3, 4}),
    'thal'    : ("Thalassemia  "
                 "0=Normal | 1=Fixed Defect | 2=Reversible Defect | 3=Unknown",
                 int,   {0, 1, 2, 3}),
}

_POSITIVE_COLS = {'age', 'trestbps', 'chol', 'thalach'}


# ---------------------------------------------------------------------------
# Artefact loading
# ---------------------------------------------------------------------------

def load_artifacts(model_path: str, pipeline_path: str):
    """
    Load the serialised best model and FeaturePipeline from disk.

    Args:
        model_path   : Absolute path to best_model.pkl.
        pipeline_path: Absolute path to feature_pipeline.pkl.

    Returns:
        tuple: (fitted model, fitted FeaturePipeline)

    Raises:
        FileNotFoundError: If either artefact is missing.
    """
    for path, label in [(model_path, 'Model'), (pipeline_path, 'Pipeline')]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{label} artefact not found:\n  {path}\n\n"
                "Please run `python main.py` first to train and save the model."
            )

    model    = joblib.load(model_path)
    pipeline = joblib.load(pipeline_path)
    print(f"  Model loaded    : {model_path}")
    print(f"  Pipeline loaded : {pipeline_path}")
    return model, pipeline


# ---------------------------------------------------------------------------
# Input collection
# ---------------------------------------------------------------------------

def collect_user_input() -> pd.DataFrame:
    """
    Interactively collect 13 patient clinical feature values.

    Validates type, discrete-set membership, and positivity constraints.

    Returns:
        pd.DataFrame: Single-row DataFrame with short-alias column names.
    """
    print("\n" + "=" * 65)
    print("       CORDIS — HEART DISEASE PREDICTION: PATIENT INPUT       ")
    print("=" * 65)
    print("Enter the following clinical measurements for the patient:\n")

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
                    print(f"    [!] Must be positive. Please re-enter.")
                    continue

                values[col] = val
                break

            except ValueError:
                kind = "whole number" if dtype is int else "numeric value"
                print(f"    [!] Expected a {kind}. Try again.")

    return pd.DataFrame([values], columns=FEATURE_COLUMNS)


# ---------------------------------------------------------------------------
# Prediction + display
# ---------------------------------------------------------------------------

def predict_and_display(model, pipeline, input_df: pd.DataFrame) -> None:
    """
    Transform input, run prediction, and print result to console.

    Args:
        model    : Loaded fitted classifier.
        pipeline : Loaded fitted FeaturePipeline.
        input_df : Single-row feature DataFrame.
    """
    # Transform
    try:
        X_t = pipeline.transform(input_df)
    except Exception as exc:
        print(f"\n[ERROR] Feature transformation failed: {exc}")
        sys.exit(1)

    # Predict label
    try:
        pred = model.predict(X_t)[0]
    except Exception as exc:
        print(f"\n[ERROR] Prediction failed: {exc}")
        sys.exit(1)

    # Probability estimate
    try:
        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_t)[0][1]
        elif hasattr(model, 'decision_function'):
            score = model.decision_function(X_t)[0]
            prob  = 1.0 / (1.0 + np.exp(-score))
        else:
            prob = float(pred)
    except Exception as exc:
        print(f"\n[WARNING] Could not compute probability: {exc}")
        prob = float(pred)

    # Display
    print("\n" + "=" * 65)
    print("                     PREDICTION RESULT                       ")
    print("=" * 65)
    print(f"\n  Heart Disease Risk  : {prob * 100:.2f}%")

    if pred == 1:
        print("  Prediction         : \033[91mYES - Disease Likely\033[0m")
        print("\n  [!] Outcome: Clinical profile suggests PRESENCE of")
        print("     heart disease. Further diagnostic evaluation advised.")
    else:
        print("  Prediction         : \033[92mNO  - Disease Unlikely\033[0m")
        print("\n  [OK] Outcome: Clinical profile suggests LOW risk of")
        print("     heart disease based on current measurements.")

    print("=" * 65 + "\n")


# ---------------------------------------------------------------------------
# Sample reference test
# ---------------------------------------------------------------------------

def run_sample_prediction(model, pipeline) -> None:
    """
    Run a known sample patient (row 1, Cleveland dataset) to confirm
    the loaded model is producing valid predictions.
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

    print("\n" + "=" * 65)
    print("          SAMPLE REFERENCE TEST (known positive case)         ")
    print("=" * 65)
    for k, v in sample.items():
        print(f"  {k:<12}: {v}")

    try:
        df   = pd.DataFrame([sample], columns=FEATURE_COLUMNS)
        X_t  = pipeline.transform(df)
        pred = model.predict(X_t)[0]

        if hasattr(model, 'predict_proba'):
            prob = model.predict_proba(X_t)[0][1]
        elif hasattr(model, 'decision_function'):
            score = model.decision_function(X_t)[0]
            prob  = 1.0 / (1.0 + np.exp(-score))
        else:
            prob = float(pred)

        print(f"\n  Risk Probability   : {prob * 100:.2f}%")
        verdict = "YES — Disease Likely" if pred == 1 else "NO — Disease Unlikely"
        print(f"  Prediction         : {verdict}")

    except Exception as exc:
        print(f"\n[ERROR] Sample test failed: {exc}")

    print("=" * 65)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 65)
    print("       CORDIS — HEART DISEASE PREDICTION TOOL v2.0           ")
    print("=" * 65)
    print("\nLoading saved model artefacts...\n")

    try:
        model, pipeline = load_artifacts(MODEL_PATH, PIPELINE_PATH)
    except FileNotFoundError as exc:
        print(f"\n[ERROR] {exc}")
        sys.exit(1)

    print(f"\n  Model type: {type(model).__name__}")

    # Sanity check with known sample
    run_sample_prediction(model, pipeline)

    # Interactive prediction loop
    while True:
        input_df = collect_user_input()
        predict_and_display(model, pipeline, input_df)

        again = input("Predict for another patient? (y/n): ").strip().lower()
        if again != 'y':
            print("\nExiting CORDIS Prediction Tool. Goodbye!\n")
            break


if __name__ == "__main__":
    main()
