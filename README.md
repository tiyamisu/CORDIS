# CORDIS: Cardiovascular Diagnostic Support System

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-F7931E?logo=scikit-learn)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.5%2B-9ACD32)](https://xgboost.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**CORDIS** (*Cardiovascular Diagnostic Intelligence System*) is a production-quality, end-to-end machine learning framework for early detection of coronary artery disease. It transforms 13 clinical and diagnostic measurements into interpretable, probability-based heart disease predictions using five cross-validated classifiers.

---

## 🩺 Project Overview

Heart disease is the leading cause of death globally. CORDIS leverages the **Cleveland Heart Disease Dataset** (UCI ML Repository) — one of the most cited medical benchmarks — to build, evaluate, and deploy clinical-grade binary classifiers. The system ingests non-invasive patient measurements (blood pressure, cholesterol, ECG results, etc.) and outputs a calibrated disease probability alongside a binary diagnosis.

**Problem Type:** Binary Classification  
**Target Variable:** Presence of Heart Disease (0 = No Disease, 1 = Disease Present)

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Multi-model training** | Logistic Regression, Decision Tree, KNN, Random Forest, XGBoost |
| **Cross-validated tuning** | StratifiedKFold GridSearchCV (5-fold, F1 optimised) |
| **Outlier handling** | IQR-based winsorization on continuous features |
| **Feature engineering** | 5 derived clinical features (BP/Cholesterol ratio, HR reserve, etc.) |
| **Comprehensive EDA** | 13 visualisation types saved at 300 dpi |
| **Production-ready CLI** | Interactive prediction tool with input validation |
| **Model serialisation** | joblib persistence for zero-retrain deployment |

---

## 📊 Dataset Description

The Cleveland Heart Disease Dataset contains **303 unique patient records** (the `heart_dataset.csv` includes augmented rows; duplicates are automatically removed before training).

| # | Feature | Alias | Type | Range / Values |
|---|---|---|---|---|
| 1 | Age | `age` | Continuous | 29–77 years |
| 2 | Sex | `sex` | Binary | 0 = Female, 1 = Male |
| 3 | Chest Pain Type | `cp` | Ordinal | 0 = Asymptomatic, 1 = Atypical angina, 2 = Non-anginal, 3 = Typical angina |
| 4 | Resting Blood Pressure | `trestbps` | Continuous | mm Hg |
| 5 | Serum Cholesterol | `chol` | Continuous | mg/dl |
| 6 | Fasting Blood Sugar | `fbs` | Binary | 1 = >120 mg/dl, 0 = ≤120 mg/dl |
| 7 | Resting ECG | `restecg` | Ordinal | 0 = Normal, 1 = ST-T abnormality, 2 = LV hypertrophy |
| 8 | Max Heart Rate | `thalach` | Continuous | bpm |
| 9 | Exercise Angina | `exang` | Binary | 1 = Yes, 0 = No |
| 10 | ST Depression | `oldpeak` | Continuous | Exercise-induced ST depression |
| 11 | Slope of Peak ST | `slope` | Ordinal | 0 = Downsloping, 1 = Flat, 2 = Upsloping |
| 12 | Vessels (Fluoroscopy) | `ca` | Ordinal | 0–4 major vessels |
| 13 | Thalassemia | `thal` | Ordinal | 0 = Normal, 1 = Fixed defect, 2 = Reversible defect |
| 14 | **Target** | `target` | **Binary** | **1 = Disease, 0 = No Disease** |

---

## 📂 Repository Structure

```text
CORDIS/
├── Datasets/
│   └── heart_dataset.csv           ← Cleveland Heart Disease dataset (augmented)
├── Images/                         ← EDA & evaluation plots (auto-generated)
├── Papers/                         ← Reference academic literature
├── Report/                         ← Academic project report
├── Literature_Survey.xlsx          ← Tabulated literature review
├── Code/
│   ├── main.py                     ← End-to-end ML pipeline
│   ├── predict.py                  ← Standalone interactive prediction tool
│   ├── run_eda_plots.py            ← Comprehensive EDA script (13 figures)
│   ├── generate_comparison.py      ← Performance table generator
│   ├── requirements.txt            ← Python dependencies
│   └── src/
│       ├── __init__.py
│       ├── preprocessing.py        ← Load, rename, clean, split
│       ├── features.py             ← Feature engineering & scaling pipeline
│       ├── train.py                ← Multi-model GridSearchCV training
│       ├── evaluate.py             ← Metrics, confusion matrix, report
│       └── visualize.py            ← Confusion matrices, ROC, importances
├── README.md                       ← This file
└── how_to_run.md                   ← Quick-start instructions
```

---

## 🛠️ Installation & Setup

### 1. Clone & Navigate
```bash
git clone https://github.com/tiyamisu/CORDIS.git
cd CORDIS
```

### 2. Create Virtual Environment
```powershell
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```
```bash
# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r Code/requirements.txt
```

---

## 🚀 Usage

### Step 1 — Generate EDA Visualisations
Run from the **project root** directory:
```bash
python Code/run_eda_plots.py
```
*Outputs 13 PNG figures + 1 text summary to `Images/`*

### Step 2 — Train & Evaluate Models
Run from the **Code/** directory:
```bash
cd Code
python main.py
```
This executes the full pipeline:
- Loads `Datasets/heart_dataset.csv`, deduplicates, and cleans
- Trains 5 models with 5-fold GridSearchCV
- Evaluates on independent 20% test set
- Saves plots to `Images/`, metrics CSV to `Code/outputs/`
- Serialises the best model to `Code/outputs/best_model.pkl`
- Runs an interactive patient prediction at the end

### Step 3 — Predict (Standalone)
```bash
cd Code
python predict.py
```
*Loads saved model; prompts for 13 clinical values; displays diagnosis + probability.*

---

## 📈 Model Performance

> Results on stratified 20% held-out test set (121 patients). Best model selected by **ROC-AUC**.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (L₂) | 76.86% | 76.27% | 76.27% | 76.27% | 0.8650 |
| Decision Tree | 79.34% | 76.56% | 83.05% | 79.67% | 0.8438 |
| K-Nearest Neighbors | 82.64% | 81.67% | 83.05% | 82.35% | 0.8790 |
| **Random Forest** ⭐ | **78.51%** | **77.97%** | **77.97%** | **77.97%** | **0.9002** |
| XGBoost | 78.51% | 78.95% | 76.27% | 77.59% | 0.8647 |

**Best Model:** Random Forest (ROC-AUC = 0.9002) — saved to `Code/outputs/best_model.pkl`

Sample prediction: 63yo male with typical angina → Risk: **68.50%** → YES — Disease Likely ✅

---

## 🔬 Feature Engineering

Five domain-specific features are derived within `FeaturePipeline`:

| Feature | Formula | Clinical Rationale |
|---|---|---|
| `bp_chol_ratio` | `trestbps / (chol + ε)` | Combined cardiovascular risk index |
| `thalach_age_ratio` | `thalach / (age + ε)` | Age-normalised cardiac output capacity |
| `oldpeak_slope_interaction` | `oldpeak × slope` | Combined ST segment risk signal |
| `age_risk_flag` | `1 if age > 55 else 0` | High-risk age group indicator |
| `hr_reserve` | `220 − age − thalach` | Heart rate reserve (fitness proxy) |

---

## 🎨 Visualisation Gallery

| Figure | Description |
|---|---|
| `eda_01_target_distribution.png` | Class balance bar + pie |
| `eda_02_age_distribution.png` | Age histogram & KDE by target |
| `eda_03_sex_distribution.png` | Sex × disease countplot |
| `eda_04_chest_pain_distribution.png` | Chest pain type × disease |
| `eda_05_numerical_histograms.png` | Grid of 5 continuous feature histograms |
| `eda_06_boxplots_by_target.png` | Boxplots of continuous features vs target |
| `eda_07_correlation_heatmap.png` | Pearson correlation heatmap |
| `eda_08_pairplot.png` | Scatter pair plot (key features) |
| `eda_09_outlier_boxplots.png` | IQR outlier detection boxplots |
| `eda_10_class_balance.png` | Detailed class proportion chart |
| `eda_11_categorical_analysis.png` | All categorical features vs target |
| `eda_12_distribution_by_target.png` | KDE density plots by target |
| `confusion_matrices.png` | Combined confusion matrix canvas |
| `roc_curves.png` | Overlaid ROC curves with AUC scores |
| `feature_importances.png` | Top features from best tree model |
| `model_comparison.png` | Grouped bar chart of all metrics |

---

## 🔮 Future Enhancements

- **Explainable AI:** SHAP force plots and LIME explanations for individual predictions
- **Web Interface:** Flask/FastAPI REST API with a Streamlit dashboard
- **Federated Learning:** Privacy-preserving multi-hospital model aggregation
- **Calibration Curves:** Platt scaling for well-calibrated probability outputs
- **Active Learning:** Committee-based borderline case querying

---

## 📝 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🎓 Academic Report

For in-depth methodology, literature survey, and theoretical analysis:
> **[Report/CORDIS_Report.md](Report/CORDIS_Report.md)**
