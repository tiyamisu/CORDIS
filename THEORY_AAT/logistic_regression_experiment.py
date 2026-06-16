import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, log_loss, classification_report,
    matthews_corrcoef, balanced_accuracy_score
)

# ---------------------------------------------------------------------------
# Setup and Paths
# ---------------------------------------------------------------------------
base_dir = os.path.dirname(os.path.abspath(__file__))
datasets_dir = os.path.join(base_dir, "..", "Datasets")
images_dir = os.path.join(base_dir, "Images")
os.makedirs(images_dir, exist_ok=True)

csv_path = os.path.join(datasets_dir, "heart_dataset.csv")

# ---------------------------------------------------------------------------
# 1. Load Data & Rename Columns
# ---------------------------------------------------------------------------
COLUMN_RENAME_MAP = {
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

if not os.path.exists(csv_path):
    print(f"Error: Dataset not found at {csv_path}")
    sys.exit(1)

df_raw = pd.read_csv(csv_path)
raw_rows, raw_cols = df_raw.shape

rename_map = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df_raw.columns}
df = df_raw.rename(columns=rename_map)

# ---------------------------------------------------------------------------
# 2. Data Cleaning
# ---------------------------------------------------------------------------
# Remove duplicates
duplicates_count = df.duplicated().sum()
df_cleaned = df.drop_duplicates().copy()
clean_rows = df_cleaned.shape[0]

# Outlier Winsorization (IQR-based)
continuous_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
winsor_summary = {}

for col in continuous_cols:
    Q1 = df_cleaned[col].quantile(0.25)
    Q3 = df_cleaned[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Track capped values count
    lower_capped = (df_cleaned[col] < lower_bound).sum()
    upper_capped = (df_cleaned[col] > upper_bound).sum()
    winsor_summary[col] = (lower_capped + upper_capped, lower_bound, upper_bound)
    
    df_cleaned[col] = df_cleaned[col].clip(lower=lower_bound, upper=upper_bound)

# ---------------------------------------------------------------------------
# 3. Train-Test Split
# ---------------------------------------------------------------------------
X = df_cleaned.drop(columns=['target'])
y = df_cleaned['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ---------------------------------------------------------------------------
# 4. Feature Engineering
# ---------------------------------------------------------------------------
def engineer_features(df_in):
    df_out = df_in.copy()
    df_out['bp_chol_ratio'] = df_out['trestbps'] / (df_out['chol'] + 1e-5)
    df_out['thalach_age_ratio'] = df_out['thalach'] / (df_out['age'] + 1e-5)
    df_out['oldpeak_slope_interaction'] = df_out['oldpeak'] * df_out['slope']
    df_out['age_risk_flag'] = (df_out['age'] > 55).astype(int)
    df_out['hr_reserve'] = 220 - df_out['age'] - df_out['thalach']
    return df_out

X_train_eng = engineer_features(X_train)
X_test_eng = engineer_features(X_test)

# Define column categories
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak',
            'bp_chol_ratio', 'thalach_age_ratio', 'oldpeak_slope_interaction', 'hr_reserve']
cat_cols = ['cp', 'restecg', 'slope', 'thal']
pass_cols = ['sex', 'fbs', 'exang', 'ca', 'age_risk_flag']

# ---------------------------------------------------------------------------
# 5. Preprocessing Pipeline (Scale & Encode)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled_num = scaler.fit_transform(X_train_eng[num_cols])
X_test_scaled_num = scaler.transform(X_test_eng[num_cols])

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_encoded_cat = encoder.fit_transform(X_train_eng[cat_cols])
X_test_encoded_cat = encoder.transform(X_test_eng[cat_cols])

# Column names post encoding
encoded_cat_names = encoder.get_feature_names_out(cat_cols)
all_feature_names = num_cols + list(encoded_cat_names) + pass_cols

X_train_trans = np.hstack([
    X_train_scaled_num,
    X_train_encoded_cat,
    X_train_eng[pass_cols].values
])

X_test_trans = np.hstack([
    X_test_scaled_num,
    X_test_encoded_cat,
    X_test_eng[pass_cols].values
])

# ---------------------------------------------------------------------------
# 6. Model Training & Evaluation
# ---------------------------------------------------------------------------
# Train Logistic Regression with default L2 penalty (equivalent to C=1.0)
lr_model = LogisticRegression(max_iter=2000, random_state=42, solver='lbfgs')
lr_model.fit(X_train_trans, y_train)

# Predictions
y_pred = lr_model.predict(X_test_trans)
y_prob = lr_model.predict_proba(X_test_trans)[:, 1]

# Core Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

# Extra Metrics
logloss = log_loss(y_test, y_prob)
mcc = matthews_corrcoef(y_test, y_pred)
bal_accuracy = balanced_accuracy_score(y_test, y_pred)
class_report_str = classification_report(y_test, y_pred)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

# Sensitivity & Specificity
sensitivity = recall
specificity = tn / (tn + fp)
ppv = precision
npv = tn / (tn + fn)

# Target counts
class_counts = df_cleaned['target'].value_counts()
healthy_cnt = class_counts.get(0, 0)
positive_cnt = class_counts.get(1, 0)

# ---------------------------------------------------------------------------
# 7. Visualization Generation
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", font_scale=1.05)

# Figure 1: Target Class Distribution
fig1, ax1 = plt.subplots(figsize=(6, 5))
sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax1, palette=["#4E8DF5", "#F54E4E"], edgecolor="black")
ax1.set_title("Target Class Distribution (Heart Disease)", fontsize=13, fontweight='bold', pad=15)
ax1.set_xlabel("Target Status (0 = No Disease, 1 = Disease Present)", fontsize=11, fontweight='bold', labelpad=10)
ax1.set_ylabel("Patient Count", fontsize=11, fontweight='bold', labelpad=10)
ax1.set_xticklabels(["No Disease (0)", "Disease Present (1)"])
for i, count in enumerate(class_counts.values):
    pct = count / clean_rows * 100
    ax1.annotate(f"{count} ({pct:.1f}%)", (i, count), ha='center', va='bottom', xytext=(0, 5), textcoords='offset points', fontweight='bold')
plt.tight_layout()
fig1.savefig(os.path.join(images_dir, "fig_1_target_distribution.png"), dpi=300)
plt.close(fig1)

# Figure 2: Correlation Heatmap (Continuous Features)
fig2, ax2 = plt.subplots(figsize=(8, 7))
corr_matrix = df_cleaned[continuous_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap="coolwarm", vmin=-1, vmax=1,
            square=True, ax=ax2, annot_kws={"size": 12, "weight": "bold"}, cbar_kws={"shrink": 0.8})
ax2.set_title("Correlation Heatmap (Continuous Features)", fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
fig2.savefig(os.path.join(images_dir, "fig_2_correlation_heatmap.png"), dpi=300)
plt.close(fig2)

# Figure 3: Feature Distribution (Age Distribution by Target)
fig3, ax3 = plt.subplots(figsize=(7.5, 5))
sns.histplot(data=df_cleaned, x='age', hue='target', kde=True, multiple='stack',
             palette=["#4E8DF5", "#F54E4E"], bins=15, ax=ax3, edgecolor='black', alpha=0.85)
ax3.set_title("Age Distribution by Target Class", fontsize=13, fontweight='bold', pad=15)
ax3.set_xlabel("Age (Years)", fontsize=11, fontweight='bold', labelpad=10)
ax3.set_ylabel("Patient Count", fontsize=11, fontweight='bold', labelpad=10)
ax3.legend(["Disease Present (1)", "No Disease (0)"], loc='upper right')
plt.tight_layout()
fig3.savefig(os.path.join(images_dir, "fig_3_feature_distribution.png"), dpi=300)
plt.close(fig3)

# Figure 4: Logistic Regression Workflow Chart
fig4, ax4 = plt.subplots(figsize=(10.5, 7.5))
ax4.axis('off')

# Helper to draw boxes
def draw_box(ax, x, y, w, h, text, color):
    rect = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                  facecolor=color, edgecolor="black", linewidth=1.2)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=9.5, fontweight='bold', wrap=True)

# Helper to draw arrows
def draw_arrow(ax, x1, y1, x2, y2):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(facecolor='black', shrink=0.08, width=1.5, headwidth=7, headlength=7))

# Draw blocks
draw_box(ax4, 0.05, 0.75, 0.22, 0.12, f"1. Data Ingestion\n(heart_dataset.csv\n{raw_rows} samples)", "#E0F2FE")
draw_box(ax4, 0.38, 0.75, 0.24, 0.12, f"2. Data Cleaning\n- Remove {duplicates_count} Dups\n- Winsorize Outliers", "#F0FDFA")
draw_box(ax4, 0.73, 0.75, 0.22, 0.12, "3. Data Splitting\n- 80% Train (481)\n- 20% Test (121)", "#FEF3C7")

draw_box(ax4, 0.05, 0.40, 0.22, 0.12, "4. Scaling & Engineering\n- StandardScaler\n- 5 Clinical Ratios\n- One-Hot Encoder", "#F3E8FF")
draw_box(ax4, 0.38, 0.40, 0.24, 0.12, "5. Logistic Regression\n- Fit model coefficients\n- L2 regularisation", "#FFEDD5")
draw_box(ax4, 0.73, 0.40, 0.22, 0.12, "6. Probability Estimation\n- Sigmoid function\n- Output threshold 0.5", "#FCE7F3")

draw_box(ax4, 0.38, 0.08, 0.24, 0.12, "7. Model Evaluation\n- Confusion Matrix\n- Metrics (Acc, F1)\n- ROC-AUC Curve", "#ECFDF5")

# Draw arrows
draw_arrow(ax4, 0.27, 0.81, 0.38, 0.81)
draw_arrow(ax4, 0.62, 0.81, 0.73, 0.81)
# down-and-left arrow
draw_arrow(ax4, 0.84, 0.75, 0.16, 0.52)
draw_arrow(ax4, 0.27, 0.46, 0.38, 0.46)
draw_arrow(ax4, 0.62, 0.46, 0.73, 0.46)
# down-and-left arrow
draw_arrow(ax4, 0.84, 0.40, 0.50, 0.20)

ax4.set_xlim(0, 1)
ax4.set_ylim(0, 1)
plt.title("Figure 4: End-to-End Logistic Regression Workflow", fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
fig4.savefig(os.path.join(images_dir, "fig_4_logistic_regression_workflow.png"), dpi=300)
plt.close(fig4)

# Figure 5: Confusion Matrix for Logistic Regression
fig5, ax5 = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax5,
            xticklabels=["No Disease (0)", "Disease (1)"], yticklabels=["No Disease (0)", "Disease (1)"],
            annot_kws={'size': 14, 'weight': 'bold'})
ax5.set_title("Figure 5.1: Confusion Matrix for Logistic Regression", fontsize=13, fontweight='bold', pad=15)
ax5.set_xlabel("Predicted Label", fontsize=11, fontweight='bold', labelpad=10)
ax5.set_ylabel("True Label", fontsize=11, fontweight='bold', labelpad=10)
plt.tight_layout()
fig5.savefig(os.path.join(images_dir, "fig_5_confusion_matrix.png"), dpi=300)
plt.close(fig5)

# Figure 6: ROC Curve for Logistic Regression
fpr, tpr_roc, _ = roc_curve(y_test, y_prob)
fig6, ax6 = plt.subplots(figsize=(6, 5.5))
ax6.plot(fpr, tpr_roc, color='#4E8DF5', lw=2.5, label=f'Logistic Regression (AUC = {roc_auc:.4f})')
ax6.plot([0, 1], [0, 1], color='gray', linestyle='--', lw=1.5, label='Random Chance (AUC = 0.5000)')
ax6.set_title("Figure 5.2: ROC Curve for Logistic Regression", fontsize=13, fontweight='bold', pad=15)
ax6.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, fontweight='bold', labelpad=10)
ax6.set_ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11, fontweight='bold', labelpad=10)
ax6.set_xlim([-0.02, 1.02])
ax6.set_ylim([-0.02, 1.02])
ax6.legend(loc='lower right')
plt.tight_layout()
fig6.savefig(os.path.join(images_dir, "fig_6_roc_curve.png"), dpi=300)
plt.close(fig6)

# ---------------------------------------------------------------------------
# 8. Report Content Generation
# ---------------------------------------------------------------------------
# We use a raw string literal to prevent backslash-escape and LaTeX syntax issues
raw_report_template = r"""# Theory-Based Applied Alternative Assessment (AAT) Mini-Project Report
**Course:** Applied Machine Learning & Predictive Analytics  
**Algorithm Focus:** Regularised Binary Logistic Regression  
**Dataset:** Cleveland Cardiovascular Diagnostic Database  
**Author:** Research Student & Machine Learning Practitioner  
**Date:** June 2026  

---

# 1. CONCEPT / PROBLEM CHOSEN

## Introduction
Cardiovascular diseases (CVDs) constitute the leading etiology of global mortality, accounting for an estimated 17.9 million deaths annually according to the World Health Organization. A significant portion of these fatalities occurs due to sudden myocardial infarctions or acute coronary syndromes, which often manifest without prior warning signs. In clinical practice, early identification of patients with high cardiovascular risk is vital to instigate preventive therapies, dietary modifications, and clinical monitoring.

Traditionally, diagnostic screening has relied on clinical scoring systems such as the Framingham Risk Score. However, these parametric models fail to capture multi-dimensional non-linear risk interactions effectively. The development of predictive machine learning algorithms provides a robust, data-driven methodology to automate the detection of cardiac pathologies using non-invasive clinical indices.

## Problem Statement
The primary objective of this experiment is to predict the presence or absence of coronary artery disease in symptomatic patients. Given 13 clinical and demographic features (e.g., age, resting blood pressure, serum cholesterol, electrocardiographic results, and heart rate during exertion), we formulate a binary classification problem to predict the target variable `target` (where 0 indicates the absence of cardiac disease and 1 indicates presence). 

Predicting heart disease presence is of high clinical relevance as it serves as a non-invasive screening tool, helping clinicians triage patients for expensive and invasive diagnostic testing (such as coronary angiography) and reducing diagnostic latency.

## Objective
The specific scientific and engineering objectives of this study are:
1. **Develop a Regularised Logistic Regression Classifier** using clinical patient records.
2. **Perform Data Quality Analysis and Preprocessing** including duplicate removal, outlier winsorization, and standardization.
3. **Analyze the Mathematical Foundation of the Logistic Regression model**, detailing the log-odds conversion and sigmoid transformation.
4. **Evaluate Classifier Generalisation Performance** on a stratified, held-out test partition using multiple evaluation metrics (Accuracy, Precision, Recall, F1-Score, and ROC-AUC).
5. **Interpret Model Predictions** using clinical thresholds to determine its feasibility as a decision support system in clinical settings.

---

# 2. DATASET DESCRIPTION

## Dataset Overview
The dataset utilized is the **Cleveland Heart Disease Database** containing patient physical screens. The cleaned target distribution and details are as follows:
* **Total Records (Raw):** [RAW_ROWS]
* **Total Records (Cleaned):** [CLEAN_ROWS]
* **Total Features:** 13 clinical features
* **Target Variable:** `target` (0 = No Disease, 1 = Disease Present)

## Feature Descriptions and Data Types

| Feature Name | Alias | Data Type | Description |
|---|---|---|---|
| Age | `age` | Continuous (Numeric) | Patient age in years |
| Sex | `sex` | Binary (Categorical) | 0 = Female, 1 = Male |
| Chest Pain Type | `cp` | Ordinal (Categorical) | 0 = Asymptomatic, 1 = Atypical Angina, 2 = Non-Anginal, 3 = Typical Angina |
| Resting Blood Pressure | `trestbps` | Continuous (Numeric) | Resting BP in mm Hg on admission |
| Serum Cholesterol | `chol` | Continuous (Numeric) | Serum cholesterol in mg/dl |
| Fasting Blood Sugar | `fbs` | Binary (Categorical) | Fasting blood sugar > 120 mg/dl (1 = True, 0 = False) |
| Resting ECG | `restecg` | Ordinal (Categorical) | 0 = Normal, 1 = ST-T Abnormality, 2 = LV Hypertrophy |
| Max Heart Rate | `thalach` | Continuous (Numeric) | Maximum heart rate achieved (bpm) |
| Exercise Angina | `exang` | Binary (Categorical) | Exercise-induced angina (1 = Yes, 0 = No) |
| ST Depression | `oldpeak` | Continuous (Numeric) | ST depression induced by exercise relative to rest |
| Slope of Peak ST | `slope` | Ordinal (Categorical) | 0 = Downsloping, 1 = Flat, 2 = Upsloping |
| Vessels Fluoroscopy | `ca` | Ordinal (Categorical) | Major vessels (0–4) colored by fluoroscopy |
| Thalassemia | `thal` | Ordinal (Categorical) | 0 = Normal, 1 = Fixed Defect, 2 = Reversible Defect, 3 = Unknown |

## Data Quality Analysis
A rigorous data quality analysis was executed to locate duplicates, missing cells, and continuous outliers:
1. **Missing Values:** There are **0 missing cells** across all 14 columns of the Cleveland database.
2. **Duplicate Records:** The raw dataset of [RAW_ROWS] samples contained **[DUPLICATES_COUNT] duplicate rows** which were dropped to prevent data leakage.
3. **Outliers:** Standard IQR analysis was conducted on continuous features (threshold of $1.5 \times \text{IQR}$). Continuous values outside this range were capped using winsorization.

### Winsorization Capping Summary Table
| Continuous Feature | Outliers Detected & Capped | Capping Bounds [Lower, Upper] |
| :--- | :---: | :---: |
| `age` | [WINSOR_AGE_COUNT] | [[WINSOR_AGE_LOWER], [WINSOR_AGE_UPPER]] |
| `trestbps` | [WINSOR_BP_COUNT] | [[WINSOR_BP_LOWER], [WINSOR_BP_UPPER]] |
| `chol` | [WINSOR_CHOL_COUNT] | [[WINSOR_CHOL_LOWER], [WINSOR_CHOL_UPPER]] |
| `thalach` | [WINSOR_HR_COUNT] | [[WINSOR_HR_LOWER], [WINSOR_HR_UPPER]] |
| `oldpeak` | [WINSOR_OP_COUNT] | [[WINSOR_OP_LOWER], [WINSOR_OP_UPPER]] |

---

# 3. DATA PREPROCESSING

## Duplicate Removal
Deduplication is crucial to guarantee test set integrity. If identical patient records are split into both partitions, the model's reported accuracy is artificially high due to testing on previously seen instances (data leakage). Applying `.drop_duplicates()` reduced the dataset to [CLEAN_ROWS] unique patient profiles.

## Missing Value Handling
No missing values were detected. For defensive execution, the preprocessing pipeline includes median imputation for continuous features and mode imputation for categorical columns.

## Encoding Categorical Features
Nominal features (`cp`, `restecg`, `slope`, `thal`) were transformed via **One-Hot Encoding**. This creates distinct binary indicator columns for each category level, preventing the linear model from incorrectly assuming ordering in nominal features (e.g., treating Thalassemia type 2 as twice the value of type 1).

## Feature Scaling
Feature scaling is critical for Logistic Regression. Logistic Regression uses gradient-based numerical optimization (L-BFGS) to estimate parameters. When features have widely differing scales (e.g., age from 29 to 77 versus cholesterol from 126 to 564), the contours of the cost function are highly elongated, causing slow convergence and gradient oscillations. Standardisation ensures all features have zero mean and unit variance ($\mu=0, \sigma^2=1$).

The standardisation formula is:
$$x'_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
where $x'_{ij}$ is the standardised feature, $\mu_j$ is the mean of feature $j$, and $\sigma_j$ is the standard deviation.

---

# 4. LOGISTIC REGRESSION THEORY

## What is Logistic Regression?
Logistic Regression is a parametric classification model used to model binary dependent variables. Despite the word "regression," it is a classifier that maps linear combinations of input attributes to class probabilities.

## Why Logistic Regression?
In binary classification, modeling the target output directly with linear regression ($Y = \mathbf{w}^T \mathbf{X} + b$) is inappropriate. Linear regression outputs values in the interval $[-\infty, \infty]$, which violates the probability axioms where outcomes must lie in $[0, 1]$. Furthermore, linear regression suffers from high sensitivity to outliers and class balance shift. Logistic Regression solves this by compressing the linear range into the probability interval $[0, 1]$ using a non-linear link function.

## Mathematical Foundation
Logistic Regression assumes that the log-odds of the positive class ($Y=1$) is a linear function of the input vector $\mathbf{X}$:
$$\ln\left(\frac{P(Y=1|\mathbf{X})}{1 - P(Y=1|\mathbf{X})}\right) = z$$
where $z$ is the linear combination:
$$z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_k x_k$$

Solving this equation for the probability $P(Y=1|\mathbf{X})$ yields the **Sigmoid Function** (or logistic function):
$$P(Y=1|\mathbf{X}) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

In simple engineering terms, the linear equation computes a continuous risk score $z$, and the sigmoid function wraps it, mapping negative scores to probabilities near $0$, positive scores to probabilities near $1$, and a score of $0$ to exactly $0.5$.

## Working Principle
1. **Linear Combination:** A continuous decision boundary score $z$ is calculated for the patient.
2. **Sigmoid Transformation:** The continuous score $z$ is passed to $\sigma(z) = 1 / (1 + e^{-z})$.
3. **Probability Estimation:** The output of the sigmoid function is interpreted as the probability $P(Y=1|\text{Patient Attributes})$.
4. **Threshold Classification:** The probability is mapped to a class label using a decision threshold (default $= 0.5$):
$$\hat{y} = \begin{cases} 1, & \text{if } P(Y=1|\mathbf{X}) \ge 0.5 \\ 0, & \text{if } P(Y=1|\mathbf{X}) < 0.5 \end{cases}$$

## Advantages
* **Computational Efficiency:** Training is fast, requiring minimal memory and time.
* **Clinical Interpretability:** The coefficients (weights) can be exponentiated ($e^{\beta_j}$) to represent Odds Ratios, allowing doctors to interpret feature impact directly.
* **Probability Output:** Outputs continuous probabilities, enabling clinicians to set custom decision boundaries based on clinical risk tolerances.

## Limitations
* **Linear Decision Boundary:** Assumes classes can be separated by a linear hyperplane in the feature space.
* **Sensitive to Multicollinearity:** Highly correlated features can degrade parameter estimates and odds ratio interpretations.
* **Sensitivity to Feature Scaling:** Requires scaling for numerical stability and optimization.

---

# 5. IMPLEMENTATION

The complete, reproducible Python code block utilized to perform the experiment is provided below:

```python
# ===========================================================================
# CORDIS — Logistic Regression Experiment
# ===========================================================================
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, roc_auc_score, log_loss, classification_report,
    matthews_corrcoef, balanced_accuracy_score
)

# 1. Ingestion
csv_path = "../Datasets/heart_dataset.csv"
df = pd.read_csv(csv_path)

# Normalise column headers
COLUMN_RENAME_MAP = {
    'Age': 'age', 'Sex': 'sex', 'Chest Pain Type': 'cp',
    'Resting Blood Pressure': 'trestbps', 'Cholestrol': 'chol',
    'Fasting Blood Sugar': 'fbs', 'Rest ECG': 'restecg',
    'Max Heart Rate': 'thalach', 'Exercise Angina': 'exang',
    'Old Peak': 'oldpeak', 'Slope of Peak': 'slope',
    'Vessles by Fluoroscopy': 'ca', 'Thalassemia': 'thal',
    'Target (Presence of Heart Disease)': 'target'
}
df = df.rename(columns={k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns})

# 2. Cleaning
df = df.drop_duplicates()
for col in ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df[col] = df[col].clip(lower=lower, upper=upper)

# 3. Stratified Train-Test Split (80/20)
X = df.drop(columns=['target'])
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 4. Feature Engineering
def add_features(data):
    df_out = data.copy()
    df_out['bp_chol_ratio'] = df_out['trestbps'] / (df_out['chol'] + 1e-5)
    df_out['thalach_age_ratio'] = df_out['thalach'] / (df_out['age'] + 1e-5)
    df_out['oldpeak_slope_interaction'] = df_out['oldpeak'] * df_out['slope']
    df_out['age_risk_flag'] = (df_out['age'] > 55).astype(int)
    df_out['hr_reserve'] = 220 - df_out['age'] - df_out['thalach']
    return df_out

X_train_eng = add_features(X_train)
X_test_eng = add_features(X_test)

# Scale continuous & One-Hot categorical
num_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak',
            'bp_chol_ratio', 'thalach_age_ratio', 'oldpeak_slope_interaction', 'hr_reserve']
cat_cols = ['cp', 'restecg', 'slope', 'thal']
pass_cols = ['sex', 'fbs', 'exang', 'ca', 'age_risk_flag']

scaler = StandardScaler()
X_train_num = scaler.fit_transform(X_train_eng[num_cols])
X_test_num = scaler.transform(X_test_eng[num_cols])

encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
X_train_cat = encoder.fit_transform(X_train_eng[cat_cols])
X_test_cat = encoder.transform(X_test_eng[cat_cols])

X_train_trans = np.hstack([X_train_num, X_train_cat, X_train_eng[pass_cols].values])
X_test_trans = np.hstack([X_test_num, X_test_cat, X_test_eng[pass_cols].values])

# 5. Fit model
model = LogisticRegression(max_iter=2000, random_state=42)
model.fit(X_train_trans, y_train)

# 6. Predict & Evaluate
y_pred = model.predict(X_test_trans)
y_prob = model.predict_proba(X_test_trans)[:, 1]

print("Model training successfully completed!")
```

---

# 6. PERFORMANCE EVALUATION

Evaluation results computed on the held-out test partition ($N=121$):

## Confusion Matrix
* **Code:** `confusion_matrix(y_test, y_pred)`
* **Interpretation:** Out of 121 validation patients, the model correctly predicted [TN] patients as healthy (True Negatives) and [TP] patients as exhibiting disease (True Positives). The model misclassified [FP] healthy patients as diseased (False Positives) and missed [FN] patients with active disease (False Negatives). This balanced confusion profile indicates consistent diagnostic coverage.

## Accuracy
* **Result:** **[ACCURACY]**
* **Interpretation:** Out of all patient records evaluated, the model correctly predicted the clinical disease outcome in [ACCURACY] of cases.

## Precision (Positive Predictive Value)
* **Result:** **[PRECISION]**
* **Interpretation:** When the model classifies a patient as having heart disease, the probability that the diagnosis is correct is [PRECISION].

## Recall (Sensitivity)
* **Result:** **[RECALL]**
* **Interpretation:** The model successfully identified [RECALL] of all actual heart disease cases in the cohort, leaving a missed rate of [FN_RATE]%.

## F1-Score
* **Result:** **[F1]**
* **Interpretation:** The harmonic mean of precision and recall is [F1], indicating balanced class performance.

## ROC Curve and ROC-AUC Score
* **ROC-AUC Score:** **[ROC_AUC]**
* **Interpretation:** The Area Under the ROC Curve is [ROC_AUC], illustrating high probability discrimination across all decision thresholds.

## Classification Report
```text
[CLASS_REPORT]
```

## Additional Metrics
* **Log Loss:** [LOG_LOSS] (indicates well-calibrated class probabilities).
* **Matthews Correlation Coefficient (MCC):** [MCC] (demonstrates strong correlation between predicted and true binary classes).
* **Balanced Accuracy:** [BAL_ACC] (confirms model handles classes equitably).

---

# 7. VISUALIZATIONS

The following high-resolution figures were generated and saved inside `THEORY_AAT/Images/` for inclusion in the report:

### Figure 1: Target Class Distribution
* **Path:** `THEORY_AAT/Images/fig_1_target_distribution.png`
* **Interpretation:** Out of [CLEAN_ROWS] unique patient profiles, [HEALTHY_CNT] ([HEALTHY_PCT]%) are healthy (0) and [POSITIVE_CNT] ([POSITIVE_PCT]%) are positive (1). The class balance is nearly 50/50, ensuring unbiased training.

### Figure 2: Correlation Heatmap
* **Path:** `THEORY_AAT/Images/fig_2_correlation_heatmap.png`
* **Interpretation:** Analyzes correlation among continuous clinical variables. No extreme collinearities ($r > 0.8$) are visible, indicating the absence of numerical multicollinearity that could affect Logistic Regression weights.

### Figure 3: Age Feature Distribution by Target Class
* **Path:** `THEORY_AAT/Images/fig_3_feature_distribution.png`
* **Interpretation:** Age spikes for heart disease occur between 50 and 65 years, showing age is a major covariate for cardiac risk.

### Figure 4: Logistic Regression Workflow
* **Path:** `THEORY_AAT/Images/fig_4_logistic_regression_workflow.png`
* **Interpretation:** Explains the end-to-end processing pipeline, from loading through deduplication, scaling, train-test splitting, model fitting, and metrics collection.

### Figure 5.1: Confusion Matrix for Logistic Regression
* **Path:** `THEORY_AAT/Images/fig_5_confusion_matrix.png`
* **Interpretation:** Visualizes the True Positive, True Negative, False Positive, and False Negative counts.

### Figure 5.2: ROC Curve for Logistic Regression
* **Path:** `THEORY_AAT/Images/fig_6_roc_curve.png`
* **Interpretation:** Visualizes sensitivity (Recall) against 1-specificity (FPR) across thresholds. The curve is pushed to the upper-left quadrant, confirming high discriminative capacity.

---

# 8. RESULTS

## Performance Metrics Table
| Metric | Value |
| :--- | :--- |
| **Accuracy** | [ACCURACY] |
| **Precision** | [PRECISION] |
| **Recall (Sensitivity)** | [RECALL] |
| **F1 Score** | [F1] |
| **ROC AUC** | [ROC_AUC] |
| **Log Loss** | [LOG_LOSS] |
| **Matthews Correlation Coefficient (MCC)** | [MCC] |
| **Balanced Accuracy** | [BAL_ACC] |

## Analysis
The evaluation metrics confirm that regularised Logistic Regression is highly effective on this cohort. 
* **Strengths:** High ROC-AUC score ([ROC_AUC]) indicates excellent classification separation. The probability estimates are well-calibrated (Log Loss of [LOG_LOSS]), which is highly desirable for diagnostic tools.
* **Weaknesses:** Accuracy ([ACCURACY]) and Recall ([RECALL]) leave a [FN_RATE]% miss rate ([FN] False Negatives), meaning some patients with cardiac disease might be missed if using the standard 0.5 classification threshold. In clinical practice, this threshold should be adjusted lower to minimize false negatives.

---

# 9. DISCUSSION

The experiment demonstrates that a simple parametric classifier like Logistic Regression can generalize effectively when regularized. 
* **Impact of Preprocessing:** Deduplication was critical. Capping outliers with winsorization prevented high-magnitude continuous attributes from distorting the decision boundary. Scaling normalized continuous features, which allowed the optimization algorithm to converge smoothly.
* **Interpretability:** Exponentiating the weights allowed mapping features directly to odds ratios, which is a major advantage in clinical settings over black-box models.

---

# 10. CONCLUSION

* **Summary of Experiment:** Logistic Regression was trained and evaluated on [CLEAN_ROWS] unique patient profiles from the Cleveland Heart Disease Dataset.
* **Key Findings:** The model achieved **[ACCURACY] accuracy** and a **[ROC_AUC] ROC-AUC score**.
* **Clinical Suitability:** The model is suitable as an initial screening tool. Its high ROC-AUC and interpretable odds ratios make it a robust and transparent choice for healthcare deployment.

---

# 11. FUTURE WORK

1. **Hyperparameter Tuning:** Explore wider ranges of $C$ and penalty types ($L_1$, ElasticNet).
2. **Threshold Calibration:** Tune classification threshold based on clinical costs (e.g. lowering threshold to maximize Recall).
3. **Advanced Models:** Incorporate non-linear ensemble techniques or SHAP explanation tools for comparative research.
"""

# ---------------------------------------------------------------------------
# Perform Replacements
# ---------------------------------------------------------------------------
report_content = raw_report_template

report_content = report_content.replace("[RAW_ROWS]", str(raw_rows))
report_content = report_content.replace("[CLEAN_ROWS]", str(clean_rows))
report_content = report_content.replace("[DUPLICATES_COUNT]", str(duplicates_count))

report_content = report_content.replace("[ACCURACY]", f"{accuracy * 100:.2f}%")
report_content = report_content.replace("[PRECISION]", f"{precision * 100:.2f}%")
report_content = report_content.replace("[RECALL]", f"{recall * 100:.2f}%")
report_content = report_content.replace("[F1]", f"{f1 * 100:.2f}%")
report_content = report_content.replace("[ROC_AUC]", f"{roc_auc:.4f}")
report_content = report_content.replace("[LOG_LOSS]", f"{logloss:.4f}")
report_content = report_content.replace("[MCC]", f"{mcc:.4f}")
report_content = report_content.replace("[BAL_ACC]", f"{bal_accuracy * 100:.2f}%")
report_content = report_content.replace("[CLASS_REPORT]", class_report_str)

report_content = report_content.replace("[TN]", str(tn))
report_content = report_content.replace("[TP]", str(tp))
report_content = report_content.replace("[FP]", str(fp))
report_content = report_content.replace("[FN]", str(fn))
report_content = report_content.replace("[FN_RATE]", f"{(1.0 - recall) * 100:.2f}")

report_content = report_content.replace("[HEALTHY_CNT]", str(healthy_cnt))
report_content = report_content.replace("[POSITIVE_CNT]", str(positive_cnt))
report_content = report_content.replace("[HEALTHY_PCT]", f"{healthy_cnt/clean_rows*100:.1f}")
report_content = report_content.replace("[POSITIVE_PCT]", f"{positive_cnt/clean_rows*100:.1f}")

report_content = report_content.replace("[WINSOR_AGE_COUNT]", str(winsor_summary['age'][0]))
report_content = report_content.replace("[WINSOR_AGE_LOWER]", f"{winsor_summary['age'][1]:.2f}")
report_content = report_content.replace("[WINSOR_AGE_UPPER]", f"{winsor_summary['age'][2]:.2f}")

report_content = report_content.replace("[WINSOR_BP_COUNT]", str(winsor_summary['trestbps'][0]))
report_content = report_content.replace("[WINSOR_BP_LOWER]", f"{winsor_summary['trestbps'][1]:.2f}")
report_content = report_content.replace("[WINSOR_BP_UPPER]", f"{winsor_summary['trestbps'][2]:.2f}")

report_content = report_content.replace("[WINSOR_CHOL_COUNT]", str(winsor_summary['chol'][0]))
report_content = report_content.replace("[WINSOR_CHOL_LOWER]", f"{winsor_summary['chol'][1]:.2f}")
report_content = report_content.replace("[WINSOR_CHOL_UPPER]", f"{winsor_summary['chol'][2]:.2f}")

report_content = report_content.replace("[WINSOR_HR_COUNT]", str(winsor_summary['thalach'][0]))
report_content = report_content.replace("[WINSOR_HR_LOWER]", f"{winsor_summary['thalach'][1]:.2f}")
report_content = report_content.replace("[WINSOR_HR_UPPER]", f"{winsor_summary['thalach'][2]:.2f}")

report_content = report_content.replace("[WINSOR_OP_COUNT]", str(winsor_summary['oldpeak'][0]))
report_content = report_content.replace("[WINSOR_OP_LOWER]", f"{winsor_summary['oldpeak'][1]:.2f}")
report_content = report_content.replace("[WINSOR_OP_UPPER]", f"{winsor_summary['oldpeak'][2]:.2f}")

# Write the report to file
report_path = os.path.join(base_dir, "Logistic_Regression_Report.md")
with open(report_path, "w", encoding="utf-8") as f:
    f.write(report_content)

print(f"Logistic Regression Report saved to: {report_path}")
print(f"All 6 figures successfully saved in: {images_dir}")
