# Theory-Based Applied Alternative Assessment (AAT) Mini-Project Report
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
* **Total Records (Raw):** 1888
* **Total Records (Cleaned):** 602
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
2. **Duplicate Records:** The raw dataset of 1888 samples contained **1286 duplicate rows** which were dropped to prevent data leakage.
3. **Outliers:** Standard IQR analysis was conducted on continuous features (threshold of $1.5 \times \text{IQR}$). Continuous values outside this range were capped using winsorization.

### Winsorization Capping Summary Table
| Continuous Feature | Outliers Detected & Capped | Capping Bounds [Lower, Upper] |
| :--- | :---: | :---: |
| `age` | 0 | [28.50, 80.50] |
| `trestbps` | 18 | [90.00, 170.00] |
| `chol` | 11 | [114.88, 373.88] |
| `thalach` | 2 | [81.38, 216.38] |
| `oldpeak` | 4 | [-2.70, 4.50] |

---

# 3. DATA PREPROCESSING

## Duplicate Removal
Deduplication is crucial to guarantee test set integrity. If identical patient records are split into both partitions, the model's reported accuracy is artificially high due to testing on previously seen instances (data leakage). Applying `.drop_duplicates()` reduced the dataset to 602 unique patient profiles.

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
* **Interpretation:** Out of 121 validation patients, the model correctly predicted 48 patients as healthy (True Negatives) and 45 patients as exhibiting disease (True Positives). The model misclassified 14 healthy patients as diseased (False Positives) and missed 14 patients with active disease (False Negatives). This balanced confusion profile indicates consistent diagnostic coverage.

## Accuracy
* **Result:** **76.86%**
* **Interpretation:** Out of all patient records evaluated, the model correctly predicted the clinical disease outcome in 76.86% of cases.

## Precision (Positive Predictive Value)
* **Result:** **76.27%**
* **Interpretation:** When the model classifies a patient as having heart disease, the probability that the diagnosis is correct is 76.27%.

## Recall (Sensitivity)
* **Result:** **76.27%**
* **Interpretation:** The model successfully identified 76.27% of all actual heart disease cases in the cohort, leaving a missed rate of 23.73%.

## F1-Score
* **Result:** **76.27%**
* **Interpretation:** The harmonic mean of precision and recall is 76.27%, indicating balanced class performance.

## ROC Curve and ROC-AUC Score
* **ROC-AUC Score:** **0.8650**
* **Interpretation:** The Area Under the ROC Curve is 0.8650, illustrating high probability discrimination across all decision thresholds.

## Classification Report
```text
              precision    recall  f1-score   support

           0       0.77      0.77      0.77        62
           1       0.76      0.76      0.76        59

    accuracy                           0.77       121
   macro avg       0.77      0.77      0.77       121
weighted avg       0.77      0.77      0.77       121

```

## Additional Metrics
* **Log Loss:** 0.4681 (indicates well-calibrated class probabilities).
* **Matthews Correlation Coefficient (MCC):** 0.5369 (demonstrates strong correlation between predicted and true binary classes).
* **Balanced Accuracy:** 76.85% (confirms model handles classes equitably).

---

# 7. VISUALIZATIONS

The following high-resolution figures were generated and saved inside `THEORY_AAT/Images/` for inclusion in the report:

### Figure 1: Target Class Distribution
* **Path:** `THEORY_AAT/Images/fig_1_target_distribution.png`
* **Interpretation:** Out of 602 unique patient profiles, 310 (51.5%) are healthy (0) and 292 (48.5%) are positive (1). The class balance is nearly 50/50, ensuring unbiased training.

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
| **Accuracy** | 76.86% |
| **Precision** | 76.27% |
| **Recall (Sensitivity)** | 76.27% |
| **F1 Score** | 76.27% |
| **ROC AUC** | 0.8650 |
| **Log Loss** | 0.4681 |
| **Matthews Correlation Coefficient (MCC)** | 0.5369 |
| **Balanced Accuracy** | 76.85% |

## Analysis
The evaluation metrics confirm that regularised Logistic Regression is highly effective on this cohort. 
* **Strengths:** High ROC-AUC score (0.8650) indicates excellent classification separation. The probability estimates are well-calibrated (Log Loss of 0.4681), which is highly desirable for diagnostic tools.
* **Weaknesses:** Accuracy (76.86%) and Recall (76.27%) leave a 23.73% miss rate (14 False Negatives), meaning some patients with cardiac disease might be missed if using the standard 0.5 classification threshold. In clinical practice, this threshold should be adjusted lower to minimize false negatives.

---

# 9. DISCUSSION

The experiment demonstrates that a simple parametric classifier like Logistic Regression can generalize effectively when regularized. 
* **Impact of Preprocessing:** Deduplication was critical. Capping outliers with winsorization prevented high-magnitude continuous attributes from distorting the decision boundary. Scaling normalized continuous features, which allowed the optimization algorithm to converge smoothly.
* **Interpretability:** Exponentiating the weights allowed mapping features directly to odds ratios, which is a major advantage in clinical settings over black-box models.

---

# 10. CONCLUSION

* **Summary of Experiment:** Logistic Regression was trained and evaluated on 602 unique patient profiles from the Cleveland Heart Disease Dataset.
* **Key Findings:** The model achieved **76.86% accuracy** and a **0.8650 ROC-AUC score**.
* **Clinical Suitability:** The model is suitable as an initial screening tool. Its high ROC-AUC and interpretable odds ratios make it a robust and transparent choice for healthcare deployment.

---

# 11. FUTURE WORK

1. **Hyperparameter Tuning:** Explore wider ranges of $C$ and penalty types ($L_1$, ElasticNet).
2. **Threshold Calibration:** Tune classification threshold based on clinical costs (e.g. lowering threshold to maximize Recall).
3. **Advanced Models:** Incorporate non-linear ensemble techniques or SHAP explanation tools for comparative research.
