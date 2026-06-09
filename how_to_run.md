# CORDIS Execution Guide: How to Run the Project

This guide provides instructions to set up the environment and run the Cardiovascular Diagnostic Support System (CORDIS) pipeline.

---

## 1. Prerequisites and Environment Setup

This project requires **Python 3.8+** (tested up to Python 3.14).

### Step 1: Navigate to the Code Directory
Open your terminal (PowerShell, CMD, or bash) and navigate to the project's `Code` directory:
```powershell
cd "c:/Users/TIYASHA SARKAR/OneDrive/Desktop/PROJECTS/CORDIS/Code"
```

### Step 2: Create a Virtual Environment (Optional but Recommended)
Create a clean environment to install project dependencies:
```powershell
python -m venv venv
```

### Step 3: Activate the Virtual Environment
* **Windows (PowerShell):**
  ```powershell
  .\venv\Scripts\Activate.ps1
  ```
* **Windows (Command Prompt):**
  ```cmd
  .\venv\Scripts\activate.bat
  ```
* **macOS/Linux (Bash/zsh):**
  ```bash
  source venv/bin/activate
  ```

### Step 4: Install Dependencies
Install all required packages:
```powershell
pip install -r requirements.txt
```
*(Optional: If running the literature parser or Excel reader, ensure `openpyxl` is installed via `pip install openpyxl`)*

---

## 2. Directory Structure Overview
Your workspace is organized as follows:
```text
CORDIS/
├── Datasets/
│   └── heart.csv                 # Raw clinical dataset
├── Images/                       # Generated evaluation & EDA figures
├── Papers/                       # Literature survey source PDFs
├── Report/
│   └── CORDIS_Report.md          # Completed academic project report
├── Literature_Survey.xlsx        # Excel data for literature review
├── Code/
│   ├── main.py                   # Main ML training & evaluation runner
│   ├── run_eda_plots.py          # Script generating initial EDA figures
│   ├── generate_comparison.py    # Script generating final tables & comparisons
│   ├── requirements.txt          # Python packages list
│   └── src/                      # Modulated pipeline logic
│       ├── preprocessing.py      # Cleansing, splitting, and scaling
│       ├── features.py           # Feature transformer pipelines
│       ├── train.py              # GridSearchCV models & hyperparameters
│       ├── evaluate.py           # Report metrics calculation
│       └── visualize.py          # Plots (confusion matrices, ROC curves, etc.)
└── how_to_run.md                 # This instruction file
```

---

## 3. Running the Scripts

### Step 1: Run Exploratory Data Analysis (EDA)
Generate demographic distribution, correlation heatmaps, and cardiovascular feature distributions:
```powershell
python run_eda_plots.py
```
* **Output:** Saves target, age, gender distributions, and feature correlations as PNGs inside the `Images/` folder.

### Step 2: Run the Machine Learning Pipeline (Training & Optimization)
Run the end-to-end classification pipeline including data cleaning, robust preprocessing, 5-fold cross-validated grid search tuning, and testing:
```powershell
python main.py
```
* **Output:** 
  * Prints best parameters, cross-validation scores, and classification reports to the console.
  * Saves quantitative scores to `Code/outputs/model_metrics.csv`.
  * Generates and saves the ROC curve (`roc_curves.png`), Feature Importances (`feature_importances.png`), and individual/combined Confusion Matrices (`confusion_matrices.png`, `confusion_matrix_LogisticRegression.png`, etc.) in the `Images/` folder.

### Step 3: Generate Comparative Visualizations and Tables
Summarize all evaluation metrics and compile a comparative performance chart:
```powershell
python generate_comparison.py
```
* **Output:**
  * Prints a formatted Markdown table comparing Accuracy, Precision, Recall, and F1 Score for all four tuned classifiers.
  * Generates and saves a comparative bar chart (`accuracy_comparison.png`) under both the `Images/` folder and the root directory.

---

## 4. Viewing the Report
After running the scripts and generating all figures, you can review the full research write-up in:
* **Academic Report File:** [CORDIS_Report.md](file:///c:/Users/TIYASHA%20SARKAR/OneDrive/Desktop/PROJECTS/CORDIS/Report/CORDIS_Report.md)
