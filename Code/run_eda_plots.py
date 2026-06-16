"""
run_eda_plots.py
================
CORDIS — Comprehensive Exploratory Data Analysis
================================================
Generates all EDA visualisations for the heart disease dataset.

Outputs saved to:  ../Images/  (relative to project root)

Figures produced:
    eda_00_dataset_summary.txt         — statistical summary (text)
    eda_01_target_distribution.png     — class balance countplot
    eda_02_age_distribution.png        — age histogram with KDE by target
    eda_03_sex_distribution.png        — sex × target countplot
    eda_04_chest_pain_distribution.png — chest pain type × target
    eda_05_numerical_histograms.png    — grid of numerical feature histograms
    eda_06_boxplots_by_target.png      — boxplots of continuous features vs target
    eda_07_correlation_heatmap.png     — full Pearson correlation heatmap
    eda_08_pairplot.png                — scatter pair plot (key features)
    eda_09_outlier_boxplots.png        — IQR outlier visualisation
    eda_10_class_balance.png           — class ratio pie/bar
    eda_11_categorical_analysis.png    — all categorical features vs target
    eda_12_distribution_by_target.png  — KDE plots continuous features vs target

Usage:
    cd ..   (project root)
    python Code/run_eda_plots.py
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

DATA_PATH  = os.path.join(_PROJECT_ROOT, 'Datasets', 'heart_dataset.csv')
IMAGES_DIR = os.path.join(_PROJECT_ROOT, 'Images')
os.makedirs(IMAGES_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Column rename map (mirrors preprocessing.py)
# ---------------------------------------------------------------------------
COLUMN_RENAME_MAP = {
    'Age'                                  : 'age',
    'Sex'                                  : 'sex',
    'Chest Pain Type'                      : 'cp',
    'Resting Blood Pressure'               : 'trestbps',
    'Cholestrol'                           : 'chol',
    'Fasting Blood Sugar'                  : 'fbs',
    'Rest ECG'                             : 'restecg',
    'Max Heart Rate'                       : 'thalach',
    'Exercise Angina'                      : 'exang',
    'Old Peak'                             : 'oldpeak',
    'Slope of Peak'                        : 'slope',
    'Vessles by Fluoroscopy'               : 'ca',
    'Thalassemia'                          : 'thal',
    'Target (Presence of Heart Disease)'   : 'target',
}

# Feature sets
NUMERIC_COLS      = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_COLS  = ['cp', 'restecg', 'slope', 'thal', 'ca']
BINARY_COLS       = ['sex', 'fbs', 'exang']

# Colour palette
COLOURS = {0: '#4E8DF5', 1: '#F54E4E'}
PALETTE = ['#4E8DF5', '#F54E4E']

# Global style
sns.set_theme(style='whitegrid', font_scale=1.05)
plt.rcParams.update({
    'figure.dpi'   : 120,
    'figure.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
})


# ---------------------------------------------------------------------------
# Load & rename
# ---------------------------------------------------------------------------
def load_and_rename(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        sys.exit(f"[EDA] ERROR — Dataset not found: {path}")
    df = pd.read_csv(path)
    rename = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df.columns}
    df = df.rename(columns=rename)
    return df


def save(fig, name: str) -> None:
    path = os.path.join(IMAGES_DIR, name)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  [EDA] Saved: {name}")


# ===========================================================================
# EDA 00 — Text summary
# ===========================================================================
def eda_00_summary(df: pd.DataFrame) -> None:
    txt_path = os.path.join(IMAGES_DIR, 'eda_00_dataset_summary.txt')
    raw_rows = pd.read_csv(DATA_PATH).shape[0]
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("  CORDIS - DATASET STATISTICAL SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Raw rows (with duplicates) : {raw_rows}\n")
        f.write(f"Unique rows (after dedup)  : {df.shape[0]}\n")
        f.write(f"Columns                    : {df.shape[1]}\n")
        f.write(f"Missing values             : {df.isnull().sum().sum()}\n")
        f.write(f"Duplicate rows             : {df.duplicated().sum()}\n\n")
        f.write("-" * 60 + "\n")
        f.write("Descriptive Statistics (Numerical Features)\n")
        f.write("-" * 60 + "\n")
        f.write(df[NUMERIC_COLS].describe().to_string())
        f.write("\n\n")
        f.write("-" * 60 + "\n")
        f.write("Target Class Distribution\n")
        f.write("-" * 60 + "\n")
        vc = df['target'].value_counts()
        for val, count in vc.items():
            label = "Disease Present (1)" if val == 1 else "No Disease  (0)"
            f.write(f"  {label:25} : {count:5d}  ({count/len(df)*100:.1f}%)\n")
        f.write("\n")
        f.write("-" * 60 + "\n")
        f.write("Missing Value Report (per column)\n")
        f.write("-" * 60 + "\n")
        miss = df.isnull().sum()
        for col, n in miss.items():
            f.write(f"  {col:<20} : {n}\n")
    print(f"  [EDA] Saved: eda_00_dataset_summary.txt")


# ===========================================================================
# EDA 01 — Target Distribution
# ===========================================================================
def eda_01_target_distribution(df: pd.DataFrame) -> None:
    counts = df['target'].value_counts().sort_index()
    labels = ['No Disease (0)', 'Disease (1)']
    colors = [COLOURS[0], COLOURS[1]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Bar chart
    bars = axes[0].bar(labels, counts.values, color=colors, edgecolor='white', width=0.5)
    for bar, val in zip(bars, counts.values):
        axes[0].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 2, f'{val}\n({val/len(df)*100:.1f}%)',
                     ha='center', va='bottom', fontweight='bold', fontsize=11)
    axes[0].set_title('Heart Disease Class Distribution', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Patient Count', fontsize=11)
    axes[0].set_xlabel('')
    axes[0].set_ylim(0, counts.max() * 1.18)
    axes[0].grid(axis='y', linestyle='--', alpha=0.5)

    # Pie chart
    wedges, texts, autotexts = axes[1].pie(
        counts.values, labels=labels, autopct='%1.1f%%',
        colors=colors, startangle=140,
        wedgeprops={'edgecolor': 'white', 'linewidth': 2},
        textprops={'fontsize': 11},
    )
    for at in autotexts:
        at.set_fontweight('bold')
    axes[1].set_title('Class Proportion', fontsize=13, fontweight='bold')

    fig.suptitle('Target Variable: Presence of Heart Disease',
                 fontsize=15, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'eda_01_target_distribution.png')


# ===========================================================================
# EDA 02 — Age Distribution
# ===========================================================================
def eda_02_age_distribution(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram + KDE overlaid by target
    for target_val, label in [(0, 'No Disease'), (1, 'Disease')]:
        subset = df[df['target'] == target_val]['age']
        axes[0].hist(subset, bins=20, alpha=0.55, color=COLOURS[target_val],
                     label=label, edgecolor='white')
    axes[0].set_title('Age Distribution by Heart Disease Status',
                       fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Age (years)', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].legend()

    # KDE
    for target_val, label in [(0, 'No Disease'), (1, 'Disease')]:
        subset = df[df['target'] == target_val]['age']
        axes[1].plot(*_kde(subset), color=COLOURS[target_val], lw=2.5, label=label)
        axes[1].fill_between(*_kde(subset), alpha=0.2, color=COLOURS[target_val])
    axes[1].set_title('Age Density by Heart Disease Status',
                       fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Age (years)', fontsize=11)
    axes[1].set_ylabel('Density', fontsize=11)
    axes[1].legend()

    fig.tight_layout()
    save(fig, 'eda_02_age_distribution.png')


def _kde(series: pd.Series, n_pts: int = 300):
    """Return (x, y) KDE arrays for plotting."""
    from scipy.stats import gaussian_kde
    kde  = gaussian_kde(series.dropna())
    x    = np.linspace(series.min(), series.max(), n_pts)
    return x, kde(x)


# ===========================================================================
# EDA 03 — Sex Distribution
# ===========================================================================
def eda_03_sex_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    sex_target = df.groupby(['sex', 'target']).size().unstack(fill_value=0)
    sex_target.index = ['Female (0)', 'Male (1)']
    sex_target.columns = ['No Disease', 'Disease']
    sex_target.plot(kind='bar', ax=ax, color=PALETTE, edgecolor='white', width=0.55)

    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge', fontweight='bold', fontsize=10)

    ax.set_title('Heart Disease Prevalence by Sex',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Biological Sex', fontsize=11)
    ax.set_ylabel('Patient Count', fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title='Diagnosis', fontsize=10)
    ax.set_ylim(0, sex_target.values.max() * 1.25)
    fig.tight_layout()
    save(fig, 'eda_03_sex_distribution.png')


# ===========================================================================
# EDA 04 — Chest Pain Type Distribution
# ===========================================================================
def eda_04_chest_pain(df: pd.DataFrame) -> None:
    cp_labels = {0: 'Asymptomatic', 1: 'Atypical\nAngina',
                 2: 'Non-Anginal', 3: 'Typical\nAngina'}
    df2 = df.copy()
    df2['cp_label'] = df2['cp'].map(cp_labels)
    df2['target_label'] = df2['target'].map({0: 'No Disease', 1: 'Disease'})

    fig, ax = plt.subplots(figsize=(10, 5))
    order = [cp_labels[i] for i in sorted(cp_labels)]
    sns.countplot(data=df2, x='cp_label', hue='target_label',
                  order=order, palette=PALETTE, ax=ax,
                  edgecolor='white')
    ax.set_title('Chest Pain Type vs Heart Disease Status',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Chest Pain Type', fontsize=11)
    ax.set_ylabel('Patient Count', fontsize=11)
    ax.legend(title='Diagnosis', fontsize=10)

    for container in ax.containers:
        ax.bar_label(container, fmt='%d', label_type='edge',
                     fontweight='bold', fontsize=9, padding=2)
    fig.tight_layout()
    save(fig, 'eda_04_chest_pain_distribution.png')


# ===========================================================================
# EDA 05 — Numerical Feature Histograms
# ===========================================================================
def eda_05_numerical_histograms(df: pd.DataFrame) -> None:
    titles = {
        'age'     : 'Age (years)',
        'trestbps': 'Resting Blood Pressure (mm Hg)',
        'chol'    : 'Serum Cholesterol (mg/dl)',
        'thalach' : 'Max Heart Rate (bpm)',
        'oldpeak' : 'ST Depression (oldpeak)',
    }
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    for i, col in enumerate(NUMERIC_COLS):
        ax = axes[i]
        ax.hist(df[col], bins=25, color='#4E8DF5', alpha=0.75, edgecolor='white')
        mean_val = df[col].mean()
        median_val = df[col].median()
        ax.axvline(mean_val, color='#F54E4E', lw=2, linestyle='--',
                   label=f'Mean: {mean_val:.1f}')
        ax.axvline(median_val, color='#4EF59A', lw=2, linestyle='-.',
                   label=f'Median: {median_val:.1f}')
        ax.set_title(titles.get(col, col), fontsize=11, fontweight='bold')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.legend(fontsize=9)

    axes[-1].axis('off')   # hide unused subplot
    fig.suptitle('Numerical Feature Distributions',
                 fontsize=15, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'eda_05_numerical_histograms.png')


# ===========================================================================
# EDA 06 — Boxplots by Target
# ===========================================================================
def eda_06_boxplots_by_target(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18, 6))
    target_labels = {0: 'No Disease', 1: 'Disease'}
    df2 = df.copy()
    df2['target_label'] = df2['target'].map(target_labels)

    for i, col in enumerate(NUMERIC_COLS):
        sns.boxplot(
            data=df2, x='target_label', y=col, ax=axes[i],
            palette=PALETTE, width=0.5, linewidth=1.5,
            flierprops=dict(marker='o', markersize=4, alpha=0.4),
        )
        axes[i].set_title(col, fontsize=12, fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel(col, fontsize=10)
        axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=15, fontsize=9)

    fig.suptitle('Continuous Features vs Heart Disease Status (Boxplots)',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'eda_06_boxplots_by_target.png')


# ===========================================================================
# EDA 07 — Correlation Heatmap
# ===========================================================================
def eda_07_correlation_heatmap(df: pd.DataFrame) -> None:
    corr = df.drop(columns=['target']).corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f',
        cmap='coolwarm', center=0, vmin=-1, vmax=1,
        square=True, linewidths=0.4,
        cbar_kws={'shrink': 0.75, 'label': 'Pearson r'},
        ax=ax,
        annot_kws={'size': 9},
    )
    ax.set_title('Correlation Heatmap — Heart Disease Features',
                 fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    save(fig, 'eda_07_correlation_heatmap.png')


# ===========================================================================
# EDA 08 — Pairplot (key features)
# ===========================================================================
def eda_08_pairplot(df: pd.DataFrame) -> None:
    key_cols = ['age', 'chol', 'thalach', 'oldpeak', 'target']
    df2 = df[key_cols].copy()
    df2['target'] = df2['target'].map({0: 'No Disease', 1: 'Disease'})

    g = sns.pairplot(
        df2, hue='target',
        palette={'No Disease': COLOURS[0], 'Disease': COLOURS[1]},
        diag_kind='kde', plot_kws={'alpha': 0.5, 's': 20},
        corner=True,
    )
    g.figure.suptitle('Pairplot — Key Clinical Features',
                       fontsize=13, fontweight='bold', y=1.02)
    save(g.figure, 'eda_08_pairplot.png')


# ===========================================================================
# EDA 09 — Outlier Visualisation
# ===========================================================================
def eda_09_outlier_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))

    for i, col in enumerate(NUMERIC_COLS):
        axes[i].boxplot(
            df[col].dropna(),
            vert=True, patch_artist=True,
            boxprops=dict(facecolor='#4E8DF5', color='navy', alpha=0.7),
            medianprops=dict(color='#F5C44E', linewidth=2.5),
            whiskerprops=dict(color='navy', linewidth=1.2),
            capprops=dict(color='navy', linewidth=1.5),
            flierprops=dict(marker='o', color='#F54E4E',
                            markersize=5, alpha=0.6, label='Outlier'),
        )
        q1  = df[col].quantile(0.25)
        q3  = df[col].quantile(0.75)
        iqr = q3 - q1
        n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
        axes[i].set_title(f'{col}\n({n_out} outlier(s))',
                          fontsize=11, fontweight='bold')
        axes[i].set_xlabel('')

    fig.suptitle('IQR Outlier Detection — Continuous Features',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'eda_09_outlier_boxplots.png')


# ===========================================================================
# EDA 10 — Class Balance Detailed
# ===========================================================================
def eda_10_class_balance(df: pd.DataFrame) -> None:
    # Already shown in EDA 01, this gives a more detailed breakdown
    counts = df['target'].value_counts().sort_index()
    total  = len(df)

    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ['No Disease\n(0)', 'Disease\n(1)']
    bars   = ax.bar(labels, counts.values, color=PALETTE,
                    edgecolor='white', width=0.45, zorder=3)

    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 1,
                f'{val}  ({val / total * 100:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=12)

    ax.axhline(total / 2, color='grey', linestyle='--',
               lw=1.2, label='50% balance line')
    ax.set_title('Heart Disease Class Balance',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_ylabel('Patient Count', fontsize=11)
    ax.set_ylim(0, counts.max() * 1.2)
    ax.grid(axis='y', linestyle='--', alpha=0.4, zorder=0)
    ax.legend(fontsize=10)

    fig.tight_layout()
    save(fig, 'eda_10_class_balance.png')


# ===========================================================================
# EDA 11 — Categorical Feature Analysis
# ===========================================================================
def eda_11_categorical_analysis(df: pd.DataFrame) -> None:
    cat_all = CATEGORICAL_COLS + BINARY_COLS
    n_cols  = 4
    n_rows  = -(-len(cat_all) // n_cols)   # ceiling division

    fig, axes = plt.subplots(n_rows, n_cols,
                              figsize=(5 * n_cols, 4.5 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cat_all):
        ax = axes[i]
        group = df.groupby([col, 'target']).size().unstack(fill_value=0)
        group.columns = ['No Disease', 'Disease']
        group.plot(kind='bar', ax=ax, color=PALETTE,
                   edgecolor='white', width=0.6)
        ax.set_title(col, fontsize=11, fontweight='bold')
        ax.set_xlabel('')
        ax.set_ylabel('Count', fontsize=9)
        ax.legend(fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        for container in ax.containers:
            ax.bar_label(container, fmt='%d', fontsize=8, padding=2)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle('Categorical Feature Distribution by Target',
                 fontsize=14, fontweight='bold', y=1.01)
    fig.tight_layout()
    save(fig, 'eda_11_categorical_analysis.png')


# ===========================================================================
# EDA 12 — KDE Distribution by Target
# ===========================================================================
def eda_12_distribution_by_target(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 5, figsize=(18, 5))

    for i, col in enumerate(NUMERIC_COLS):
        ax = axes[i]
        for target_val, label in [(0, 'No Disease'), (1, 'Disease')]:
            subset = df[df['target'] == target_val][col].dropna()
            x, y   = _kde(subset)
            ax.plot(x, y, color=COLOURS[target_val], lw=2.5, label=label)
            ax.fill_between(x, y, alpha=0.15, color=COLOURS[target_val])
        ax.set_title(col, fontsize=12, fontweight='bold')
        ax.set_xlabel(col, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=9)

    fig.suptitle('Feature Density Distributions by Heart Disease Status',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    save(fig, 'eda_12_distribution_by_target.png')


# ===========================================================================
# Main orchestration
# ===========================================================================

def main() -> None:
    print("=" * 60)
    print("  CORDIS EDA — Generating visualisations")
    print("=" * 60)

    df_raw  = pd.read_csv(DATA_PATH)
    rename  = {k: v for k, v in COLUMN_RENAME_MAP.items() if k in df_raw.columns}
    df_raw  = df_raw.rename(columns=rename)

    # Deduplicate for meaningful EDA
    df = df_raw.drop_duplicates().reset_index(drop=True)

    print(f"  Dataset (raw)  : {df_raw.shape[0]} rows × {df_raw.shape[1]} cols")
    print(f"  After dedup    : {df.shape[0]} rows")
    print(f"  Missing values : {df.isnull().sum().sum()}")
    print(f"  Saving plots to: {IMAGES_DIR}\n")

    eda_00_summary(df)
    eda_01_target_distribution(df)
    eda_02_age_distribution(df)
    eda_03_sex_distribution(df)
    eda_04_chest_pain(df)
    eda_05_numerical_histograms(df)
    eda_06_boxplots_by_target(df)
    eda_07_correlation_heatmap(df)
    eda_08_pairplot(df)
    eda_09_outlier_boxplots(df)
    eda_10_class_balance(df)
    eda_11_categorical_analysis(df)
    eda_12_distribution_by_target(df)

    print("\n" + "=" * 60)
    print(f"  [OK] All {13} EDA outputs saved to: {IMAGES_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
