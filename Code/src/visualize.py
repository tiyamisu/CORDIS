"""
visualize.py
============
Evaluation visualisation utilities for the CORDIS Heart Disease
Prediction project.

Generates and saves:
    - Confusion matrices (combined canvas + per-model PNGs)
    - ROC curves (all models overlaid)
    - Feature importance bar chart (tree-based models)
    - Model metric comparison bar chart

All figures are saved at 300 dpi to the supplied output directory.
"""

import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Consistent visual style
sns.set_theme(style='whitegrid', font_scale=1.05)
PALETTE = [
    '#4E8DF5', '#F54E4E', '#4EF59A', '#F5C44E', '#A44EF5'
]


# ---------------------------------------------------------------------------
# Confusion Matrices
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_dir: str,
) -> None:
    """
    Generate confusion matrix heatmaps for all models.

    Saves:
        confusion_matrices.png        — combined side-by-side canvas
        confusion_matrix_<name>.png   — individual model files

    Args:
        trained_models: Name → fitted estimator.
        X_test        : Transformed test features.
        y_test        : True binary labels.
        save_dir      : Directory to save figures.
    """
    os.makedirs(save_dir, exist_ok=True)
    n = len(trained_models)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5))
    if n == 1:
        axes = [axes]

    class_labels = ['No Disease', 'Disease']

    for idx, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)

        # Combined canvas
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            ax=axes[idx], cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels,
            annot_kws={'size': 14, 'weight': 'bold'},
        )
        axes[idx].set_title(name, fontsize=12, fontweight='bold', pad=10)
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)

        # Individual file
        fig_ind, ax_ind = plt.subplots(figsize=(5, 4.5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues', ax=ax_ind,
            cbar=False,
            xticklabels=class_labels,
            yticklabels=class_labels,
            annot_kws={'size': 16, 'weight': 'bold'},
        )
        ax_ind.set_title(f'{name} — Confusion Matrix',
                         fontsize=12, fontweight='bold', pad=10)
        ax_ind.set_ylabel('True Label', fontsize=10)
        ax_ind.set_xlabel('Predicted Label', fontsize=10)
        fig_ind.tight_layout()
        ind_path = os.path.join(save_dir, f'confusion_matrix_{name}.png')
        fig_ind.savefig(ind_path, dpi=300, bbox_inches='tight')
        plt.close(fig_ind)

    fig.suptitle('Confusion Matrices — All Models',
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    combined_path = os.path.join(save_dir, 'confusion_matrices.png')
    fig.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIZ] Confusion matrices saved to: {combined_path}")


# ---------------------------------------------------------------------------
# ROC Curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    trained_models: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    save_dir: str,
) -> None:
    """
    Overlay ROC curves for all trained models.

    Args:
        trained_models: Name → fitted estimator.
        X_test        : Transformed test features.
        y_test        : True binary labels.
        save_dir      : Directory to save the figure.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))

    for i, (name, model) in enumerate(trained_models.items()):
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, 'decision_function'):
            scores = model.decision_function(X_test)
            y_prob = 1.0 / (1.0 + np.exp(-scores))
        else:
            y_prob = model.predict(X_test).astype(float)

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        color = PALETTE[i % len(PALETTE)]

        ax.plot(fpr, tpr, lw=2.5, color=color,
                label=f'{name}  (AUC = {roc_auc:.3f})')

    # Random-guess diagonal
    ax.plot([0, 1], [0, 1], 'k--', lw=1.2, label='Random Guess (AUC = 0.500)')
    ax.fill_between([0, 1], [0, 1], alpha=0.03, color='grey')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 − Specificity)', fontsize=12)
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    ax.set_title('ROC Curves — Heart Disease Classifier Comparison',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=10, frameon=True, framealpha=0.9)
    ax.grid(True, linestyle='--', alpha=0.5)

    fig.tight_layout()
    save_path = os.path.join(save_dir, 'roc_curves.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIZ] ROC curves saved to: {save_path}")


# ---------------------------------------------------------------------------
# Feature Importances
# ---------------------------------------------------------------------------

def plot_feature_importances(
    trained_models: Dict[str, Any],
    feature_names: List[str],
    save_dir: str,
    top_n: int = 20,
) -> None:
    """
    Plot feature importances from the best tree-based model.

    Preference order: XGBoost > RandomForest > DecisionTree.

    Args:
        trained_models: Name → fitted estimator.
        feature_names : Ordered list of feature names from FeaturePipeline.
        save_dir      : Directory to save the figure.
        top_n         : Number of top features to display.
    """
    os.makedirs(save_dir, exist_ok=True)
    priority = ['XGBoost', 'RandomForest', 'DecisionTree']

    for model_name in priority:
        if model_name not in trained_models:
            continue

        model = trained_models[model_name]
        importances = model.feature_importances_
        top_n_actual = min(top_n, len(feature_names))

        indices = np.argsort(importances)[::-1][:top_n_actual]
        top_feats = [feature_names[i] for i in indices]
        top_vals  = importances[indices]

        fig, ax = plt.subplots(figsize=(10, max(6, top_n_actual * 0.35)))
        colors = plt.cm.viridis(np.linspace(0.2, 0.85, top_n_actual))
        bars = ax.barh(top_feats[::-1], top_vals[::-1], color=colors[::-1], edgecolor='white')

        for bar, val in zip(bars, top_vals[::-1]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                    f'{val:.4f}', va='center', fontsize=9)

        ax.set_xlabel('Feature Importance Score', fontsize=12)
        ax.set_title(f'Top {top_n_actual} Feature Importances ({model_name})',
                     fontsize=14, fontweight='bold', pad=12)
        ax.set_xlim(0, max(top_vals) * 1.2)
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        fig.tight_layout()

        save_path = os.path.join(save_dir, 'feature_importances.png')
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"[VIZ] Feature importances ({model_name}) saved to: {save_path}")
        return

    print("[VIZ] No tree-based model found for feature importance plotting.")


# ---------------------------------------------------------------------------
# Model Metric Comparison
# ---------------------------------------------------------------------------

def plot_model_comparison(report_df: pd.DataFrame, save_dir: str) -> None:
    """
    Side-by-side grouped bar chart comparing all model metrics.

    Args:
        report_df: DataFrame with columns [Model, Accuracy, Precision,
                   Recall, F1-Score, ROC-AUC].
        save_dir : Directory to save the figure.
    """
    os.makedirs(save_dir, exist_ok=True)

    metric_cols = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    df_melt = pd.melt(
        report_df,
        id_vars=['Model'],
        value_vars=metric_cols,
        var_name='Metric',
        value_name='Score',
    )

    fig, ax = plt.subplots(figsize=(13, 7))
    model_palette = {
        m: PALETTE[i % len(PALETTE)]
        for i, m in enumerate(report_df['Model'].unique())
    }

    sns.barplot(
        data=df_melt, x='Metric', y='Score',
        hue='Model', palette=model_palette, ax=ax,
        edgecolor='white', linewidth=0.8,
    )

    # Value annotations
    for patch in ax.patches:
        h = patch.get_height()
        if h > 0.01:
            ax.annotate(
                f'{h:.3f}',
                xy=(patch.get_x() + patch.get_width() / 2, h),
                xytext=(0, 4), textcoords='offset points',
                ha='center', va='bottom', fontsize=8, fontweight='bold',
            )

    ax.set_ylim(0, 1.20)
    ax.set_xlabel('Evaluation Metric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Metric Comparison — All Models',
                 fontsize=14, fontweight='bold', pad=15)
    ax.legend(
        title='Model', loc='upper right',
        frameon=True, framealpha=0.9, fontsize=9,
    )
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    fig.tight_layout()

    save_path = os.path.join(save_dir, 'model_comparison.png')
    fig.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[VIZ] Model comparison chart saved to: {save_path}")
