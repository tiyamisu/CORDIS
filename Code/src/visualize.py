import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.inspection import permutation_importance
from typing import Dict, Any, List

def plot_confusion_matrices(trained_models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, save_dir: str):
    """
    Generate and save confusion matrices for all models in a single comparative canvas and as individual PNGs.
    """
    n_models = len(trained_models)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4.5))
    if n_models == 1:
        axes = [axes]
        
    for i, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        # Plot in combined canvas
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i], cbar=False,
                    xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
        axes[i].set_title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
        
        # Generate and save individual plot
        plt.figure(figsize=(5, 4.5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])
        plt.title(f'{name} Confusion Matrix', fontsize=12, fontweight='bold')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        indiv_path = os.path.join(save_dir, f'confusion_matrix_{name}.png')
        plt.savefig(indiv_path, dpi=300)
        plt.close()
        print(f"Saved individual confusion matrix: {indiv_path}")
        
    plt.figure(fig.number)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'confusion_matrices.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Combined confusion matrices plotted and saved to: {save_path}")

def plot_roc_curves(trained_models: Dict[str, Any], X_test: pd.DataFrame, y_test: pd.Series, save_dir: str):
    """
    Generate and save ROC curve comparison plot.
    """
    plt.figure(figsize=(8, 6))
    
    # Elegant, premium dark styling details
    sns.set_theme(style="whitegrid")
    
    for name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_prob = model.decision_function(X_test)
            y_prob = 1 / (1 + np.exp(-y_prob)) # Sigmoid normalization
        else:
            y_prob = model.predict(X_test).astype(float)
            
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', lw=2)
        
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Guess (AUC = 0.500)')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=11)
    plt.title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold', pad=15)
    plt.legend(loc="lower right", frameon=True)
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'roc_curves.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"ROC curves plotted and saved to: {save_path}")

def plot_feature_importances(trained_models: Dict[str, Any], feature_names: List[str], save_dir: str):
    """
    Extract and plot feature importances for tree-based estimators (Random Forest / XGBoost).
    """
    # Pick first tree-based estimator or calculate permutation importance for SVC/LR
    for name in ['RandomForest', 'DecisionTree']:
        if name in trained_models:
            model = trained_models[name]
            importances = model.feature_importances_
            
            # Sort importances
            indices = np.argsort(importances)[::-1]
            top_n = min(15, len(feature_names))
            
            sorted_features = [feature_names[i] for i in indices[:top_n]]
            sorted_importances = importances[indices[:top_n]]
            
            plt.figure(figsize=(10, 6))
            sns.barplot(x=sorted_importances, y=sorted_features, palette="viridis")
            plt.title(f'Top {top_n} Feature Importances ({name})', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel('Importance Coefficient', fontsize=11)
            plt.ylabel('Features', fontsize=11)
            plt.tight_layout()
            
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, 'feature_importances.png')
            plt.savefig(save_path, dpi=300)
            plt.close()
            print(f"Feature importances ({name}) plotted and saved to: {save_path}")
            return
            
    print("No tree-based model found to plot feature importances natively.")

def plot_model_comparison(report_df: pd.DataFrame, save_dir: str):
    """
    Plot model metrics side-by-side.
    """
    # Melt dataframe for easy seaborn plotting
    df_melted = pd.melt(report_df, id_vars=['Model'], var_name='Metric', value_name='Value')
    
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Custom harmonious color palette
    palette = sns.color_palette("muted", len(report_df['Model'].unique()))
    
    ax = sns.barplot(data=df_melted, x='Metric', y='Value', hue='Model', palette=palette)
    
    # Annotate bars with values
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            ax.annotate(f'{height:.2f}',
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='bottom',
                        xytext=(0, 3), textcoords='offset points', fontsize=9)
            
    plt.ylim([0, 1.15])
    plt.title('Performance Metrics Comparison across Models', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('Score Value', fontsize=11)
    plt.xlabel('Evaluation Metrics', fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    plt.tight_layout()
    
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'model_comparison.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Model comparison chart plotted and saved to: {save_path}")
