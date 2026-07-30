"""
Utilities Module
================
Plotting, reporting, SHAP analysis, and prediction helpers.
Authors: Sanman Kadam, Varsha Gupta
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def generate_evaluation_plots(trained_models, X_test, y_test, images_dir: str):
    """
    Generates and saves feature importance, confusion matrix, and performance plots.
    """
    os.makedirs(images_dir, exist_ok=True)
    sns.set_style('whitegrid')

    # 1. Feature Importance for Tree Models
    rf_model = trained_models.get('Random Forest')
    if rf_model is not None:
        imp = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=imp.head(15), palette='Blues_r')
        plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'feature_importance.png'), dpi=150)
        plt.close()

    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, len(trained_models), figsize=(4 * len(trained_models), 4))
    if len(trained_models) == 1:
        axes = [axes]

    for idx, (name, model) in enumerate(trained_models.items()):
        y_pred = model.predict(X_test)
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
        axes[idx].set_title(name, fontsize=11, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual' if idx == 0 else '')

    plt.suptitle('Confusion Matrix Comparison Across Models', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'confusion_matrices.png'), dpi=150)
    plt.close()

    print(f"Evaluation plots saved to {images_dir}/")


def run_shap_analysis(model, X_test, images_dir: str):
    """Generates SHAP summary plot if shap is installed."""
    if not SHAP_AVAILABLE:
        print("SHAP package not installed. Skipping SHAP analysis.")
        return

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.iloc[:500])

        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test.iloc[:500], show=False)
        plt.title('SHAP Feature Contribution Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'shap_summary.png'), dpi=150)
        plt.close()
        print(f"SHAP summary plot saved to {images_dir}/shap_summary.png")
    except Exception as e:
        print(f"SHAP analysis warning: {e}")
