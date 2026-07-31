"""
Utilities Module
================
Plotting, reporting, SHAP analysis, calibration visualization,
threshold curves, radar charts, and prediction helpers.

Authors: Sanman Kadam, Varsha Gupta
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve

from src.evaluate import compute_threshold_curve
from src.config import PLOT_DPI, PLOT_STYLE, TOP_N_FEATURES, SHAP_SAMPLE_SIZE
from src.logging_config import get_logger

logger = get_logger('utils')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


def generate_evaluation_plots(trained_models, X_test, y_test, images_dir: str):
    """
    Generates and saves feature importance, confusion matrix, ROC, and PR plots.
    """
    os.makedirs(images_dir, exist_ok=True)
    sns.set_style(PLOT_STYLE)

    # 1. Feature Importance for Tree Models
    rf_model = trained_models.get('Random Forest')
    if rf_model is not None:
        imp = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=imp.head(TOP_N_FEATURES), palette='Blues_r')
        plt.title('Top 15 Feature Importances (Random Forest)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'feature_importance.png'), dpi=PLOT_DPI)
        plt.close()

    # 2. Confusion Matrices
    fig, axes = plt.subplots(1, min(len(trained_models), 5), figsize=(4 * min(len(trained_models), 5), 4))
    if min(len(trained_models), 5) == 1:
        axes = [axes]

    for idx, (name, model) in enumerate(list(trained_models.items())[:5]):
        y_pred = model.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
        axes[idx].set_title(name, fontsize=9, fontweight='bold')
        axes[idx].set_xlabel('Predicted')
        axes[idx].set_ylabel('Actual' if idx == 0 else '')

    plt.suptitle('Confusion Matrix Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'confusion_matrices.png'), dpi=PLOT_DPI)
    plt.close()

    # 3. ROC Curves
    plt.figure(figsize=(10, 7))
    for name, model in trained_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        from sklearn.metrics import roc_auc_score
        auc_val = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.4f})', linewidth=1.5)

    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Baseline')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — All Models', fontsize=14, fontweight='bold')
    plt.legend(fontsize=8, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'roc_curves.png'), dpi=PLOT_DPI)
    plt.close()

    # 4. Precision-Recall Curves
    plt.figure(figsize=(10, 7))
    for name, model in trained_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        from sklearn.metrics import average_precision_score
        ap = average_precision_score(y_test, y_proba)
        plt.plot(rec, prec, label=f'{name} (AP={ap:.4f})', linewidth=1.5)

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curves — All Models', fontsize=14, fontweight='bold')
    plt.legend(fontsize=8, loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'precision_recall_curves.png'), dpi=PLOT_DPI)
    plt.close()

    logger.info(f"Evaluation plots saved to {images_dir}/")


def generate_threshold_optimization_plot(trained_models, X_test, y_test, images_dir: str):
    """
    Generates threshold optimization curves for top 3 models showing
    how Precision, Recall, F1, F2 vary with classification threshold.
    """
    os.makedirs(images_dir, exist_ok=True)
    sns.set_style(PLOT_STYLE)

    # Pick top 3 by name presence
    model_names = ['Random Forest', 'XGBoost', 'LightGBM']
    available = [n for n in model_names if n in trained_models]
    if not available:
        available = list(trained_models.keys())[:3]

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5))
    if len(available) == 1:
        axes = [axes]

    for idx, name in enumerate(available):
        model = trained_models[name]
        y_proba = model.predict_proba(X_test)[:, 1]
        curve_df = compute_threshold_curve(y_test, y_proba)

        axes[idx].plot(curve_df['Threshold'], curve_df['Precision'], label='Precision', color='#3b82f6')
        axes[idx].plot(curve_df['Threshold'], curve_df['Recall'], label='Recall', color='#ef4444')
        axes[idx].plot(curve_df['Threshold'], curve_df['F1-Score'], label='F1', color='#10b981', linewidth=2)
        axes[idx].plot(curve_df['Threshold'], curve_df['F2-Score'], label='F2 (β=2)', color='#f59e0b', linewidth=2, linestyle='--')

        # Mark optimal F1 threshold
        best_idx = curve_df['F1-Score'].idxmax()
        best_thresh = curve_df.loc[best_idx, 'Threshold']
        best_f1 = curve_df.loc[best_idx, 'F1-Score']
        axes[idx].axvline(x=best_thresh, color='gray', linestyle=':', alpha=0.7)
        axes[idx].scatter([best_thresh], [best_f1], color='#10b981', s=80, zorder=5)
        axes[idx].annotate(f'τ*={best_thresh:.2f}', (best_thresh, best_f1),
                          textcoords="offset points", xytext=(10, 10), fontsize=9)

        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Threshold (τ)')
        axes[idx].set_ylabel('Score')
        axes[idx].legend(fontsize=8)
        axes[idx].set_ylim(0, 1.05)

    plt.suptitle('Threshold Optimization — Precision, Recall, F1 & F2 vs. Threshold',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'threshold_optimization.png'), dpi=PLOT_DPI)
    plt.close()
    logger.info(f"Threshold optimization plot saved to {images_dir}/threshold_optimization.png")


def generate_calibration_plot(calibration_data, images_dir: str):
    """
    Generates calibration curves comparing raw vs calibrated probabilities.
    """
    if not calibration_data:
        logger.info("No calibration data available. Skipping calibration plot.")
        return

    os.makedirs(images_dir, exist_ok=True)
    sns.set_style(PLOT_STYLE)

    fig, axes = plt.subplots(1, len(calibration_data), figsize=(6 * len(calibration_data), 5))
    if len(calibration_data) == 1:
        axes = [axes]

    for idx, (name, data) in enumerate(calibration_data.items()):
        axes[idx].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfectly Calibrated')
        axes[idx].plot(data['prob_pred_raw'], data['prob_true_raw'],
                      'o-', label='Raw', color='#ef4444', markersize=5)
        axes[idx].plot(data['prob_pred_cal'], data['prob_true_cal'],
                      's-', label='Calibrated', color='#10b981', markersize=5)
        axes[idx].set_title(name, fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('Mean Predicted Probability')
        axes[idx].set_ylabel('Fraction of Positives')
        axes[idx].legend(fontsize=9)

    plt.suptitle('Probability Calibration Curves (Raw vs. Isotonic Calibrated)',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'calibration_curves.png'), dpi=PLOT_DPI)
    plt.close()
    logger.info(f"Calibration curve plot saved to {images_dir}/calibration_curves.png")


def generate_radar_chart(results_df, images_dir: str):
    """
    Generates a radar/spider chart comparing all models across key metrics.
    """
    os.makedirs(images_dir, exist_ok=True)
    sns.set_style(PLOT_STYLE)

    # Metrics for radar chart
    radar_metrics = ['Test_PR_AUC', 'Test_ROC_AUC', 'Test_Recall_Default',
                     'Test_Precision_Default', 'Test_MCC', 'Test_F1_Default']
    radar_labels = ['PR-AUC', 'ROC-AUC', 'Recall', 'Precision', 'MCC', 'F1']

    # Filter to models that have these columns
    plot_df = results_df[['Model'] + radar_metrics].dropna()
    if plot_df.empty:
        return

    # Normalize MCC to 0-1 range for radar (MCC can be -1 to 1)
    if 'Test_MCC' in plot_df.columns:
        plot_df = plot_df.copy()
        plot_df['Test_MCC'] = (plot_df['Test_MCC'] + 1) / 2

    n_metrics = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # Close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(plot_df)))

    for idx, (_, row) in enumerate(plot_df.iterrows()):
        values = [row[m] for m in radar_metrics]
        values += values[:1]
        ax.plot(angles, values, 'o-', linewidth=1.5, label=row['Model'], color=colors[idx], markersize=4)
        ax.fill(angles, values, alpha=0.08, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_labels, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Model Comparison Radar Chart', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'model_radar_chart.png'), dpi=PLOT_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Radar chart saved to {images_dir}/model_radar_chart.png")


def generate_confidence_interval_plot(results_df, images_dir: str):
    """
    Generates bar chart with error bars showing CV metric ± standard deviation.
    """
    os.makedirs(images_dir, exist_ok=True)
    sns.set_style(PLOT_STYLE)

    metrics_to_plot = [
        ('CV_PR_AUC_Mean', 'CV_PR_AUC_Std', 'PR-AUC'),
        ('CV_ROC_AUC_Mean', 'CV_ROC_AUC_Std', 'ROC-AUC')
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (mean_col, std_col, title) in enumerate(metrics_to_plot):
        if mean_col not in results_df.columns:
            continue
        df_sorted = results_df.sort_values(mean_col, ascending=True)
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_sorted)))

        axes[idx].barh(
            df_sorted['Model'], df_sorted[mean_col],
            xerr=df_sorted[std_col], color=colors,
            capsize=4, edgecolor='white', linewidth=0.5
        )
        axes[idx].set_title(f'Cross-Validation {title} (± 1σ)', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel(f'{title} Score')

        # Annotate values
        for i, (mean, std) in enumerate(zip(df_sorted[mean_col], df_sorted[std_col])):
            axes[idx].text(mean + std + 0.01, i, f'{mean:.4f}', va='center', fontsize=8)

    plt.suptitle('Cross-Validation Metrics with Confidence Intervals',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'cv_confidence_intervals.png'), dpi=PLOT_DPI)
    plt.close()
    logger.info(f"Confidence interval plot saved to {images_dir}/cv_confidence_intervals.png")


def generate_feature_importance_comparison(trained_models, X_test, images_dir: str):
    """
    Generates side-by-side feature importance comparison for tree-based models.
    """
    os.makedirs(images_dir, exist_ok=True)
    sns.set_style(PLOT_STYLE)

    tree_models = {n: m for n, m in trained_models.items()
                   if hasattr(m, 'feature_importances_')}

    if not tree_models:
        return

    n_models = min(len(tree_models), 4)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 6))
    if n_models == 1:
        axes = [axes]

    palettes = ['Blues_r', 'Greens_r', 'Oranges_r', 'Purples_r']
    for idx, (name, model) in enumerate(list(tree_models.items())[:n_models]):
        imp = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False).head(10)

        sns.barplot(x='Importance', y='Feature', data=imp, palette=palettes[idx], ax=axes[idx])
        axes[idx].set_title(name, fontsize=11, fontweight='bold')

    plt.suptitle('Feature Importance Comparison (Top 10)', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(images_dir, 'feature_importance_comparison.png'), dpi=PLOT_DPI)
    plt.close()
    logger.info(f"Feature importance comparison saved to {images_dir}/")


def run_shap_analysis(model, X_test, images_dir: str):
    """Generates SHAP summary plot and waterfall plot if shap is installed."""
    if not SHAP_AVAILABLE:
        logger.info("SHAP package not installed. Skipping SHAP analysis.")
        return

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test.iloc[:SHAP_SAMPLE_SIZE])

        # 1. Summary Plot (beeswarm)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test.iloc[:SHAP_SAMPLE_SIZE], show=False)
        plt.title('SHAP Feature Contribution Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'shap_summary.png'), dpi=PLOT_DPI)
        plt.close()
        logger.info(f"SHAP summary plot saved to {images_dir}/shap_summary.png")

        # 2. SHAP Bar Plot (mean absolute)
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test.iloc[:SHAP_SAMPLE_SIZE],
                         plot_type='bar', show=False)
        plt.title('SHAP Mean Absolute Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(images_dir, 'shap_bar.png'), dpi=PLOT_DPI)
        plt.close()
        logger.info(f"SHAP bar plot saved to {images_dir}/shap_bar.png")

        # 3. Waterfall Plot for a single fraud prediction
        # Find a true positive prediction
        y_pred_proba = model.predict_proba(X_test.iloc[:SHAP_SAMPLE_SIZE])[:, 1]
        fraud_indices = np.where(y_pred_proba > 0.5)[0]
        if len(fraud_indices) > 0:
            sample_idx = fraud_indices[0]
            if isinstance(shap_values, list):
                sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
            else:
                sv = shap_values

            # Build SHAP Explanation object for waterfall
            explanation = shap.Explanation(
                values=sv[sample_idx],
                base_values=explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value,
                data=X_test.iloc[sample_idx].values,
                feature_names=X_test.columns.tolist()
            )

            plt.figure(figsize=(10, 7))
            shap.waterfall_plot(explanation, show=False)
            plt.title('SHAP Waterfall — Single Fraud Prediction', fontsize=12, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(images_dir, 'shap_waterfall.png'), dpi=PLOT_DPI)
            plt.close()
            logger.info(f"SHAP waterfall plot saved to {images_dir}/shap_waterfall.png")

    except Exception as e:
        logger.warning(f"SHAP analysis warning: {e}")
